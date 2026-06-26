"""Training loop for the MoE LLM in PyTorch (A100/CUDA optimized)."""
import argparse
import math
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.model.model import MoELLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_lr(step, warmup, max_steps, base_lr):
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def load_data(processed_dir: Path):
    train = np.load(processed_dir / "train.npy")
    val = np.load(processed_dir / "val.npy")
    return train, val


def get_batch(data: np.ndarray, batch_size: int, device: torch.device):
    idx = np.random.randint(0, data.shape[0], size=batch_size)
    batch = data[idx]
    x = torch.from_numpy(batch[:, :-1].astype(np.int64)).to(device)
    y = torch.from_numpy(batch[:, 1:].astype(np.int64)).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, val_data, batch_size, device, dtype):
    model.eval()
    n_eval = min(500, val_data.shape[0])
    idx = np.random.choice(val_data.shape[0], size=n_eval, replace=False)
    total_loss = 0.0
    n_batches = 0
    for i in range(0, n_eval, batch_size):
        batch = val_data[idx[i : i + batch_size]]
        x = torch.from_numpy(batch[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(batch[:, 1:].astype(np.int64)).to(device)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def train(cfg: Config):
    mcfg = cfg.model
    tcfg = cfg.training

    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (tcfg.use_bf16 and device.type == "cuda") else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    # Data
    processed_dir = PROJECT_ROOT / "data" / "processed"
    train_data, val_data = load_data(processed_dir)
    print(f"Train sequences: {train_data.shape[0]:,}")
    print(f"Val sequences:   {val_data.shape[0]:,}")

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Model
    model = MoELLM(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.0f}M)")

    # Resume from latest checkpoint
    start_step = 0
    ckpts = sorted(checkpoints_dir.glob("checkpoint_*.pt"))
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming from {latest}...")
        ckpt = torch.load(str(latest), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"  Loaded step {start_step}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        betas=(0.9, 0.95),
        fused=True if device.type == "cuda" else False,
    )

    # Scaler for mixed precision (not needed for bf16 but harmless)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None

    # WandB
    use_wandb = tcfg.use_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="moe-bilingual-llm",
            config={
                "model": mcfg.__dict__,
                "training": tcfg.__dict__,
            },
        )

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_file = open(checkpoints_dir / "train_log.txt", "a")

    step = 0
    total_tokens = 0
    start_time = time.time()
    model.train()

    pbar = tqdm(total=tcfg.max_steps, desc="Training")
    while step < tcfg.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_aux = 0.0

        for _ in range(tcfg.grad_accum_steps):
            x, y = get_batch(train_data, tcfg.batch_size, device)

            with torch.amp.autocast("cuda", dtype=dtype):
                logits, aux = model(x)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )
                loss = ce + tcfg.aux_loss_weight * aux

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += ce.item()
            accum_aux += aux.item() if aux.dim() == 0 else aux.mean().item()

        # Gradient clipping
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

        # LR step
        lr = get_lr(step + 1, tcfg.warmup_steps, tcfg.max_steps, tcfg.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step += 1
        avg_loss = accum_loss / tcfg.grad_accum_steps
        avg_aux = accum_aux / tcfg.grad_accum_steps
        total_tokens += tcfg.batch_size * (mcfg.max_seq_len - 1) * tcfg.grad_accum_steps
        elapsed = time.time() - start_time
        tps = total_tokens / elapsed if elapsed > 0 else 0

        pbar.update(1)
        if step % tcfg.log_interval == 0:
            pbar.set_postfix_str(
                f"loss {avg_loss:.4f} | aux {avg_aux:.4f} | lr {lr:.2e} | tok/s {tps:.0f}"
            )
            if use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/aux_loss": avg_aux,
                    "train/lr": lr,
                    "train/tokens_per_sec": tps,
                    "train/step": step,
                })

        if step % tcfg.eval_interval == 0:
            eval_loss = evaluate(model, val_data, tcfg.batch_size, device, dtype)
            ppl = math.exp(min(20, eval_loss))
            print(f"\n  >> eval loss: {eval_loss:.4f} | perplexity: {ppl:.2f}")
            log_file.write(
                f"step {step} | train_loss {avg_loss:.4f} | eval_loss {eval_loss:.4f} | ppl {ppl:.2f}\n"
            )
            log_file.flush()
            if use_wandb:
                wandb.log({"eval/loss": eval_loss, "eval/perplexity": ppl})

        if step % tcfg.checkpoint_interval == 0:
            ckpt_path = checkpoints_dir / f"checkpoint_{step:06d}.pt"
            torch.save({"model": model.state_dict(), "step": step}, str(ckpt_path))
            # Clean old checkpoints, keep only the latest 2
            ckpts = sorted(checkpoints_dir.glob("checkpoint_*.pt"))
            for old in ckpts[:-2]:
                old.unlink()
            print(f"  >> Checkpoint: {ckpt_path}")

    pbar.close()

    # Final save
    final = checkpoints_dir / "final.pt"
    torch.save({"model": model.state_dict(), "step": step}, str(final))
    print(f"Final model: {final}")
    log_file.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    train(cfg)
