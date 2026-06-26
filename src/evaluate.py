"""Evaluate the trained MoE model: bilingual perplexity and generation samples."""
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.model.model import MoELLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROMPTS_EN = [
    "The capital of France is",
    "In machine learning, a transformer is",
    "The weather today is",
    "Artificial intelligence has",
]

PROMPTS_FR = [
    "La capitale de la France est",
    "En apprentissage automatique, un transformeur est",
    "Le temps aujourd'hui est",
    "L'intelligence artificielle a",
]


@torch.no_grad()
def eval_perplexity(model, data, batch_size, device, dtype):
    n = min(500, data.shape[0])
    idx = np.random.choice(data.shape[0], size=n, replace=False)
    total = 0.0
    n_batches = 0
    for i in range(0, n, batch_size):
        batch = data[idx[i : i + batch_size]]
        x = torch.from_numpy(batch[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(batch[:, 1:].astype(np.int64)).to(device)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item()
        n_batches += 1
    avg = total / max(1, n_batches)
    return avg, math.exp(min(20, avg))


@torch.no_grad()
def generate(model, sp, prompt: str, max_tokens: int = 100, temperature: float = 0.8, device="cuda"):
    ids = [1] + sp.encode(prompt, out_type=int)  # BOS + prompt
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        if tokens.shape[1] > model.cfg.max_seq_len:
            tokens = tokens[:, -model.cfg.max_seq_len:]
        logits, _ = model(tokens)
        next_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        if next_tok.item() == 2:  # EOS
            break
        tokens = torch.cat([tokens, next_tok], dim=1)

    out_ids = tokens[0, len(ids):].tolist()
    return sp.decode(out_ids)


def main(cfg: Config, checkpoint: Path, tokenizer_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = MoELLM(cfg.model).to(device)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    # Perplexity
    processed = PROJECT_ROOT / "data" / "processed"
    if (processed / "val_en.npy").exists() and (processed / "val_fr.npy").exists():
        val_en = np.load(processed / "val_en.npy")
        val_fr = np.load(processed / "val_fr.npy")
        loss_en, ppl_en = eval_perplexity(model, val_en, cfg.training.batch_size, device, dtype)
        loss_fr, ppl_fr = eval_perplexity(model, val_fr, cfg.training.batch_size, device, dtype)
        print(f"\n{'='*60}")
        print(f"Bilingual Evaluation")
        print(f"{'='*60}")
        print(f"English  — loss: {loss_en:.4f} | perplexity: {ppl_en:.2f}")
        print(f"Français — loss: {loss_fr:.4f} | perplexity: {ppl_fr:.2f}")
        print(f"{'='*60}")

    # Generation
    print("\n--- English Generation ---")
    for prompt in PROMPTS_EN:
        out = generate(model, sp, prompt, device=device)
        print(f"[EN] {prompt} -> {prompt}{out}")

    print("\n--- French Generation ---")
    for prompt in PROMPTS_FR:
        out = generate(model, sp, prompt, device=device)
        print(f"[FR] {prompt} -> {prompt}{out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "final.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(PROJECT_ROOT / "tokenizer" / "bpe.model"),
    )
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    main(cfg, Path(args.checkpoint), Path(args.tokenizer))
