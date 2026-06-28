#!/usr/bin/env python3
"""SFT on A100 — EN+FR instruction tuning for MoE 1B."""
import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.model.model import MoELLM

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "checkpoint_024000.pt"
CONFIG = PROJECT_ROOT / "config.yaml"
TOKENIZER = PROJECT_ROOT / "tokenizer" / "bpe.model"
OUTPUT = PROJECT_ROOT / "checkpoints" / "sft_model.pt"

BOS_ID = 1
EOS_ID = 2
MAX_SEQ_LEN = 1024
BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 3e-5
MAX_STEPS = 32000


def format_example(instruction, input_text, response):
    if input_text:
        prompt = f"User: {instruction}\n{input_text}\nAssistant: {response}{chr(EOS_ID)}"
    else:
        prompt = f"User: {instruction}\nAssistant: {response}{chr(EOS_ID)}"
    return prompt


def prepare_dataset(sp):
    examples = []

    # EN: Alpaca
    print("Chargement Alpaca (EN)...")
    ds_en = load_dataset("tatsu-lab/alpaca", split="train")
    for row in ds_en:
        text = format_example(row["instruction"], row.get("input", ""), row["output"])
        ids = sp.encode(text)
        if len(ids) <= MAX_SEQ_LEN:
            prompt_len = len(sp.encode(
                f"User: {row['instruction']}\n" + (f"{row['input']}\n" if row.get("input") else "") + "Assistant: "
            ))
            examples.append({"ids": ids, "prompt_len": prompt_len})

    # FR: French Alpaca 55k
    print("Chargement French Alpaca 55k (FR)...")
    try:
        ds_fr = load_dataset("jpacifico/French-Alpaca-dataset-Instruct-55K", split="train")
        count = 0
        for row in ds_fr:
            text = format_example(row["instruction"], row.get("input", ""), row["output"])
            ids = sp.encode(text)
            if len(ids) <= MAX_SEQ_LEN:
                prompt_len = len(sp.encode(
                    f"User: {row['instruction']}\n" + (f"{row['input']}\n" if row.get("input") else "") + "Assistant: "
                ))
                examples.append({"ids": ids, "prompt_len": prompt_len})
                count += 1
        print(f"  {count} exemples FR ajoutés")
    except Exception as e:
        print(f"  French Alpaca 55k indisponible: {e}")
    except:
        print("  Vigogne non disponible, fallback Alpaca uniquement")

    random.shuffle(examples)
    print(f"Total examples: {len(examples)}")
    return examples


def collate(batch):
    max_len = max(len(ex["ids"]) for ex in batch)
    pad_id = 0
    input_ids = []
    labels = []

    for ex in batch:
        ids = ex["ids"]
        padding = max_len - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        label = ids.copy()
        for j in range(ex["prompt_len"]):
            label[j] = -100
        labels.append(label + [-100] * padding)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    print(f"Tokenizer: vocab={sp.get_piece_size()}")

    print("Chargement du modèle...")
    cfg = Config.from_yaml(str(CONFIG))
    model = MoELLM(cfg.model)
    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.train()
    print(f"  {cfg.model.n_params:,} params (step {ckpt['step']})")

    examples = prepare_dataset(sp)
    dtype = torch.bfloat16

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95), fused=True
    )
    scaler = torch.amp.GradScaler("cuda")

    pbar = tqdm(total=MAX_STEPS, desc="SFT")
    step = 0
    total_tokens = 0
    start_time = time.time()
    accum_loss = 0.0

    while step < MAX_STEPS:
        batch = random.sample(examples, min(BATCH_SIZE, len(examples)))
        collated = collate(batch)
        input_ids = collated["input_ids"].to(device)
        labels = collated["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=dtype):
            logits, aux = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            accum_loss += loss.item()
            total_tokens += BATCH_SIZE * (MAX_SEQ_LEN - 1) * GRAD_ACCUM
            pbar.update(1)

            if pbar.n % 50 == 0:
                pbar.set_postfix_str(f"loss {accum_loss / (pbar.n or 1):.4f}")

        step += 1

    pbar.close()

    print(f"\nSauvegarde: {OUTPUT}")
    torch.save({"model": model.state_dict(), "step": step, "sft": True}, str(OUTPUT))
    print("Done!")


if __name__ == "__main__":
    import time
    train()
