#!/usr/bin/env python3
"""SFT (Supervised Fine-Tuning) pour le MoE 1B — transforme le base model en chatbot."""
import sys
import json
import math
import time
import random
import numpy as np
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

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "checkpoint_021000.pt"
CONFIG = PROJECT_ROOT / "config.yaml"
TOKENIZER = PROJECT_ROOT / "tokenizer" / "bpe.model"
OUTPUT = PROJECT_ROOT / "checkpoints" / "sft_model.pt"

BOS_ID = 1
EOS_ID = 2
MAX_SEQ_LEN = 1024
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 5e-5
EPOCHS = 3
MAX_EXAMPLES = 2000


def format_prompt(instruction, input_text, response):
    if input_text:
        prompt = f"User: {instruction}\n{input_text}\nAssistant: "
    else:
        prompt = f"User: {instruction}\nAssistant: "
    return prompt, response


def prepare_dataset(sp):
    print("Téléchargement du dataset d'instructions...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    
    examples = []
    for i, row in enumerate(ds):
        if i >= MAX_EXAMPLES:
            break
        prompt, response = format_prompt(
            row["instruction"], row.get("input", ""), row["output"]
        )
        prompt_ids = sp.encode(prompt)
        response_ids = sp.encode(response)
        
        input_ids = [BOS_ID] + prompt_ids + response_ids + [EOS_ID]
        if len(input_ids) > MAX_SEQ_LEN:
            input_ids = input_ids[:MAX_SEQ_LEN]
        
        labels = input_ids.copy()
        prompt_len = 1 + len(prompt_ids)
        for j in range(prompt_len):
            labels[j] = -100
        
        examples.append({"input_ids": input_ids, "labels": labels})
    
    print(f"  {len(examples)} examples préparés")
    return examples


def collate(batch, sp):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    labels = []
    attention_mask = []
    
    for b in batch:
        padding = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [0] * padding)
        labels.append(b["labels"] + [-100] * padding)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * padding)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Chargement du tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    
    print("Chargement du modèle...")
    cfg = Config.from_yaml(str(CONFIG))
    model = MoELLM(cfg.model)
    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.train()
    print(f"  Modèle chargé: {cfg.model.n_params:,} params (step {ckpt['step']})")
    
    examples = prepare_dataset(sp)
    random.shuffle(examples)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
    )
    
    scaler = torch.amp.GradScaler("mps" if device.type == "mps" else "cpu")
    total_steps = (len(examples) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"\nEntraînement SFT: {len(examples)} examples × {EPOCHS} epochs = {total_steps} steps\n")
    
    pbar = tqdm(total=total_steps, desc="SFT")
    step = 0
    
    for epoch in range(EPOCHS):
        for i in range(0, len(examples), BATCH_SIZE):
            batch = examples[i:i + BATCH_SIZE]
            if len(batch) < BATCH_SIZE:
                continue
            
            collated = collate(batch, sp)
            input_ids = collated["input_ids"].to(device)
            labels = collated["labels"].to(device)
            
            with torch.amp.autocast(device.type, dtype=torch.float16):
                logits, aux = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.update(1)
                
                if pbar.n % 10 == 0:
                    pbar.set_postfix_str(f"loss {loss.item():.4f}")
            
            step += 1
    
    pbar.close()
    
    print(f"\nSauvegarde du modèle SFT: {OUTPUT}")
    torch.save({"model": model.state_dict(), "step": ckpt["step"], "sft": True}, str(OUTPUT))
    print("Done!")


if __name__ == "__main__":
    train()
