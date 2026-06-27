#!/usr/bin/env python3
"""Chat interactif avec le MoE 1B — charge le modèle et le tokenizer."""
import sys
import torch
import sentencepiece as spm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.model.model import MoELLM


def load_model(checkpoint_path, config_path, device="cpu"):
    cfg = Config.from_yaml(config_path)
    model = MoELLM(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, cfg


@torch.inference_mode()
def generate(model, sp, prompt, max_tokens=200, temperature=0.8, device="cpu"):
    bos_id = 1
    eos_id = 2
    max_seq_len = model.cfg.max_seq_len

    ids = [bos_id] + sp.encode(prompt)
    out_ids = []

    for _ in range(max_tokens):
        ctx = ids[-max_seq_len:]
        tokens = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(tokens)
        logits = logits[0, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id == eos_id:
            break
        ids.append(next_id)
        out_ids.append(next_id)
        yield sp.decode([next_id])


def main():
    checkpoint = PROJECT_ROOT / "checkpoints" / "final.pt"
    config = PROJECT_ROOT / "config.yaml"
    tokenizer = PROJECT_ROOT / "tokenizer" / "bpe_culturax.model"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("Chargement du modèle...")
    model, cfg = load_model(checkpoint, config, device)
    print(f"Modèle chargé: {cfg.model.n_params:,} params")

    print("Chargement du tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    print(f"Tokenizer: vocab={sp.get_piece_size()}")

    print("\n" + "=" * 60)
    print("  phiwAIn — MoE 1B (959M params)")
    print("  Tape 'quit' ou Ctrl+C pour quitter")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("vous> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        print("phiwAIn> ", end="", flush=True)
        for token in generate(model, sp, prompt, max_tokens=200, temperature=0.8, device=device):
            print(token, end="", flush=True)
        print("\n")

    print("\nAu revoir !")


if __name__ == "__main__":
    main()
