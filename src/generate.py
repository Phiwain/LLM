"""Interactive text generation with the trained MoE model."""
import argparse
import torch
import sentencepiece as spm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.model.model import MoELLM
from src.evaluate import generate

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
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
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config.from_yaml(args.config)
    model = MoELLM(cfg.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    print("MoE LLM Interactive Generation (type 'quit' to exit)")
    print(f"Model: ~{cfg.model.n_params / 1e6:.0f}M params | {cfg.model.n_experts} experts | top-{cfg.model.top_k}")
    print("=" * 60)

    while True:
        prompt = input("\nPrompt> ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue
        out = generate(model, sp, prompt, args.max_tokens, args.temperature, device=device)
        print(f"Output: {prompt}{out}")


if __name__ == "__main__":
    main()
