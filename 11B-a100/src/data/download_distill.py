"""Download distillation datasets from Hugging Face — Claude Fable 5, Mythos, GLM-5.2.

These are instruction/chat datasets (SFT quality). For pretraining, we convert
them to continuous text. For SFT, we keep them in chat format separately.

Datasets:
  1. armand0e/claude-fable-5-claude-code — Claude Code agent traces (63 rows)
  2. WithinUsAI/claude_mythos_distilled_25k — Claude Mythos distillation (25k rows)
  3. AletheiaResearch/GLM-5.2-Agent — GLM-5.2 agent traces (284 rows)
"""
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def extract_text_from_messages(messages):
    """Convert a list of chat messages into continuous text for pretraining.

    Format: "User: {content}\nAssistant: {content}\n" for each turn.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "system":
            parts.append(f"System: {content}")
    return "\n".join(parts)


def download_claude_fable5(out_dir):
    """Claude Fable 5 — Claude Code agent traces."""
    out_pretrain = out_dir / "distill_fable5_pretrain.txt"
    out_sft = out_dir / "distill_fable5_sft.jsonl"

    if out_pretrain.exists() and out_sft.exists():
        print("[fable5] Already downloaded, skipping.")
        return [out_pretrain, out_sft]

    print("[fable5] Downloading armand0e/claude-fable-5-claude-code...")
    try:
        ds = load_dataset("armand0e/claude-fable-5-claude-code", split="train")
    except Exception as e:
        print(f"[fable5] Failed: {e}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    n_pretrain = 0
    n_sft = 0

    with open(out_pretrain, "w") as f_pre, open(out_sft, "w") as f_sft:
        for row in tqdm(ds, desc="fable5"):
            messages = row.get("messages", [])
            if not messages:
                # Some rows have a single "message" field
                msg = row.get("message", "")
                if msg:
                    messages = [{"role": "assistant", "content": msg}]

            if not messages:
                continue

            # Pretraining format: continuous text
            text = extract_text_from_messages(messages)
            if len(text) > 100:
                clean = text.replace("\n", " ").replace("\r", " ")
                f_pre.write(clean + "\n")
                n_pretrain += 1

            # SFT format: keep structured
            f_sft.write(json.dumps({"messages": messages, "source": "fable5"}) + "\n")
            n_sft += 1

    print(f"[fable5] Saved {n_pretrain} pretrain docs, {n_sft} SFT examples")
    return [out_pretrain, out_sft]


def download_mythos_distilled(out_dir):
    """Claude Mythos Distilled 25K — high quality distillation data."""
    out_pretrain = out_dir / "distill_mythos_pretrain.txt"
    out_sft = out_dir / "distill_mythos_sft.jsonl"

    if out_pretrain.exists() and out_sft.exists():
        print("[mythos] Already downloaded, skipping.")
        return [out_pretrain, out_sft]

    print("[mythos] Downloading WithinUsAI/claude_mythos_distilled_25k...")
    try:
        ds = load_dataset("WithinUsAI/claude_mythos_distilled_25k", split="train")
    except Exception as e:
        print(f"[mythos] Failed: {e}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    n_pretrain = 0
    n_sft = 0

    with open(out_pretrain, "w") as f_pre, open(out_sft, "w") as f_sft:
        for row in tqdm(ds, desc="mythos"):
            messages = row.get("messages", [])
            if not messages:
                continue

            # Pretraining format
            text = extract_text_from_messages(messages)
            if len(text) > 100:
                clean = text.replace("\n", " ").replace("\r", " ")
                f_pre.write(clean + "\n")
                n_pretrain += 1

            # SFT format with metadata
            sft_row = {
                "messages": messages,
                "category": row.get("category", "unknown"),
                "source": "mythos_distilled",
            }
            f_sft.write(json.dumps(sft_row) + "\n")
            n_sft += 1

    print(f"[mythos] Saved {n_pretrain} pretrain docs, {n_sft} SFT examples")
    return [out_pretrain, out_sft]


def download_glm52_agent(out_dir):
    """GLM-5.2 Agent — coding agent traces."""
    out_pretrain = out_dir / "distill_glm52_pretrain.txt"
    out_sft = out_dir / "distill_glm52_sft.jsonl"

    if out_pretrain.exists() and out_sft.exists():
        print("[glm52] Already downloaded, skipping.")
        return [out_pretrain, out_sft]

    print("[glm52] Downloading AletheiaResearch/GLM-5.2-Agent...")
    try:
        ds = load_dataset("AletheiaResearch/GLM-5.2-Agent", split="train")
    except Exception as e:
        print(f"[glm52] Failed: {e}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    n_pretrain = 0
    n_sft = 0

    with open(out_pretrain, "w") as f_pre, open(out_sft, "w") as f_sft:
        for row in tqdm(ds, desc="glm52"):
            messages = row.get("messages", [])
            if not messages:
                continue

            # Pretraining format
            text = extract_text_from_messages(messages)
            if len(text) > 100:
                clean = text.replace("\n", " ").replace("\r", " ")
                f_pre.write(clean + "\n")
                n_pretrain += 1

            # SFT format
            f_sft.write(json.dumps({"messages": messages, "source": "glm52_agent"}) + "\n")
            n_sft += 1

    print(f"[glm52] Saved {n_pretrain} pretrain docs, {n_sft} SFT examples")
    return [out_pretrain, out_sft]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download distillation datasets for 11B MoE")
    parser.add_argument(
        "--sources", type=str, nargs="+",
        default=["all"],
        choices=["fable5", "mythos", "glm52", "all"],
        help="Which distillation datasets to download",
    )
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "data" / "distill"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    sources = args.sources
    if "all" in sources:
        sources = ["fable5", "mythos", "glm52"]

    print(f"Downloading distillation sources: {sources}")
    print(f"Output: {out_dir}")
    print()

    all_files = []
    if "fable5" in sources:
        all_files.extend(download_claude_fable5(out_dir))
    if "mythos" in sources:
        all_files.extend(download_mythos_distilled(out_dir))
    if "glm52" in sources:
        all_files.extend(download_glm52_agent(out_dir))

    # Summary
    print()
    print("=" * 60)
    print("DISTILLATION DATASETS SUMMARY")
    print("=" * 60)
    for f in sorted(out_dir.glob("*.txt")):
        with open(f, "r") as fh:
            n = sum(1 for _ in fh)
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {n:>8,} docs  {size_mb:>8.1f} MB")

    sft_files = sorted(out_dir.glob("*.jsonl"))
    if sft_files:
        print()
        print("SFT files (for fine-tuning phase):")
        for f in sft_files:
            with open(f, "r") as fh:
                n = sum(1 for _ in fh)
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:<40} {n:>8,} examples  {size_mb:>8.1f} MB")
    print()
