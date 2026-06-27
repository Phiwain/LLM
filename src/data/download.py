"""Download and subsample CulturaX EN+FR from Hugging Face (gated, needs HF_TOKEN)."""
import sys
import os
import argparse
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def download_subset(lang: str, n_docs: int, out_dir: Path):
    """Stream CulturaX for a given language and save n_docs to a text file."""
    out_file = out_dir / f"{lang}.txt"
    if out_file.exists():
        with open(out_file, "r") as f:
            existing = sum(1 for _ in f)
        if existing >= n_docs:
            print(f"[{lang}] Already have {existing} docs in {out_file}, skipping.")
            return out_file

    print(f"[{lang}] Streaming CulturaX, collecting {n_docs} documents...")
    ds = load_dataset(
        "uonlp/CulturaX",
        lang,
        split="train",
        streaming=True,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_file, "w") as f:
        for doc in tqdm(ds, desc=f"Downloading {lang}"):
            text = doc.get("text", "").strip()
            if not text or len(text) < 200:
                continue
            clean = text.replace("\n", " ").replace("\r", " ")
            f.write(clean + "\n")
            count += 1
            if count >= n_docs:
                break

    print(f"[{lang}] Saved {count} documents to {out_file}")
    return out_file


def sample_for_tokenizer(lang_files: list[Path], n_docs: int, out_path: Path):
    """Create a combined sample file for tokenizer training."""
    random.seed(42)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    per_file = n_docs // len(lang_files)
    for f in lang_files:
        with open(f, "r") as fh:
            sampled = []
            for i, line in enumerate(fh):
                if i >= per_file * 3:
                    break
                sampled.append(line)
            lines.extend(random.sample(sampled, min(per_file, len(sampled))))

    random.shuffle(lines)
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"Tokenizer training corpus: {len(lines)} docs -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="both", choices=["en", "fr", "both"])
    parser.add_argument("--n-docs", type=int, default=750000)
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    lang_files = []
    if args.lang in ("en", "both"):
        lang_files.append(download_subset("en", args.n_docs, out_dir))
    if args.lang in ("fr", "both"):
        lang_files.append(download_subset("fr", args.n_docs, out_dir))

    tok_sample = out_dir.parent / "tokenizer_sample.txt"
    sample_for_tokenizer(lang_files, n_docs=200000, out_path=tok_sample)
