"""Tokenize raw text data and pack into fixed-length sequences for training.

Uses multiprocessing for fast tokenization.
"""
import sys
import argparse
import numpy as np
import sentencepiece as spm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EOS_ID = 2

# Global tokenizer (initialized per worker)
_sp = None


def _init_worker(model_path):
    global _sp
    _sp = spm.SentencePieceProcessor(model_file=model_path)


def _tokenize_line(line):
    global _sp
    line = line.strip()
    if not line:
        return []
    ids = _sp.encode(line, out_type=int)
    ids.append(EOS_ID)
    return ids


def tokenize_files(input_files, sp_model_path, seq_len, val_ratio, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count total lines for progress
    total_lines = 0
    for f in input_files:
        with open(f, "r") as fh:
            for _ in fh:
                total_lines += 1
    print(f"Total documents: {total_lines:,}")

    # Tokenize with multiprocessing
    all_tokens = []
    n_workers = min(cpu_count(), 16)

    with Pool(n_workers, initializer=_init_worker, initargs=(str(sp_model_path),)) as pool:
        for f in input_files:
            print(f"Tokenizing {f}...")
            with open(f, "r") as fh:
                lines = fh.readlines()
            results = list(
                tqdm(
                    pool.imap(_tokenize_line, lines, chunksize=1000),
                    total=len(lines),
                    desc=f"Tokenizing {f.name}",
                )
            )
            for ids in results:
                all_tokens.extend(ids)

    tokens = np.array(all_tokens, dtype=np.uint16)
    del all_tokens

    print(f"Total tokens: {len(tokens):,}")

    # Pack into sequences of seq_len
    n_seqs = len(tokens) // seq_len
    tokens = tokens[: n_seqs * seq_len]
    sequences = tokens.reshape(n_seqs, seq_len)

    print(f"Packed into {n_seqs:,} sequences of {seq_len} tokens")

    # Shuffle before split
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_seqs)
    sequences = sequences[perm]

    n_val = max(1, int(n_seqs * val_ratio))
    val_seqs = sequences[:n_val]
    train_seqs = sequences[n_val:]

    # Split val into EN and FR halves for bilingual eval
    np.save(out_dir / "train.npy", train_seqs)
    np.save(out_dir / "val.npy", val_seqs)
    np.save(out_dir / "val_en.npy", val_seqs[: n_val // 2])
    np.save(out_dir / "val_fr.npy", val_seqs[n_val // 2 :])

    print(f"Train: {train_seqs.shape}, Val: {val_seqs.shape}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(PROJECT_ROOT / "tokenizer" / "bpe.model"),
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    input_files = [raw_dir / "en.txt", raw_dir / "fr.txt"]
    input_files = [f for f in input_files if f.exists()]

    if not input_files:
        print("No raw data files found. Run download.py first.")
        sys.exit(1)

    tokenize_files(
        input_files=input_files,
        sp_model_path=Path(args.tokenizer),
        seq_len=args.seq_len,
        val_ratio=args.val_ratio,
        out_dir=Path(args.out_dir),
    )
