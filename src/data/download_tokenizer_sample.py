"""Download a Wikipedia sample (EN+FR) to train the tokenizer."""
from datasets import load_dataset
from pathlib import Path
import os

OUT = Path(__file__).resolve().parent.parent / "data" / "tokenizer_sample.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

texts = []
for lang in ["en", "fr"]:
    print(f"Loading Wikipedia {lang}...")
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    count = 0
    for doc in ds:
        t = doc.get("text", "").strip()
        if t and len(t) > 50:
            texts.append(t)
            count += 1
            if count >= 100000:
                break
    print(f"  {count} docs from {lang}")

print(f"Writing {len(texts)} docs to {OUT}...")
with open(OUT, "w") as f:
    for t in texts:
        clean = t.replace("\n", " ").replace("\r", " ")
        f.write(clean + "\n")

size_mb = os.path.getsize(OUT) / 1e6
print(f"Done! {size_mb:.1f} MB, {len(texts)} docs")
