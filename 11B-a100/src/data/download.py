"""Download datasets for the 11B MoE — CODE-FIRST + minimal FR/EN base.

Priorities:
  1. CODE (bulk):     codeparrot/github-code (~5M files, Python+JS+Go+Rust+C++)
  2. CODE (instruct): PawanKrd/claude-fable-5-code (603, Claude code+math)
  3. CODE (instruct): HuggingFaceH4/CodeAlpaca_20K (20k, code instruction)
  4. DISTILL:          WithinUsAI/claude_mythos_distilled_25k (25k, distillation)
  5. DISTILL:          armand0e/claude-fable-5-claude-code (63, agent traces)
  6. DISTILL:          AletheiaResearch/GLM-5.2-Agent (284, GLM agent traces)
  7. BASE FR:          wikimedia/wikipedia FR (~500k articles)
  8. BASE EN:          wikimedia/wikipedia EN (~500k articles)

Ratio cible: ~70% code, 20% distillation, 10% base FR/EN
"""
import json
import sys
import argparse
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def extract_text_from_messages(messages):
    """Convert chat messages into continuous text for pretraining."""
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


# ---------- CODE (bulk) ----------

def download_github_code(n_docs, out_dir):
    """GitHub Code — bulk source code from GitHub (dev web + system languages)."""
    out_file = out_dir / "github_code.txt"
    if out_file.exists():
        with open(out_file, "r") as f:
            existing = sum(1 for _ in f)
        if existing >= n_docs:
            print(f"[github-code] Already have {existing} docs, skipping.")
            return out_file

    print(f"[github-code] Streaming, collecting {n_docs} code files...")
    try:
        ds = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train",
            languages=["Python", "JavaScript", "TypeScript", "HTML", "CSS", "PHP", "Ruby",
                        "Go", "Rust", "C", "C++", "Java", "Shell", "SQL"],
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[github-code] Failed: {e}")
        return None

    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_file, "w") as f:
        for doc in tqdm(ds, desc="github-code"):
            code = doc.get("code", "").strip()
            if not code or len(code) < 100:
                continue
            lang = doc.get("language", "")
            clean = code.replace("\n", " ").replace("\r", " ")
            f.write(clean + "\n")
            count += 1
            if count >= n_docs:
                break

    print(f"[github-code] Saved {count} code files to {out_file}")
    return out_file


def download_web_dev_bulk(n_docs, out_dir):
    """GitHub Code — HTML + CSS + PHP + JS only, for concentrated web dev data."""
    out_file = out_dir / "webdev_code.txt"
    if out_file.exists():
        with open(out_file, "r") as f:
            existing = sum(1 for _ in f)
        if existing >= n_docs:
            print(f"[webdev-code] Already have {existing} docs, skipping.")
            return out_file

    print(f"[webdev-code] Streaming HTML/CSS/JS/PHP from GitHub Code...")
    try:
        ds = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train",
            languages=["HTML", "CSS", "JavaScript", "TypeScript", "PHP"],
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[webdev-code] Failed: {e}")
        return None

    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_file, "w") as f:
        for doc in tqdm(ds, desc="webdev-code"):
            code = doc.get("code", "").strip()
            if not code or len(code) < 100:
                continue
            lang = doc.get("language", "")
            # Prefix with language tag so the model learns the context
            clean = f"[{lang}] " + code.replace("\n", " ").replace("\r", " ")
            f.write(clean + "\n")
            count += 1
            if count >= n_docs:
                break

    print(f"[webdev-code] Saved {count} web dev files to {out_file}")
    return out_file


# ---------- CODE (instruction) ----------

def download_web_dev_instructions(out_dir):
    """Web dev instruction datasets — HTML, CSS, JS, Python web frameworks."""
    out_pretrain = out_dir / "webdev_instructions_pretrain.txt"
    out_sft = out_dir / "webdev_instructions_sft.jsonl"

    if out_pretrain.exists() and out_sft.exists():
        print("[webdev] Already downloaded, skipping.")
        return [out_pretrain, out_sft]

    out_dir.mkdir(parents=True, exist_ok=True)
    n_pre = 0
    n_sft = 0

    datasets_to_try = [
        ("HuggingFaceH4/CodeAlpaca_20K", "train", None),  # already downloaded but we filter web-related
    ]

    print("[webdev] Downloading web dev instruction data...")

    # Use CodeAlpaca but filter for web-related prompts (HTML, CSS, JavaScript, React, Flask, Django)
    web_keywords = ["html", "css", "javascript", "react", "vue", "angular", "flask", "django",
                    "fastapi", "express", "node", "frontend", "backend", "web", "dom", "api",
                    "responsive", "bootstrap", "tailwind", "sass", "webpack"]

    try:
        ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
    except Exception as e:
        print(f"[webdev] CodeAlpaca failed: {e}")
        return []

    with open(out_pretrain, "w") as f_pre, open(out_sft, "w") as f_sft:
        for row in tqdm(ds, desc="webdev-filter"):
            prompt = row.get("prompt", "").strip()
            completion = row.get("completion", "").strip()
            if not prompt or not completion:
                continue
            prompt_lower = prompt.lower()
            if not any(kw in prompt_lower for kw in web_keywords):
                continue
            text = f"User: {prompt}\nAssistant: {completion}"
            clean = text.replace("\n", " ").replace("\r", " ")
            f_pre.write(clean + "\n")
            n_pre += 1
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            f_sft.write(json.dumps({"messages": messages, "source": "webdev_alpaca"}) + "\n")
            n_sft += 1

    print(f"[webdev] Saved {n_pre} web dev pretrain docs, {n_sft} SFT examples")
    return [out_pretrain, out_sft]


def download_fable5_code(out_dir):
    """PawanKrd/claude-fable-5-code — Claude Fable 5 coding + math (603 examples)."""
    out_pretrain = out_dir / "fable5_code_pretrain.txt"
    out_sft = out_dir / "fable5_code_sft.jsonl"

    if out_pretrain.exists() and out_sft.exists():
        print("[fable5-code] Already downloaded, skipping.")
        return [out_pretrain, out_sft]

    print("[fable5-code] Downloading PawanKrd/claude-fable-5-code...")
    try:
        ds = load_dataset("PawanKrd/claude-fable-5-code", split="train")
    except Exception as e:
        print(f"[fable5-code] Failed: {e}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    n_pre = 0
    n_sft = 0
    with open(out_pretrain, "w") as f_pre, open(out_sft, "w") as f_sft:
        for row in tqdm(ds, desc="fable5-code"):
            messages = row.get("messages", [])
            if not messages:
                continue
            text = extract_text_from_messages(messages)
            if len(text) > 100:
                clean = text.replace("\n", " ").replace("\r", " ")
                f_pre.write(clean + "\n")
                n_pre += 1
            f_sft.write(json.dumps({"messages": messages, "category": row.get("category", ""), "source": "fable5_code"}) + "\n")
            n_sft += 1

    print(f"[fable5-code] Saved {n_pre} pretrain docs, {n_sft} SFT examples")
    return [out_pretrain, out_sft]


def download_code_alpaca(out_dir):
    """HuggingFaceH4/CodeAlpaca_20K — 20k code instruction examples."""
    out_pretrain = out_dir / "code_alpaca_pretrain.txt"
    out_sft = out_dir / "code_alpaca_sft.jsonl"

    if out_pretrain.exists() and out_sft.exists():
        print("[code-alpaca] Already downloaded, skipping.")
        return [out_pretrain, out_sft]

    print("[code-alpaca] Downloading HuggingFaceH4/CodeAlpaca_20K...")
    try:
        ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
    except Exception as e:
        print(f"[code-alpaca] Failed: {e}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    n_pre = 0
    n_sft = 0
    with open(out_pretrain, "w") as f_pre, open(out_sft, "w") as f_sft:
        for row in tqdm(ds, desc="code-alpaca"):
            prompt = row.get("prompt", "").strip()
            completion = row.get("completion", "").strip()
            if not prompt or not completion:
                continue
            text = f"User: {prompt}\nAssistant: {completion}"
            clean = text.replace("\n", " ").replace("\r", " ")
            f_pre.write(clean + "\n")
            n_pre += 1
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            f_sft.write(json.dumps({"messages": messages, "source": "code_alpaca"}) + "\n")
            n_sft += 1

    print(f"[code-alpaca] Saved {n_pre} pretrain docs, {n_sft} SFT examples")
    return [out_pretrain, out_sft]


# ---------- DISTILLATION ----------

def download_mythos(out_dir):
    """WithinUsAI/claude_mythos_distilled_25k — Claude Mythos distillation."""
    out_pretrain = out_dir / "mythos_pretrain.txt"
    out_sft = out_dir / "mythos_sft.jsonl"

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
    n_pre = 0
    n_sft = 0
    with open(out_pretrain, "w") as f_pre, open(out_sft, "w") as f_sft:
        for row in tqdm(ds, desc="mythos"):
            messages = row.get("messages", [])
            if not messages:
                continue
            text = extract_text_from_messages(messages)
            if len(text) > 100:
                clean = text.replace("\n", " ").replace("\r", " ")
                f_pre.write(clean + "\n")
                n_pre += 1
            f_sft.write(json.dumps({"messages": messages, "category": row.get("category", ""), "source": "mythos"}) + "\n")
            n_sft += 1

    print(f"[mythos] Saved {n_pre} pretrain docs, {n_sft} SFT examples")
    return [out_pretrain, out_sft]


def download_fable5_agent(out_dir):
    """armand0e/claude-fable-5-claude-code — Claude Code agent traces."""
    out_pretrain = out_dir / "fable5_agent_pretrain.txt"
    out_sft = out_dir / "fable5_agent_sft.jsonl"

    if out_pretrain.exists() and out_sft.exists():
        print("[fable5-agent] Already downloaded, skipping.")
        return [out_pretrain, out_sft]

    print("[fable5-agent] Downloading armand0e/claude-fable-5-claude-code...")
    try:
        ds = load_dataset("armand0e/claude-fable-5-claude-code", split="train")
    except Exception as e:
        print(f"[fable5-agent] Failed: {e}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    n_pre = 0
    n_sft = 0
    with open(out_pretrain, "w") as f_pre, open(out_sft, "w") as f_sft:
        for row in tqdm(ds, desc="fable5-agent"):
            messages = row.get("messages", [])
            if not messages:
                msg = row.get("message", "")
                if msg:
                    messages = [{"role": "assistant", "content": msg}]
            if not messages:
                continue
            text = extract_text_from_messages(messages)
            if len(text) > 100:
                clean = text.replace("\n", " ").replace("\r", " ")
                f_pre.write(clean + "\n")
                n_pre += 1
            f_sft.write(json.dumps({"messages": messages, "source": "fable5_agent"}) + "\n")
            n_sft += 1

    print(f"[fable5-agent] Saved {n_pre} pretrain docs, {n_sft} SFT examples")
    return [out_pretrain, out_sft]


def download_glm52_agent(out_dir):
    """AletheiaResearch/GLM-5.2-Agent — GLM-5.2 coding agent traces."""
    out_pretrain = out_dir / "glm52_agent_pretrain.txt"
    out_sft = out_dir / "glm52_agent_sft.jsonl"

    if out_pretrain.exists() and out_sft.exists():
        print("[glm52-agent] Already downloaded, skipping.")
        return [out_pretrain, out_sft]

    print("[glm52-agent] Downloading AletheiaResearch/GLM-5.2-Agent...")
    try:
        ds = load_dataset("AletheiaResearch/GLM-5.2-Agent", split="train")
    except Exception as e:
        print(f"[glm52-agent] Failed: {e}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    n_pre = 0
    n_sft = 0
    with open(out_pretrain, "w") as f_pre, open(out_sft, "w") as f_sft:
        for row in tqdm(ds, desc="glm52-agent"):
            messages = row.get("messages", [])
            if not messages:
                continue
            text = extract_text_from_messages(messages)
            if len(text) > 100:
                clean = text.replace("\n", " ").replace("\r", " ")
                f_pre.write(clean + "\n")
                n_pre += 1
            f_sft.write(json.dumps({"messages": messages, "source": "glm52_agent"}) + "\n")
            n_sft += 1

    print(f"[glm52-agent] Saved {n_pre} pretrain docs, {n_sft} SFT examples")
    return [out_pretrain, out_sft]


# ---------- BASE FR/EN ----------

def download_wikipedia(lang, n_docs, out_dir):
    """Wikipedia — minimal base text (FR + EN)."""
    out_file = out_dir / f"wikipedia_{lang}.txt"
    if out_file.exists():
        with open(out_file, "r") as f:
            existing = sum(1 for _ in f)
        if existing >= n_docs:
            print(f"[wiki-{lang}] Already have {existing} docs, skipping.")
            return out_file

    print(f"[wiki-{lang}] Streaming Wikipedia, collecting {n_docs} documents...")
    config = f"20231101.{lang}"
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_file, "w") as f:
        for doc in tqdm(ds, desc=f"wiki-{lang}"):
            text = doc.get("text", "").strip()
            if not text or len(text) < 200:
                continue
            clean = text.replace("\n", " ").replace("\r", " ")
            f.write(clean + "\n")
            count += 1
            if count >= n_docs:
                break

    print(f"[wiki-{lang}] Saved {count} documents to {out_file}")
    return out_file


# ---------- MAIN ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for 11B MoE (code-first)")
    parser.add_argument(
        "--sources", type=str, nargs="+",
        default=["all"],
        choices=["github", "webdev", "fable5", "alpaca", "mythos", "fable5_agent", "glm52", "wiki", "all"],
    )
    parser.add_argument("--github-files", type=int, default=12000000, help="GitHub code files (system languages, ~7B tokens)")
    parser.add_argument("--webdev-files", type=int, default=3000000, help="Web dev files (HTML/CSS/JS/PHP, ~2B tokens)")
    parser.add_argument("--wiki-en", type=int, default=500000, help="Wikipedia EN docs")
    parser.add_argument("--wiki-fr", type=int, default=500000, help="Wikipedia FR docs")
    parser.add_argument("--raw-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--distill-dir", type=str, default=str(PROJECT_ROOT / "data" / "distill"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    distill_dir = Path(args.distill_dir)
    sources = args.sources
    if "all" in sources:
        sources = ["github", "webdev", "fable5", "alpaca", "mythos", "fable5_agent", "glm52", "wiki"]

    print("=" * 60)
    print("11B MoE — CODE-FIRST DATA DOWNLOAD (dev web + system)")
    print("=" * 60)
    print(f"Sources: {sources}")
    print(f"Raw output: {raw_dir}")
    print(f"Distill output: {distill_dir}")
    print()

    # CODE (bulk) — goes to raw_dir for pretraining
    if "github" in sources:
        download_github_code(args.github_files, raw_dir)
    if "webdev" in sources:
        download_web_dev_bulk(args.webdev_files, raw_dir)

    # CODE (instruction) — goes to distill_dir (pretrain + SFT)
    if "fable5" in sources:
        download_fable5_code(distill_dir)
    if "alpaca" in sources:
        download_code_alpaca(distill_dir)

    # DISTILLATION — goes to distill_dir (pretrain + SFT)
    if "mythos" in sources:
        download_mythos(distill_dir)
    if "fable5_agent" in sources:
        download_fable5_agent(distill_dir)
    if "glm52" in sources:
        download_glm52_agent(distill_dir)

    # BASE FR/EN — goes to raw_dir for pretraining
    if "wiki" in sources:
        download_wikipedia("en", args.wiki_en, raw_dir)
        download_wikipedia("fr", args.wiki_fr, raw_dir)

    # Copy distill pretrain files to raw_dir so prepare.py picks them up
    print()
    print(">>> Copying distillation pretrain files to raw_dir...")
    import shutil
    for f in distill_dir.glob("*_pretrain.txt"):
        dest = raw_dir / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            print(f"  Copied {f.name}")

    # Merge for tokenizer
    tok_sample = raw_dir.parent / "tokenizer_sample.txt"
    random.seed(42)
    txt_files = sorted(raw_dir.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name != "tokenizer_sample.txt"]
    print(f"\n>>> Building tokenizer sample from {len(txt_files)} files...")
    with open(tok_sample, "w") as out_f:
        for txt_file in txt_files:
            with open(txt_file, "r") as in_f:
                lines = []
                for i, line in enumerate(in_f):
                    if i >= 5000:
                        break
                    lines.append(line)
                if lines:
                    sample = random.sample(lines, min(2000, len(lines)))
                    out_f.writelines(sample)
    print(f"Tokenizer sample -> {tok_sample}")

    # Summary
    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    total_docs = 0
    for f in sorted(raw_dir.glob("*.txt")):
        if f.name == "tokenizer_sample.txt":
            continue
        with open(f, "r") as fh:
            n = sum(1 for _ in fh)
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {n:>10,} docs  {size_mb:>8.0f} MB")
        total_docs += n
    print(f"  {'TOTAL':<40} {total_docs:>10,} docs")
    est_tokens = total_docs * 600
    print(f"\n  Estimated tokens: ~{est_tokens/1e9:.1f}B")

    sft_files = sorted(distill_dir.glob("*.jsonl"))
    if sft_files:
        print(f"\nSFT files (for fine-tuning phase):")
        total_sft = 0
        for f in sft_files:
            with open(f, "r") as fh:
                n = sum(1 for _ in fh)
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:<40} {n:>8,} examples  {size_mb:>6.1f} MB")
            total_sft += n
        print(f"  {'TOTAL SFT':<40} {total_sft:>8,} examples")
    print()
