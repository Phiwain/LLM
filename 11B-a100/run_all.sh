#!/bin/bash
# Full pipeline: download -> tokenizer -> prepare -> train -> evaluate
# Code-first: GitHub code + distillation (Claude Fable 5, Mythos, GLM-5.2) + minimal FR/EN base
set -e
echo "============================================"
echo "  11B MoE Code-First LLM — Full Pipeline"
echo "  ~70% code, ~20% distillation, ~10% FR/EN base"
echo "============================================"

echo ""
echo ">>> [1/5] Downloading datasets (code + dev web + distill + base FR/EN)..."
echo "    Target: ~10B tokens (12M GitHub + 3M web dev + distillation + FR/EN base)"
python3 -m src.data.download --sources all --github-files 12000000 --webdev-files 3000000 --wiki-en 500000 --wiki-fr 500000 2>&1 | grep -E "Saved|Streaming|Downloading|Already|TOTAL|SFT|tokens|Copied|tokenizer"

echo ""
echo ">>> [2/5] Training BPE tokenizer..."
python3 -m src.tokenizer.train_tokenizer --vocab-size 32000 2>&1 | tail -2

echo ""
echo ">>> [3/5] Tokenizing and packing data..."
python3 -m src.data.prepare --seq-len 1024 2>&1 | grep -E "Found|Total|Packed|Train:|Val:|Saved"

echo ""
echo ">>> [4/5] Training 11B MoE model..."
python3 -m src.train 2>&1

echo ""
echo ">>> [5/5] Evaluating..."
python3 -m src.evaluate 2>&1

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Interactive: python -m src.generate"
echo "  SFT data: data/distill/*.jsonl"
echo "============================================"
