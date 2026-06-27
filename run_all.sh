#!/bin/bash
# Full pipeline: download -> tokenizer -> prepare -> train -> evaluate
# Run on RunPod A100
set -e

echo "============================================"
echo "  MoE Bilingual LLM — Full Pipeline"
echo "============================================"

echo ""
echo ">>> [1/5] Downloading CulturaX EN+FR..."
python3 -m src.data.download --lang both --n-docs 750000

echo ""
echo ">>> [2/5] Training BPE tokenizer..."
python3 -m src.tokenizer.train_tokenizer --vocab-size 32000

echo ""
echo ">>> [3/5] Tokenizing and packing data..."
python3 -m src.data.prepare --seq-len 1024

echo ""
echo ">>> [4/5] Training MoE model..."
python3 -m src.train

echo ""
echo ">>> [5/5] Evaluating..."
python3 -m src.evaluate

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Checkpoints: checkpoints/"
echo "  Interactive: python -m src.generate"
echo "============================================"
