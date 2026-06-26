#!/bin/bash
# RunPod A100 setup script — run this after spawning the pod
# Usage: bash setup_runpod.sh
set -e

echo "============================================"
echo "  MoE Bilingual LLM — RunPod A100 Setup"
echo "============================================"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU pod?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# PyTorch CUDA check
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install dependencies
echo ""
echo ">>> Installing Python dependencies..."
pip install -r requirements.txt 2>&1 | tail -5

# Verify
python3 -c "
import torch, datasets, sentencepiece, transformers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('All dependencies OK')
"

echo ""
echo "Setup complete! Next steps:"
echo "  1. python -m src.data.download       # Download CulturaX EN+FR"
echo "  2. python -m src.tokenizer.train_tokenizer  # Train BPE tokenizer"
echo "  3. python -m src.data.prepare         # Tokenize & pack data"
echo "  4. python -m src.train                # Train MoE model"
echo "  5. python -m src.evaluate             # Evaluate"
echo "  6. python -m src.generate             # Interactive generation"
