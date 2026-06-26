#!/bin/bash
# RunPod A100 setup for 11B MoE — run after spawning the pod
set -e
echo "============================================"
echo "  11B MoE Bilingual LLM — RunPod A100 Setup"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"

echo ""
echo ">>> Installing dependencies..."
pip install --break-system-packages -q datasets sentencepiece safetensors tqdm pyyaml bitsandbytes accelerate 2>&1 | tail -3

python3 -c "
import torch, datasets, sentencepiece
import bitsandbytes as bnb
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'bitsandbytes: {bnb.__version__}')
print('All dependencies OK')
"

echo ""
echo "Setup complete! Next steps:"
echo "  1. python -m src.data.download --lang both --n-docs 1000000"
echo "  2. python -m src.tokenizer.train_tokenizer --vocab-size 32000"
echo "  3. python -m src.data.prepare --seq-len 1024"
echo "  4. python -m src.train"
echo "  5. python -m src.evaluate"
echo "  Or run all: bash run_all.sh"
