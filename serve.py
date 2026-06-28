#!/usr/bin/env python3
"""Serveur API compatible OpenAI pour le MoE 1B — utilisable avec Msty."""
import sys
import time
import uuid
import json
import torch
import sentencepiece as spm
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.model.model import MoELLM

app = FastAPI(title="PhiwAIn API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
sp = None
device = None
cfg = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "phiwain-1b"
    messages: List[Message]
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.8
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str = "phiwain-1b"
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.8
    stream: Optional[bool] = False


def load():
    global model, sp, device, cfg
    checkpoint = PROJECT_ROOT / "checkpoints" / "sft_model.pt"
    config_path = PROJECT_ROOT / "config.yaml"
    tokenizer_path = PROJECT_ROOT / "tokenizer" / "bpe.model"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("Chargement du modèle...")
    cfg = Config.from_yaml(str(config_path))
    model = MoELLM(cfg.model)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"Modèle chargé: {cfg.model.n_params:,} params (step {ckpt['step']})")

    print("Chargement du tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    print(f"Tokenizer: vocab={sp.get_piece_size()}")
    print("Serveur prêt sur http://localhost:8080")


@torch.inference_mode()
def generate_tokens(prompt, max_tokens=200, temperature=0.8):
    bos_id = 1
    eos_id = 2
    max_seq_len = cfg.model.max_seq_len

    ids = [bos_id] + sp.encode(prompt)
    prev_len = len(sp.decode(ids[1:]))
    for _ in range(max_tokens):
        ctx = ids[-max_seq_len:]
        tokens = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(tokens)
        logits = logits[0, -1] / max(temperature, 0.01)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id == eos_id:
            break
        ids.append(next_id)
        full = sp.decode(ids[1:])
        yield full[prev_len:]
        prev_len = len(full)


def build_prompt(messages):
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"Système: {m.content}")
        elif m.role == "user":
            parts.append(f"User: {m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}")
    parts.append("Assistant: ")
    return "\n".join(parts)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "phiwain-1b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "phiwain",
            }
        ],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "phiwain",
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = build_prompt(req.messages)

    if req.stream:
        def stream():
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            for token in generate_tokens(prompt, req.max_tokens, req.temperature):
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "phiwain-1b",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            done = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "phiwain-1b",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    full_text = ""
    for token in generate_tokens(prompt, req.max_tokens, req.temperature):
        full_text += token

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "phiwain-1b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(sp.encode(prompt)),
            "completion_tokens": len(sp.encode(full_text)),
            "total_tokens": len(sp.encode(prompt)) + len(sp.encode(full_text)),
        },
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    full_text = ""
    for token in generate_tokens(req.prompt, req.max_tokens, req.temperature):
        full_text += token

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "phiwain-1b",
        "choices": [
            {
                "index": 0,
                "text": full_text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(sp.encode(req.prompt)),
            "completion_tokens": len(sp.encode(full_text)),
            "total_tokens": len(sp.encode(req.prompt)) + len(sp.encode(full_text)),
        },
    }


if __name__ == "__main__":
    load()
    uvicorn.run(app, host="0.0.0.0", port=8080)
