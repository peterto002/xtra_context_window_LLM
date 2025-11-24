import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch

from rope_alibi_bdh_style_llm import BiasAttentionALiBiRopeBDHStyleConfig, BiasAttentionALiBiRopeBDHStyle

force_cpu = True
device = torch.device("cuda" if not force_cpu and torch.cuda.is_available() else "cpu")

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"Using device: {device} with dtype {dtype}")

# ---------- Configuration ----------
MODEL_CONFIG = BiasAttentionALiBiRopeBDHStyleConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 12000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

HERE = os.path.dirname(__file__)
input_file_path = os.path.join(HERE, "input.txt")
model_file_path = os.path.join(HERE, "weba-app/BiasAttentionALiBiRopeBDHStyle_PT_model.pt")


def get_batch(split):
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)):]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i: i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1: i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to: {path}")


def load_model(path):
    model = BiasAttentionALiBiRopeBDHStyle(MODEL_CONFIG).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Model loaded from: {path}")
    return model


def generate_with_prompt(model, prompt_text: str, max_new_tokens: int = 100, top_k: int | None = 3, temperature: float = 1.0):
    # byte-level prompt → torch.long
    prompt = torch.tensor(
        bytearray(prompt_text, "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature)
    decoded = bytes(out.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
    return decoded


def train_and_save():
    print("training...")

    model = BiasAttentionALiBiRopeBDHStyle(MODEL_CONFIG).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")
    loss_acc = 0.0
    loss_steps = 0
    for step in range(MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)
        x, y = get_batch("train")
        loss_acc += loss
        loss_steps += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step % LOG_FREQ == 0:
            print(f"Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3}")
            loss_acc = 0
            loss_steps = 0

    # done → save uncompiled state dict (PyTorch handles this fine from compiled model)
    # We access the underlying module via ._orig_mod if present; fallback to model itself.
    to_save = getattr(model, "_orig_mod", model)
    save_model(to_save, model_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Train from scratch and save the model.")
    parser.add_argument("--prompt", type=str, default="Who is Julia?", help="Prompt used when evaluating a saved model.")
    parser.add_argument("--tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument("--top_k", type=int, default=3, efault=3, help="Top-k sampling (set <=0 to disable).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    args = parser.parse_args()

    if args.retrain or (not os.path.exists(model_file_path)):
        if not args.retrain:
            print(f"[WARN] No saved model found at {model_file_path}. Training from scratch...")
        train_and_save()

    # Evaluate / generate with saved model
    model = load_model(model_file_path)
    text = generate_with_prompt(
        model=model,
        prompt_text=args.prompt,
        max_new_tokens=args.tokens,
        top_k=(None if args.top_k and args.top_k <= 0 else args.top_k),
        temperature=args.temperature,
    )
    print("=== SAMPLE ===")
    print(text)
