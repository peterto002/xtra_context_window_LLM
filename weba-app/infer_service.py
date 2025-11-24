import os
from contextlib import nullcontext

import torch

from rope_alibi_bdh_style_llm import BiasAttentionALiBiRopeBDHStyleConfig, BiasAttentionALiBiRopeBDHStyle

force_cpu = False
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

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "BiasAttentionALiBiRopeBDHStyle_PT_model.pt")

# -------- Load once --------
_MODEL_CONFIG = BiasAttentionALiBiRopeBDHStyleConfig()
_model = BiasAttentionALiBiRopeBDHStyle(_MODEL_CONFIG).to(device)
_state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
_model.load_state_dict(_state)
_model.eval()


def _to_topk(v):
    # top_k <= 0 means disable
    return None if (v is None or int(v) <= 0) else int(v)


@torch.no_grad()
def generate(prompt_text: str, max_new_tokens: int, top_k: int | None, temperature: float) -> str:
    prompt = torch.tensor(
        bytearray(prompt_text, "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)

    with ctx:
        out = _model.generate(
            prompt,
            max_new_tokens=int(max_new_tokens),
            top_k=_to_topk(top_k),
            temperature=float(temperature),
        )
    decoded = bytes(out.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
    return decoded
