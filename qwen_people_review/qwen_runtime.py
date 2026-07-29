"""InternVL runtime for people-observation inference (replaces Qwen2.5-VL)."""

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "OpenGVLab/InternVL2_5-4B"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CACHE: dict[tuple, Any] = {}


def compose_prompt(base_prompt: str, context: dict) -> str:
    parts = [base_prompt.strip()]
    if context:
        parts.append("Optional cart context:\n" + json.dumps(context, indent=2))
    parts.append("Return only the requested valid JSON object.")
    return "\n\n".join(parts)


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    known_keys = "people|people_count|person_id|motion|distance|torso_direction|face_direction"
    cleaned = re.sub(rf'"({known_keys})\s*:', r'"\1":', cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("Model did not return a JSON object.")
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            preview = cleaned[:240].replace("\n", " ")
            raise ValueError(f"Model returned malformed JSON: {exc}. Output: {preview}") from exc
    if isinstance(value, list):
        value = {"objects": value}
    if not isinstance(value, dict):
        raise ValueError("Model returned JSON but the root value is not an object.")
    return value


def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_dtype(device):
    import torch
    if device.type in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


def _build_transform(input_size: int = 448):
    import torchvision.transforms as T
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _load_image_tensor(image_path: Path, input_size: int = 448):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    return _build_transform(input_size)(img).unsqueeze(0)  # (1, 3, H, W)


def load_model(model_id: str, local_files_only: bool = False):
    key = (model_id, local_files_only)
    if key in _CACHE:
        return _CACHE[key]

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing InternVL runtime packages. Install torch, transformers, "
            "accelerate, timm, and torchvision."
        ) from exc

    device = _get_device()
    dtype = _get_dtype(device)

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,  # Flash attention requires CUDA; disabled for Mac/CPU
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    _CACHE[key] = (model, tokenizer, device, dtype)
    return _CACHE[key]


def run_inference(
    image_path: Path,
    base_prompt: str,
    context: dict,
    model_id: str,
    seed: int,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_new_tokens: int = 700,
    local_files_only: bool = False,
) -> dict:
    import torch

    model, tokenizer, device, dtype = load_model(model_id, local_files_only)

    effective_prompt = compose_prompt(base_prompt, context)
    question = f"<image>\n{effective_prompt}"
    pixel_values = _load_image_tensor(image_path).to(device=device, dtype=dtype)

    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
    if temperature > 0:
        generation_config.update({"temperature": temperature, "top_p": top_p})

    torch.manual_seed(seed)
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, question, generation_config)

    return extract_json(response)
