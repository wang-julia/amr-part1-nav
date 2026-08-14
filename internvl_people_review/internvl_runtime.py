"""Standalone InternVL2.5 runtime for people-observation review."""

import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "OpenGVLab/InternVL2_5-4B"
_CACHE: dict[tuple, tuple] = {}


def compose_prompt(base_prompt: str, context: dict) -> str:
    parts = [base_prompt.strip()]
    if context:
        parts.append("Optional scene context:\n" + json.dumps(context, indent=2))
    parts.append("Return only the requested valid JSON object.")
    return "\n\n".join(parts)


_ZONE_PAT = r"(?:left|center_left|center|center_right|right)"


def _repair_json(text: str) -> str:
    """Fix common InternVL JSON generation mistakes before parsing.

    zone_scan malformation: model writes "zone_scan": "left": open, ...
    The zone name is quoted, which terminates the JSON string early and puts
    a bare colon outside the string, causing a parse error.
    Fix: remove the spurious closing quote from the first zone name so the
    entire description becomes a valid string value.
      Before: "zone_scan": "left": open, center_left: open, right: open",
      After:  "zone_scan": "left: open, center_left: open, right: open",
    """
    text = re.sub(
        r'("zone_scan"\s*:\s*"' + _ZONE_PAT + r')"(\s*:)',
        r'\1\2',
        text,
    )
    return text


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    known_keys = (
        "environment_type|space_analysis|left_space|center_left_space|center_space|"
        "center_right_space|right_space|space_notes|zone_scan|recommended_zone|zone_reason|"
        "people|people_count|crowd_density|person_id|is_cluster|approximate_count|motion|"
        "depth|depth_rank|torso_direction|zone|predicted_zone|prediction_reason|costmap|"
        "left|center_left|center|center_right|right|recommended_direction|recommended_angle|"
        "angle_reason|reason"
    )
    cleaned = re.sub(rf'"({known_keys})\s*:', r'"\1":', cleaned)
    cleaned = _repair_json(cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("InternVL did not return a JSON object.")
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            preview = cleaned[:240].replace("\n", " ")
            raise ValueError(
                f"InternVL returned malformed JSON: {exc}. Output: {preview}"
            ) from exc
    if isinstance(value, list):
        value = {"objects": value}
    if not isinstance(value, dict):
        raise ValueError(
            "InternVL returned JSON, but the root value is not an object or object list."
        )
    return value


def load_model(model_id: str, local_files_only: bool = False):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing InternVL runtime packages. Install torch, transformers, "
            "accelerate, timm, and torchvision in this app's Python environment."
        ) from exc

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    key = (model_id, local_files_only, device)
    if key in _CACHE:
        return _CACHE[key]

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_flash_attn=False,
        local_files_only=local_files_only,
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    _CACHE[key] = (model, tokenizer, device, dtype)
    return _CACHE[key]


def _preprocess_image(image_path: Path, dtype: Any, device: str):
    from PIL import Image
    from torchvision import transforms

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (448, 448), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return transform(img).unsqueeze(0).to(device, dtype=dtype)


def run_inference(
    image_path: Path,
    base_prompt: str,
    context: dict,
    model_id: str,
    seed: int,
    temperature: float = 0.35,
    top_p: float = 0.9,
    max_new_tokens: int = 700,
    local_files_only: bool = False,
) -> dict:
    import torch

    model, tokenizer, device, dtype = load_model(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    effective_prompt = compose_prompt(base_prompt, context)
    question = f"<image>\n{effective_prompt}"
    pixel_values = _preprocess_image(image_path, dtype, device)

    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
    )
    if temperature > 0:
        generation_config.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, question, generation_config)

    return extract_json(response)
