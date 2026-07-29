"""Standalone Qwen2.5-VL runtime for people-observation review."""

import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
_CACHE: dict[tuple, tuple[Any, Any, Any]] = {}


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
    # Repair only a missing closing quote on a known schema key, such as
    # Qwen's occasional `"people:[]` empty-scene response.
    known_keys = "people|people_count|person_id|motion|distance|torso_direction|face_direction"
    cleaned = re.sub(rf'"({known_keys})\s*:', r'"\1":', cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("Qwen did not return a JSON object.")
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            preview = cleaned[:240].replace("\n", " ")
            raise ValueError(f"Qwen returned malformed JSON: {exc}. Output: {preview}") from exc
    if isinstance(value, list):
        value = {"objects": value}
    if not isinstance(value, dict):
        raise ValueError("Qwen returned JSON, but the root value is not an object or object list.")
    return value


def load_model(
    model_id: str,
    min_pixels: int = 128 * 28 * 28,
    max_pixels: int = 1024 * 28 * 28,
    local_files_only: bool = False,
):
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing Qwen runtime packages. Install torch, transformers, "
            "accelerate, and qwen-vl-utils in this app's Python environment."
        ) from exc

    use_4bit = torch.cuda.is_available() and os.environ.get("QWEN_LOAD_IN_4BIT", "1") == "1"
    key = (model_id, min_pixels, max_pixels, local_files_only, use_4bit)
    if key in _CACHE:
        return _CACHE[key]
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        local_files_only=local_files_only,
    )
    _CACHE[key] = (model, processor, process_vision_info)
    return _CACHE[key]


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
    model, processor, process_vision_info = load_model(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    effective_prompt = compose_prompt(base_prompt, context)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": effective_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generation_options = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_options.update({"temperature": temperature, "top_p": top_p})
    with torch.no_grad():
        generated = model.generate(**inputs, **generation_options)
    trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return extract_json(output_text)
