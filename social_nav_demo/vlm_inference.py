import json
import re
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_OUTPUT_DIR = Path(__file__).parent / "vlm_outputs"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

_QWEN_CACHE: Dict[Tuple[str, str, int, int, bool], Tuple[object, object, object]] = {}


def load_replay_output(
    frame_id: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict:
    """Default reliable demo mode: load prepared VLM-style JSON."""
    path = output_dir / f"{frame_id}.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_json_object(text: str) -> Dict:
    """Parse model output even if it wraps JSON in markdown fences."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _load_qwen_model(
    model_id: str,
    device_map: str = "auto",
    min_pixels: int = 128 * 28 * 28,
    max_pixels: int = 512 * 28 * 28,
    local_files_only: bool = False,
) -> Tuple[object, object, object]:
    """Lazy-load Qwen2.5-VL so replay mode stays lightweight."""
    cache_key = (model_id, device_map, min_pixels, max_pixels, local_files_only)
    if cache_key in _QWEN_CACHE:
        return _QWEN_CACHE[cache_key]

    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen mode requires torch, transformers, accelerate, and qwen-vl-utils "
            "inside the Python environment running generate_demo.py."
        ) from exc

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        local_files_only=local_files_only,
    )
    _QWEN_CACHE[cache_key] = (model, processor, process_vision_info)
    return _QWEN_CACHE[cache_key]


def run_qwen_vl_inference(
    image_path: Path,
    prompt: str,
    context: Dict,
    model_id: str = DEFAULT_QWEN_MODEL,
    max_new_tokens: int = 700,
    device_map: str = "auto",
    min_pixels: int = 128 * 28 * 28,
    max_pixels: int = 512 * 28 * 28,
    local_files_only: bool = False,
) -> Dict:
    """Run local Qwen2.5-VL and return parsed semantic-costmap JSON."""
    if not image_path.exists():
        raise FileNotFoundError(f"Qwen image not found: {image_path}")

    model, processor, process_vision_info = _load_qwen_model(
        model_id,
        device_map,
        min_pixels,
        max_pixels,
        local_files_only,
    )
    qwen_prompt = (
        f"{prompt.strip()}\n\n"
        "Mock navigation context:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Return only valid JSON for the semantic costmap schema."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": qwen_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Import here so replay mode does not depend on torch being installed.
    import torch

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return _extract_json_object(output_text)


def infer_semantic_json(
    frame_id: str,
    mode: str = "replay",
    image_path: Path | None = None,
    prompt: str = "",
    context: Dict | None = None,
    model_id: str = DEFAULT_QWEN_MODEL,
    fallback_to_replay: bool = True,
    save_output_dir: Path | None = None,
    qwen_device_map: str = "auto",
    qwen_max_new_tokens: int = 700,
    qwen_min_pixels: int = 128 * 28 * 28,
    qwen_max_pixels: int = 512 * 28 * 28,
    qwen_local_files_only: bool = False,
    use_saved_qwen: bool = False,
) -> Dict:
    if mode == "replay":
        return load_replay_output(frame_id)
    if mode == "qwen":
        if use_saved_qwen and save_output_dir is not None:
            saved_path = save_output_dir / f"{frame_id}.json"
            if saved_path.exists():
                return load_replay_output(frame_id, save_output_dir)
        if image_path is None or context is None:
            raise ValueError("image_path and context are required for qwen mode.")
        try:
            result = run_qwen_vl_inference(
                image_path=image_path,
                prompt=prompt,
                context=context,
                model_id=model_id,
                max_new_tokens=qwen_max_new_tokens,
                device_map=qwen_device_map,
                min_pixels=qwen_min_pixels,
                max_pixels=qwen_max_pixels,
                local_files_only=qwen_local_files_only,
            )
        except Exception as exc:
            if not fallback_to_replay:
                raise
            result = load_replay_output(frame_id)
            result = {
                **result,
                "inference_mode": "replay_fallback_after_qwen_error",
                "qwen_error": str(exc),
            }

        if save_output_dir is not None:
            save_output_dir.mkdir(parents=True, exist_ok=True)
            path = save_output_dir / f"{frame_id}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    raise ValueError(f"Unsupported inference mode: {mode}")
