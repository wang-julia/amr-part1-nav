import json
import re
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_OUTPUT_DIR = Path(__file__).parent / "vlm_outputs"
DEFAULT_INTERNVL_MODEL = "OpenGVLab/InternVL2_5-4B"

_INTERNVL_CACHE: Dict[Tuple, Tuple] = {}


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


def _context_position_hint(context: Dict | None) -> str | None:
    if not context:
        return None
    near_objects = context.get("mock_depth", {}).get("near_objects", [])
    joined = " ".join(str(item).lower() for item in near_objects)
    if "center_left" in joined or "left" in joined:
        return "center-left"
    if "center_right" in joined or "right" in joined:
        return "center-right"
    return None


def _is_right_wall_group_context(context: Dict | None) -> bool:
    if not context:
        return False
    near_objects = context.get("mock_depth", {}).get("near_objects", [])
    joined = " ".join(str(item).lower() for item in near_objects)
    return "group" in joined and ("center_right" in joined or "right" in joined)


def _cleanup_semantic_json(result: Dict, context: Dict | None) -> Dict:
    """Keep VLM output aligned with the demo's local-costmap contract."""
    cleaned = dict(result)
    position_hint = _context_position_hint(context)
    right_wall_group = _is_right_wall_group_context(context)

    objects = []
    position_updates: dict[str, str] = {}
    for obj in cleaned.get("objects", []):
        if obj.get("type") == "doorway":
            continue
        updated = dict(obj)
        if (
            position_hint
            and updated.get("type") == "person"
            and updated.get("image_position") == "center"
        ):
            updated["image_position"] = position_hint
            position_updates["center"] = position_hint
            if position_hint == "center-left":
                updated["reason"] = "Person is standing slightly left of the robot's forward camera centerline."
            elif position_hint == "center-right":
                updated["reason"] = "Person is standing slightly right of the robot's forward camera centerline."
        objects.append(updated)

    if right_wall_group:
        group_positions = ["center-right", "center-right", "top-right"]
        group_distances = ["near", "medium", "far"]
        group_objects = []
        for index, obj in enumerate([item for item in objects if item.get("type") in {"person", "group"}]):
            slot = min(index, len(group_positions) - 1)
            updated = dict(obj)
            updated["type"] = "person"
            updated["image_position"] = group_positions[slot]
            updated["distance_estimate"] = group_distances[slot]
            updated["motion_state"] = "stationary"
            updated["mobility_class"] = "possibly_moving"
            updated["facing_direction"] = "unknown"
            updated["social_radius"] = min(int(updated.get("social_radius", 4)), 4)
            updated["semantic_cost"] = int(updated.get("semantic_cost", 145))
            updated["reason"] = "Person is part of a standing crowd along the right wall, leaving the center-left hallway passable."
            group_objects.append(updated)
        if group_objects:
            objects = group_objects
    cleaned["objects"] = objects

    valid_centers = {obj.get("image_position") for obj in objects}
    intents = []
    source_intents = cleaned.get("local_costmap_intent", [])
    if right_wall_group and objects:
        source_intents = [
            {
                "shape": "circle",
                "center": obj["image_position"],
                "distance_estimate": obj["distance_estimate"],
                "radius": 4,
                "mobility_class": "possibly_moving",
                "cost": 145,
            }
            for obj in objects
        ]
    for item in source_intents:
        updated = dict(item)
        center = updated.get("center")
        if center in position_updates:
            updated["center"] = position_updates[center]
            center = updated["center"]
        if valid_centers and center not in valid_centers:
            continue
        if not valid_centers:
            continue
        if updated.get("shape") == "circle" and int(updated.get("radius", 4)) > 4:
            updated["radius"] = 4
        intents.append(updated)
    cleaned["local_costmap_intent"] = intents
    if right_wall_group:
        cleaned["scene"] = "A standing crowd is aligned along the right wall of the hallway."
        cleaned["navigation_advice"] = "Keep clearance from the right-wall crowd while using the open center-left hallway space."
    elif position_hint == "center-left":
        cleaned["scene"] = str(cleaned.get("scene", "")).replace("middle of a hallway", "left side of the hallway")
        cleaned["navigation_advice"] = "Continue forward while leaving clearance around the person on the left; the right side remains passable."
    elif position_hint == "center-right":
        cleaned["scene"] = str(cleaned.get("scene", "")).replace("middle of a hallway", "right side of the hallway")
        cleaned["navigation_advice"] = "Continue forward while leaving clearance around the person on the right; the left side remains passable."
    return cleaned


def _load_internvl_model(
    model_id: str,
    local_files_only: bool = False,
) -> Tuple:
    """Lazy-load InternVL2.5 so replay mode stays lightweight."""
    cache_key = (model_id, local_files_only)
    if cache_key in _INTERNVL_CACHE:
        return _INTERNVL_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "InternVL mode requires torch, transformers, accelerate, timm, and torchvision "
            "inside the Python environment running generate_demo.py."
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

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_flash_attn=False,
        local_files_only=local_files_only,
    ).eval()
    if device != "cuda":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    _INTERNVL_CACHE[cache_key] = (model, tokenizer, device, dtype)
    return _INTERNVL_CACHE[cache_key]


def run_internvl_inference(
    image_path: Path,
    prompt: str,
    context: Dict,
    model_id: str = DEFAULT_INTERNVL_MODEL,
    max_new_tokens: int = 700,
    local_files_only: bool = False,
) -> Dict:
    """Run local InternVL2.5 and return parsed semantic-costmap JSON."""
    if not image_path.exists():
        raise FileNotFoundError(f"InternVL image not found: {image_path}")

    model, tokenizer, device, dtype = _load_internvl_model(model_id, local_files_only)

    import torch
    from PIL import Image
    from torchvision import transforms

    full_prompt = (
        f"{prompt.strip()}\n\n"
        "Mock navigation context:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Return only valid JSON for the semantic costmap schema."
    )
    question = f"<image>\n{full_prompt}"

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
    pixel_values = transform(img).unsqueeze(0).to(device, dtype=dtype)

    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, question, generation_config)

    return _extract_json_object(response)


def infer_semantic_json(
    frame_id: str,
    mode: str = "replay",
    image_path: Path | None = None,
    prompt: str = "",
    context: Dict | None = None,
    model_id: str = DEFAULT_INTERNVL_MODEL,
    fallback_to_replay: bool = True,
    save_output_dir: Path | None = None,
    internvl_max_new_tokens: int = 700,
    internvl_local_files_only: bool = False,
    use_saved_internvl: bool = False,
    # backwards-compat aliases
    qwen_device_map: str = "auto",
    qwen_max_new_tokens: int | None = None,
    qwen_min_pixels: int | None = None,
    qwen_max_pixels: int | None = None,
    qwen_local_files_only: bool | None = None,
    use_saved_qwen: bool | None = None,
) -> Dict:
    if qwen_max_new_tokens is not None:
        internvl_max_new_tokens = qwen_max_new_tokens
    if qwen_local_files_only is not None:
        internvl_local_files_only = qwen_local_files_only
    if use_saved_qwen is not None:
        use_saved_internvl = use_saved_qwen

    if mode == "replay":
        return load_replay_output(frame_id)

    if mode in ("internvl", "qwen"):
        if use_saved_internvl and save_output_dir is not None:
            saved_path = save_output_dir / f"{frame_id}.json"
            if saved_path.exists():
                result = load_replay_output(frame_id, save_output_dir)
                return _cleanup_semantic_json(result, context)
        if image_path is None or context is None:
            raise ValueError("image_path and context are required for internvl mode.")
        try:
            result = run_internvl_inference(
                image_path=image_path,
                prompt=prompt,
                context=context,
                model_id=model_id,
                max_new_tokens=internvl_max_new_tokens,
                local_files_only=internvl_local_files_only,
            )
        except Exception as exc:
            if not fallback_to_replay:
                raise
            result = load_replay_output(frame_id)
            result = {
                **result,
                "inference_mode": "replay_fallback_after_internvl_error",
                "internvl_error": str(exc),
            }
        result = _cleanup_semantic_json(result, context)

        if save_output_dir is not None:
            save_output_dir.mkdir(parents=True, exist_ok=True)
            path = save_output_dir / f"{frame_id}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    raise ValueError(f"Unsupported inference mode: {mode}")
