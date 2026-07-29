"""Disk-backed inference history shared by the server and batch evaluator."""

from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from observations import evaluate_observation


ROOT = Path(__file__).resolve().parent
HISTORY_DIR = ROOT / "history"
IMAGES_DIR = HISTORY_DIR / "images"
INDEX_PATH = HISTORY_DIR / "runs.json"
HISTORY_LOCK = Lock()


def _ensure_directories() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _read_unlocked() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    try:
        value = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _write_unlocked(entries: list[dict[str, Any]]) -> None:
    _ensure_directories()
    temporary = INDEX_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    temporary.replace(INDEX_PATH)


def save_run(
    *,
    image_bytes: bytes,
    image_suffix: str,
    filename: str,
    batch_id: str,
    trial_index: int,
    prompt: str,
    effective_prompt: str,
    context: dict,
    semantic_json: dict,
    evaluation: dict,
    model: str,
    seed: int,
    temperature: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Persist one independent model result and its source image."""
    run_id = uuid4().hex
    suffix = (
        image_suffix.lower()
        if image_suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
        else ".jpg"
    )
    image_name = f"{run_id}{suffix}"
    entry = {
        "id": run_id,
        "batch_id": batch_id,
        "trial_index": int(trial_index),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "image_file": image_name,
        "model": model,
        "seed": int(seed),
        "temperature": float(temperature),
        "elapsed_seconds": float(elapsed_seconds),
        "prompt": prompt,
        "effective_prompt": effective_prompt,
        "context": context,
        "semantic_json": semantic_json,
        "evaluation": evaluation,
    }
    with HISTORY_LOCK:
        _ensure_directories()
        (IMAGES_DIR / image_name).write_bytes(image_bytes)
        entries = _read_unlocked()
        entries.insert(0, entry)
        _write_unlocked(entries)
    return entry


def list_runs() -> list[dict[str, Any]]:
    """Return compact newest-first summaries for the history list."""
    with HISTORY_LOCK:
        entries = _read_unlocked()
    summaries = []
    for entry in entries:
        semantic_json = entry.get("semantic_json", {})
        if not isinstance(semantic_json, dict):
            semantic_json = {}
        people = semantic_json.get("people")
        person_records = len(people) if isinstance(people, list) else 0
        evaluation = entry.get("evaluation")
        if not isinstance(evaluation, dict):
            evaluation = evaluate_observation(semantic_json)
        summaries.append(
            {
                "id": entry.get("id"),
                "batch_id": entry.get("batch_id"),
                "trial_index": entry.get("trial_index", 1),
                "created_at": entry.get("created_at"),
                "filename": entry.get("filename", "unknown image"),
                "model": entry.get("model", "unknown model"),
                "people_count": semantic_json.get("people_count", 0),
                "person_records": person_records,
                "schema_valid": evaluation.get("valid", False),
                "issue_count": len(evaluation.get("issues", [])),
                "elapsed_seconds": entry.get("elapsed_seconds", 0),
                "image_url": f"/api/history/{entry.get('id')}/image",
            }
        )
    return summaries


def get_run(run_id: str) -> dict[str, Any] | None:
    with HISTORY_LOCK:
        entry = next((item for item in _read_unlocked() if item.get("id") == run_id), None)
    if entry is None:
        return None
    result = dict(entry)
    if not isinstance(result.get("evaluation"), dict):
        result["evaluation"] = evaluate_observation(result.get("semantic_json"))
    result.pop("costmap", None)
    result.pop("understanding", None)
    result["image_url"] = f"/api/history/{run_id}/image"
    return result


def get_image(run_id: str) -> tuple[bytes, str] | None:
    entry = get_run(run_id)
    if entry is None:
        return None
    image_path = IMAGES_DIR / str(entry.get("image_file", ""))
    if not image_path.is_file() or image_path.parent != IMAGES_DIR:
        return None
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return image_path.read_bytes(), content_type


def clear_runs() -> int:
    """Delete only app-owned history records and their copied images."""
    with HISTORY_LOCK:
        entries = _read_unlocked()
        for entry in entries:
            image_name = Path(str(entry.get("image_file", ""))).name
            if image_name:
                (IMAGES_DIR / image_name).unlink(missing_ok=True)
        _write_unlocked([])
    return len(entries)
