"""Run every image in a folder once and save the results to app history."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from detection_runtime import analyze_image, detect_zone_obstacles
from history_store import save_run
from internvl_runtime import DEFAULT_MODEL, compose_prompt, run_inference
from navigation import build_navigation
from observations import evaluate_observation


ROOT = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(ROOT / "evaluation_set" / "images"),
        help="Folder containing images to evaluate.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Batch identifier written into every history record (default: random hex).",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "evaluation_set" / "internvl_results.json"),
        help="Path where the JSON report is written.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Path to a prompt file to use instead of the default prompt.txt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many images.",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Only process filenames that contain this substring (case-insensitive).",
    )
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Skip YOLO/depth detection and run InternVL ungrounded (for A/B comparison).",
    )
    args = parser.parse_args()

    model_id = os.environ.get("INTERNVL_MODEL", DEFAULT_MODEL)
    batch_id = args.batch_id or secrets.token_hex(8)

    prompt_path = ROOT / "prompt.txt"
    if args.prompt:
        prompt_file = Path(args.prompt)
        if not prompt_file.is_absolute():
            prompt_file = ROOT / prompt_file
        prompt = prompt_file.read_text(encoding="utf-8")
    elif prompt_path.exists():
        prompt = prompt_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    folder = Path(args.folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {folder}")

    images = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if args.match:
        match_lower = args.match.lower()
        images = [p for p in images if match_lower in p.name.lower()]
    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        print("No matching images found.")
        return

    completed: list[dict] = []
    failures: list[dict] = []

    for index, image_path in enumerate(images, start=1):
        seed = secrets.randbelow(2_147_483_647)
        started = time.perf_counter()
        print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
        try:
            # YOLO + Depth are authoritative for people/position/depth and feed the
            # VLM as detected_people; the VLM only reads background + builds the
            # costmap + picks a heading (see prompt.txt). --no-grounding skips the
            # injection (the VLM gets no sensor help) for comparison.
            detected, depth_map, img_w, img_h = analyze_image(image_path)
            person_zones = {d["zone"] for d in detected}
            zone_obstacles = detect_zone_obstacles(depth_map, img_w, img_h, person_zones)
            context: dict = {}
            if detected and not args.no_grounding:
                context["detected_people"] = [
                    {
                        "id": d["id"],
                        "zone": d["zone"],
                        "depth": d["depth"],
                        "depth_rank": d["depth_rank"],
                    }
                    for d in detected
                ]

            raw_result = run_inference(
                image_path=image_path,
                base_prompt=prompt,
                context=context,
                model_id=model_id,
                seed=seed,
                temperature=max(0.0, min(args.temperature, 1.5)),
                top_p=0.9,
                max_new_tokens=int(os.environ.get("INTERNVL_MAX_NEW_TOKENS", "640")),
                local_files_only=os.environ.get("INTERNVL_LOCAL_FILES_ONLY", "0") == "1",
            )
            semantic_json = build_navigation(raw_result, detected, zone_obstacles)
            evaluation = evaluate_observation(semantic_json)
            elapsed = round(time.perf_counter() - started, 2)
            entry = save_run(
                image_bytes=image_path.read_bytes(),
                image_suffix=image_path.suffix,
                filename=image_path.name,
                batch_id=batch_id,
                trial_index=1,
                prompt=prompt,
                effective_prompt=compose_prompt(prompt, context),
                context=context,
                semantic_json=semantic_json,
                evaluation=evaluation,
                model=model_id,
                seed=seed,
                temperature=args.temperature,
                elapsed_seconds=elapsed,
                detections=detected,
            )
            completed.append(
                {
                    "history_id": entry["id"],
                    "filename": image_path.name,
                    "elapsed_seconds": elapsed,
                    "semantic_json": semantic_json,
                    "evaluation": evaluation,
                    "detections": detected,
                }
            )
            print(f"  saved {entry['id']} in {elapsed:.2f}s", flush=True)
        except Exception as exc:
            failures.append({"filename": image_path.name, "error": str(exc)})
            print(f"  failed: {exc}", flush=True)

    report = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "temperature": args.temperature,
        "prompt": prompt,
        "grounding_enabled": not args.no_grounding,
        "completed": completed,
        "failures": failures,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed: {len(completed)}; failed: {len(failures)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
