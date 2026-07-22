"""Run every image in a folder once and save the results to app history."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from history_store import save_run
from observations import evaluate_observation
from qwen_runtime import DEFAULT_MODEL, compose_prompt, run_inference


ROOT = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=str(ROOT / "evaluation_set" / "images"))
    parser.add_argument("--prompt", default=str(ROOT / "prompt.txt"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--match", default="", help="Only evaluate filenames containing this text")
    parser.add_argument("--report", default=str(ROOT / "evaluation_set" / "qwen_results_latest.json"))
    parser.add_argument("--batch-id", default="", help="History label for this independent trial")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    context: dict = {}
    model_id = os.environ.get("QWEN_MODEL", DEFAULT_MODEL)
    images = sorted(path for path in folder.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if args.match:
        images = [path for path in images if args.match.lower() in path.name.lower()]
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No supported images found in {folder}")

    generated_batch_id = f"folder-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    batch_id = args.batch_id.strip() or generated_batch_id
    completed = []
    failures = []
    print(f"Model: {model_id}")
    print(f"Images: {len(images)}")
    print(f"Batch: {batch_id}")

    for index, image_path in enumerate(images, start=1):
        seed = secrets.randbelow(2_147_483_647)
        started = time.perf_counter()
        print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
        try:
            raw_result = run_inference(
                image_path=image_path,
                base_prompt=prompt,
                context=context,
                model_id=model_id,
                seed=seed,
                temperature=max(0.0, min(args.temperature, 1.5)),
                top_p=0.9,
                max_new_tokens=int(os.environ.get("QWEN_MAX_NEW_TOKENS", "640")),
                local_files_only=os.environ.get("QWEN_LOCAL_FILES_ONLY", "0") == "1",
            )
            semantic_json = raw_result
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
            )
            completed.append(
                {
                    "history_id": entry["id"],
                    "filename": image_path.name,
                    "elapsed_seconds": elapsed,
                    "semantic_json": semantic_json,
                    "evaluation": evaluation,
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
        "completed": completed,
        "failures": failures,
    }
    report_path = Path(args.report).resolve()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed: {len(completed)}; failed: {len(failures)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
