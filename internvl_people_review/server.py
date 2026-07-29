"""Standalone InternVL2.5 people-observation review server."""

import argparse
import base64
import json
import os
import secrets
import tempfile
import time
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Timer
from urllib.parse import urlparse

from history_store import clear_runs, get_image, get_run, list_runs, save_run
from internvl_runtime import DEFAULT_MODEL, compose_prompt, run_inference
from observations import evaluate_observation


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
PROMPT_PATH = ROOT / "prompt.txt"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MODEL_LOCK = Lock()

DEFAULT_CONTEXT = {}


def runtime_status() -> dict:
    status = {
        "model": os.environ.get("INTERNVL_MODEL", DEFAULT_MODEL),
        "torch": False,
        "transformers": False,
        "cuda": False,
        "device": "runtime unavailable",
    }
    try:
        import torch

        status["torch"] = True
        status["cuda"] = bool(torch.cuda.is_available())
        if status["cuda"]:
            status["device"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            status["device"] = "Apple MPS"
        else:
            status["device"] = "CPU"
    except Exception as exc:
        status["error"] = str(exc)
    try:
        import timm  # noqa: F401
        import transformers  # noqa: F401

        status["transformers"] = True
    except Exception:
        pass
    status["ready"] = status["torch"] and status["transformers"]
    return status


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_IMAGE_BYTES * 2:
            raise ValueError("The request is empty or exceeds the 20 MB image limit.")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("The request body must be a JSON object.")
        return value

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/config":
            prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self.send_json(
                {
                    "prompt": prompt,
                    "context": DEFAULT_CONTEXT,
                    "effective_prompt": compose_prompt(prompt, DEFAULT_CONTEXT),
                    "runtime": runtime_status(),
                }
            )
            return
        if path == "/api/health":
            self.send_json({"ok": True, "runtime": runtime_status()})
            return
        if path == "/api/history":
            self.send_json({"runs": list_runs()})
            return
        if path.startswith("/api/history/") and path.endswith("/image"):
            run_id = path.removeprefix("/api/history/").removesuffix("/image").strip("/")
            image = get_image(run_id)
            if image is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body, content_type = image
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/api/history/"):
            run_id = path.removeprefix("/api/history/").strip("/")
            entry = get_run(run_id)
            if entry is None:
                self.send_json({"error": "History result not found."}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(entry)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/infer":
            self.handle_infer()
            return
        if self.path == "/api/history/clear":
            self.send_json({"deleted": clear_runs()})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_infer(self) -> None:
        image_path = None
        started = time.perf_counter()
        try:
            request = self.read_json()
            prompt = str(request.get("prompt", "")).strip()
            context = request.get("context")
            if not prompt:
                raise ValueError("The prompt cannot be empty.")
            if not isinstance(context, dict):
                raise ValueError("Navigation context must be a JSON object.")

            encoded = str(request.get("image_base64", ""))
            if "," in encoded:
                encoded = encoded.split(",", 1)[1]
            raw_image = base64.b64decode(encoded, validate=True)
            if not raw_image or len(raw_image) > MAX_IMAGE_BYTES:
                raise ValueError("Choose a non-empty image no larger than 20 MB.")

            suffix = Path(str(request.get("filename", "upload.jpg"))).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}:
                suffix = ".jpg"
            handle = tempfile.NamedTemporaryFile(
                mode="wb", suffix=suffix, prefix="internvl_review_", delete=False
            )
            with handle:
                handle.write(raw_image)
            image_path = Path(handle.name)

            seed = int(request.get("seed") or secrets.randbelow(2_147_483_647))
            temperature = float(request.get("temperature", 0.35))
            model_id = os.environ.get("INTERNVL_MODEL", DEFAULT_MODEL)
            batch_id = str(request.get("batch_id") or secrets.token_hex(8))
            trial_index = max(1, int(request.get("trial_index", 1)))
            with MODEL_LOCK:
                semantic_json = run_inference(
                    image_path=image_path,
                    base_prompt=prompt,
                    context=context,
                    model_id=model_id,
                    seed=seed,
                    temperature=max(0.0, min(temperature, 1.5)),
                    top_p=0.9,
                    max_new_tokens=int(os.environ.get("INTERNVL_MAX_NEW_TOKENS", "640")),
                    local_files_only=os.environ.get("INTERNVL_LOCAL_FILES_ONLY", "0") == "1",
                )
            evaluation = evaluate_observation(semantic_json)
            elapsed_seconds = round(time.perf_counter() - started, 2)
            effective_prompt = compose_prompt(prompt, context)
            history_entry = save_run(
                image_bytes=raw_image,
                image_suffix=suffix,
                filename=str(request.get("filename", "upload.jpg")),
                batch_id=batch_id,
                trial_index=trial_index,
                prompt=prompt,
                effective_prompt=effective_prompt,
                context=context,
                semantic_json=semantic_json,
                evaluation=evaluation,
                model=model_id,
                seed=seed,
                temperature=max(0.0, min(temperature, 1.5)),
                elapsed_seconds=elapsed_seconds,
            )
            self.send_json(
                {
                    "semantic_json": semantic_json,
                    "evaluation": evaluation,
                    "effective_prompt": effective_prompt,
                    "seed": seed,
                    "elapsed_seconds": elapsed_seconds,
                    "history_id": history_entry["id"],
                    "batch_id": batch_id,
                    "trial_index": trial_index,
                    "runtime": runtime_status(),
                }
            )
        except Exception as exc:
            self.send_json(
                {
                    "error": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "runtime": runtime_status(),
                },
                HTTPStatus.BAD_REQUEST,
            )
        finally:
            if image_path is not None:
                image_path.unlink(missing_ok=True)

    def log_message(self, message_format: str, *args) -> None:
        print(f"[internvl-review] {self.address_string()} {message_format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://localhost:{args.port}/"
    status = runtime_status()
    print(f"InternVL Manual Review: {url}")
    print(f"Model: {status['model']}")
    print(f"Compute: {status['device']}")
    if not args.no_browser:
        Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping InternVL Manual Review.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
