# Qwen People Observation Review

A standalone local application for checking whether Qwen can reliably extract
four people-observation categories from a robot-camera image:

1. Number of visible people
2. Motion: `walking`, `standing`, or `running`
3. Robot-to-person distance: `close`, `mid`, or `far`
4. Direction, reported separately for the torso and face. Both use `front`,
   `left`, `right`, `back`, `frontleft`, `frontright`, `backleft`, or
   `backright`.

The application does not generate a costmap or navigation advice. Upload one
image, edit the prompt directly, and generate three independent Qwen results.
Every successful inference is saved to local History with the source image,
raw model JSON, schema evaluation, exact prompt, seed, model, and runtime.

## Setup on Windows

Python 3.12 and an NVIDIA CUDA-capable GPU are recommended.

```powershell
git clone https://github.com/wang-julia/vlm-amr-26.git
cd vlm-amr-26\qwen_people_review
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The setup script creates `.venv` and installs CUDA-enabled PyTorch,
Transformers, Accelerate, `bitsandbytes`, and `qwen-vl-utils`. CUDA runs use
4-bit Qwen weights by default to fit an 8 GB GPU.

## Run

```powershell
.\run.ps1
```

Open [http://localhost:8020/](http://localhost:8020/).

The first inference may download `Qwen/Qwen2.5-VL-3B-Instruct` if it is not
already cached. To use an existing local model directory:

```powershell
$env:QWEN_MODEL = "C:\models\Qwen2.5-VL-3B-Instruct"
.\run.ps1
```

## Output Contract

```json
{
  "people_count": 1,
  "people": [
    {
      "person_id": 1,
      "motion": "walking",
      "distance": "mid",
      "torso_direction": "frontleft",
      "face_direction": "left"
    }
  ]
}
```

Torso and face direction are required even when motion is `standing`. They are
relative to the camera viewing axis: deeper into the image is `front`, and
toward the camera is `back`. The app records Qwen's raw JSON and reports schema
problems separately; it does not silently replace missing or invalid values.

## Source Layout

- `server.py`: local HTTP API and application server
- `history_store.py`: persistent local inference history
- `observations.py`: non-destructive output-schema evaluation
- `evaluate_folder.py`: one-pass evaluation of an image folder
- `qwen_runtime.py`: Qwen2.5-VL model loading and inference
- `prompt.txt`: editable default core-observation prompt
- `web/index.html`: upload, trial comparison, and history interface
- `setup.ps1`: isolated Windows CUDA environment setup

To evaluate every photo in `evaluation_set/images` once and save the results
to History:

```powershell
.\.venv\Scripts\python.exe evaluate_folder.py
```

Give each evaluation cycle a distinct History label and report file:

```powershell
.\.venv\Scripts\python.exe evaluate_folder.py `
  --batch-id four-core-trial-02 `
  --report evaluation_set\qwen_results_four_core_trial_02.json
```

History entries and reports generated before this four-field contract remain
useful as legacy evidence, but the app flags them as incomplete under the new
schema.
