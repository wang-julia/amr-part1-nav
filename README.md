# VLM (Vision-Language Model) Runner

This repository contains a minimal runner for a Qwen vision-language model
that analyzes a scene image and returns a JSON-like analysis useful for
robotic navigation and social navigation reasoning.

Files
- `VLM/run_vlm.py`: Main script. It performs the following sectors:
  - Imports & configuration: sets `MODEL_ID` and `IMAGE_PATH`.
  - Prompt: defines the JSON-like schema the model should output.
  - Model & processor setup: loads the pretrained model and processor.
  - Message assembly: builds the chat-style `messages` with image + prompt.
  - Preprocessing: converts text/images/videos into tensors via the
    processor and `process_vision_info` helper.
  - Generation: runs `model.generate` under `torch.no_grad()`.
  - Post-processing: trims input tokens from outputs and decodes text.

How to run

1. Create a Python virtual environment and install dependencies (you may
   already have these in your environment):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the script (ensure `IMAGE_PATH` points to a valid image):

```bash
python VLM/run_vlm.py
```

Notes and suggestions
- The script uses `device_map="auto"` and `torch_dtype=torch.float16`; make
  sure you have a compatible GPU or adjust these options for CPU use.
- The prompt asks for a strict JSON-like format — if you plan to parse
  the output programmatically, validate the text before JSON parsing.
- The helper `qwen_vl_utils.process_vision_info` is used to extract image
  and video tensors; check that module if you need custom preprocessing.

Next steps
- If you want, I can initialize a git repository here, commit these
  changes, add your remote `https://github.com/wang-julia/vlm-amr-26.git`,
  and push — tell me whether to proceed and which branch name to use.
