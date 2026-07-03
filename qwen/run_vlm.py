"""Run a Qwen vision-language model on an image and print the analysis."""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# Config
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "phase1/images/test.jpg"


# Prompt (expected JSON-like schema)
prompt = """
You are helping an autonomous mobile robot understand a scene.

Analyze the image and return the result in this exact JSON-like format:

{
  "scene_summary": "...",
  "humans": [
    {
      "location": "...",
      "activity": "...",
      "navigation_relevance": "low/medium/high"
    }
  ],
  "obstacles": [
    {
      "object": "...",
      "location": "...",
      "navigation_relevance": "low/medium/high"
    }
  ],
  "social_navigation_advice": [
    "..."
  ]
}
"""


# Model setup
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
  MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)


# Message (image + prompt)
messages = [{"role": "user", "content": [{"type": "image", "image": IMAGE_PATH}, {"type": "text", "text": prompt}]}]


# Preprocess text and vision inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)


# Tokenize inputs and move to model device
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)


# Inference
with torch.no_grad():
  generated_ids = model.generate(**inputs, max_new_tokens=400)


# Decode generated tokens (remove input prefix) and print
generated_ids_trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)