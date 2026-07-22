# Four-Field Qwen Evaluation Set

This folder contains the current people-only image group for evaluating:

1. Person count
2. Walking, standing, or running
3. Close, mid, or far robot-to-person distance
4. Mandatory camera-relative eight-way torso and face directions

It does not evaluate costmaps, planning, grouping, image-side position, turns,
blind corners, or navigation advice.

## Contents

- `images/`: 17 test photos in JPEG, WebP, and AVIF formats
- `prompt.txt`: the frozen torso-and-face prompt for the next batch
- `ground_truth_labels.json`: reviewed human labels, currently including the approved image 04 calibration label
- `qwen_results_latest.json`: results from the previous schema; overwritten by the next `evaluate_folder.py` run
- `qwen_results_prompt_v1.json`: retained earlier prompt experiment
- `qwen_results_four_core_trial_01.json`: previous single-direction deterministic run
- `qwen_results_four_core_trial_02_torso_face.json`: first separate torso/face direction run
- `qwen_results_four_core_trial_03_direction_cues.json`: refined camera-relative visible-surface direction run
- `qwen_results_calibration_04_trial_04.json` through `trial_10_crop_head_turn.json`: isolated image 04 prompt-calibration attempts, including anatomical-consistency validation
- `contact_sheet.jpg`, `expected.json`, and `SOURCES.md`: legacy materials from the earlier photo set, not the answer key for the current group

## Evaluation

Run each image independently with the exact text in `prompt.txt`. Review the
result in the web app's History view and retain incorrect model answers.

Suggested weighted score:

1. Person count: 30%
2. Motion: 25%
3. Distance: 20%
4. Torso direction: 12.5%
5. Face direction: 12.5%

Target at least 70% overall and at least 70% on person count. Report per-field
scores so success on person count cannot hide weak motion or direction results.

The next batch command is:

```powershell
.\.venv\Scripts\python.exe evaluate_folder.py
```
