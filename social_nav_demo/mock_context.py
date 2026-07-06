import json
from pathlib import Path
from typing import Dict, List


FRAME_CONTEXTS: Dict[str, Dict] = {
    "frame_001": {
        "robot_pose": {"x": 18, "y": 82, "heading": "north-east"},
        "local_waypoints": [[20, 78], [28, 70], [36, 62], [45, 52]],
        "planned_path_direction": "forward-right",
        "mock_lidar": {
            "front": "clear",
            "front_left": "clear",
            "front_right": "clear",
            "right": "clear",
        },
        "mock_depth": {
            "near_objects": [],
            "free_space": ["front", "front_right", "front_left"],
        },
    },
    "frame_002": {
        "robot_pose": {"x": 28, "y": 70, "heading": "north-east"},
        "local_waypoints": [[30, 67], [38, 59], [47, 50], [56, 42]],
        "planned_path_direction": "forward-right",
        "mock_lidar": {
            "front": "clear",
            "front_left": "clear",
            "front_right": "soft-return",
            "right": "clear",
        },
        "mock_depth": {
            "near_objects": ["walking_person_center_right"],
            "free_space": ["front_left", "far_left"],
        },
    },
    "frame_003": {
        "robot_pose": {"x": 42, "y": 58, "heading": "north-east"},
        "local_waypoints": [[44, 55], [52, 47], [60, 39], [68, 31]],
        "planned_path_direction": "forward-right",
        "mock_lidar": {
            "front": "clear",
            "front_left": "occupied",
            "front_right": "clear",
            "right": "unknown-gap",
        },
        "mock_depth": {
            "near_objects": ["doorway_center_left", "blind_corner_top_right"],
            "free_space": ["front_right", "center"],
        },
    },
    "frame_004": {
        "robot_pose": {"x": 55, "y": 46, "heading": "north-east"},
        "local_waypoints": [[58, 43], [65, 36], [73, 28], [82, 20]],
        "planned_path_direction": "forward-right",
        "mock_lidar": {
            "front": "soft-return",
            "front_left": "occupied",
            "front_right": "partially-clear",
            "right": "clear",
        },
        "mock_depth": {
            "near_objects": ["group_center"],
            "free_space": ["right", "far_right"],
        },
    },
}


def get_mock_context(frame_id: str) -> Dict:
    return FRAME_CONTEXTS[frame_id]


def get_timeline() -> List[Dict]:
    return [
        {
            "frame_id": "frame_001",
            "title": "Frame 1 - clear hallway",
            "summary": "Baseline and enhanced paths are nearly identical because no social risk is visible.",
        },
        {
            "frame_id": "frame_002",
            "title": "Frame 2 - walking person near planned path",
            "summary": "The replayed VLM JSON adds personal-space cost and extends it in the person's facing direction.",
        },
        {
            "frame_id": "frame_003",
            "title": "Frame 3 - doorway and blind corner",
            "summary": "Medium semantic caution zones bias the route away from uncertain space.",
        },
        {
            "frame_id": "frame_004",
            "title": "Frame 4 - group partially blocks corridor",
            "summary": "A larger social zone makes the enhanced path longer but lower risk.",
        },
    ]


def write_timeline_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": "Office hallway navigation",
        "description": "A four-frame mocked camera and navigation-context sequence for local VLM semantic costmap fusion.",
        "timeline": [
            {**item, "context": FRAME_CONTEXTS[item["frame_id"]]}
            for item in get_timeline()
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
