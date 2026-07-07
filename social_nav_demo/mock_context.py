import json
from pathlib import Path
from typing import Dict, List


FRAME_CONTEXTS: Dict[str, Dict] = {
    "frame_001": {
        "robot_pose": {"x": 52, "y": 84, "heading": "north"},
        "local_waypoints": [[52, 80], [52, 68], [52, 56], [52, 44]],
        "planned_path_direction": "forward",
        "mock_lidar": {
            "front": "clear",
            "front_left": "clear",
            "front_right": "clear",
            "right": "clear",
        },
        "mock_depth": {
            "near_objects": [],
            "free_space": ["front", "front_left", "front_right"],
        },
    },
    "frame_002": {
        "robot_pose": {"x": 52, "y": 66, "heading": "north"},
        "local_waypoints": [[52, 62], [52, 52], [52, 42], [52, 30]],
        "planned_path_direction": "forward",
        "mock_lidar": {
            "front": "clear",
            "front_left": "soft-return",
            "front_right": "clear",
            "right": "clear",
        },
        "mock_depth": {
            "near_objects": ["standing_person_center_left"],
            "free_space": ["front_right", "right"],
        },
    },
    "frame_003": {
        "robot_pose": {"x": 52, "y": 54, "heading": "north"},
        "local_waypoints": [[52, 48], [52, 40], [48, 30], [40, 23]],
        "planned_path_direction": "forward-left",
        "mock_lidar": {
            "front": "clear",
            "front_left": "soft-return",
            "front_right": "clear",
            "right": "clear",
        },
        "mock_depth": {
            "near_objects": ["moving_person_center_left"],
            "free_space": ["front", "front_right"],
        },
    },
    "frame_004": {
        "robot_pose": {"x": 52, "y": 34, "heading": "north-west"},
        "local_waypoints": [[52, 30], [48, 25], [40, 23], [30, 23]],
        "planned_path_direction": "forward",
        "mock_lidar": {
            "front": "clear",
            "front_left": "unknown-gap",
            "front_right": "soft-return",
            "right": "occupied",
        },
        "mock_depth": {
            "near_objects": ["turning_point_center_left", "standing_person_center_right"],
            "free_space": ["front", "left corridor"],
        },
    },
    "frame_005": {
        "robot_pose": {"x": 48, "y": 23, "heading": "west"},
        "baseline_path_waypoints": [[48, 23], [13, 23]],
        "local_waypoints": [[42, 23], [34, 23], [26, 23], [13, 23]],
        "planned_path_direction": "forward",
        "mock_lidar": {
            "front": "partially-clear",
            "front_left": "clear",
            "front_right": "soft-return",
            "right": "occupied",
        },
        "mock_depth": {
            "near_objects": ["standing_group_center_right"],
            "free_space": ["front_left", "far_left"],
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
            "summary": "The robot sees a clear corridor stem, so the semantic layer adds no extra local cost.",
        },
        {
            "frame_id": "frame_002",
            "title": "Frame 2 - standing person",
            "summary": "A standing person is possibly moving, so the demo adds a cautious circular personal-space zone.",
        },
        {
            "frame_id": "frame_003",
            "title": "Frame 3 - moving person",
            "summary": "A moving person adds current personal-space cost plus a directional predicted-motion zone.",
        },
        {
            "frame_id": "frame_004",
            "title": "Frame 4 - turning point and person",
            "summary": "A turning point and nearby person create caution zones near the T-junction.",
        },
        {
            "frame_id": "frame_005",
            "title": "Frame 5 - crowd near corridor wall",
            "summary": "A group along the right corridor wall gets a narrow wall-side social caution band.",
        },
    ]


def write_timeline_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": "T-junction robot photo navigation",
        "description": "A five-photo robot camera and navigation-context sequence for local VLM semantic costmap fusion.",
        "timeline": [
            {**item, "context": FRAME_CONTEXTS[item["frame_id"]]}
            for item in get_timeline()
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
