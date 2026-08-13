"""Deterministic costmap + heading fusion.

The VLM only rates the background space per zone (open/narrow/blocked). This
module combines that with the authoritative YOLO detections (zone) and depth
labels (near/mid/far) to build the 5-zone costmap and pick a heading — pure
arithmetic, so a person in a zone always raises that zone's cost and the
decision never silently ignores a detection.
"""

from __future__ import annotations

from typing import Any

ZONES = ("left", "center_left", "center", "center_right", "right")
# Bearing (degrees) of each zone's center across the robot's -90..90 forward arc.
ZONE_ANGLE = {"left": -72, "center_left": -36, "center": 0, "center_right": 36, "right": 72}
# Cost a detected person adds to their zone, by depth bucket.
DEPTH_COST = {"near": 60, "mid": 35, "far": 15}
# Cost the background rating adds to a zone.
SPACE_COST = {"open": 0, "narrow": 20, "blocked": 45}
HOLD_THRESHOLD = 100  # if even the best zone is this costly, hold instead of moving


def build_costmap(detections: list[dict], space_analysis: Any) -> dict[str, int]:
    space = space_analysis if isinstance(space_analysis, dict) else {}
    cost = {zone: 0 for zone in ZONES}
    for d in detections or []:
        zone = d.get("zone")
        if zone in cost:
            cost[zone] += DEPTH_COST.get(d.get("depth"), DEPTH_COST["far"])
    for zone in ZONES:
        cost[zone] += SPACE_COST.get(space.get(f"{zone}_space"), 0)
        cost[zone] = min(100, cost[zone])
    return cost


def recommend_heading(cost: dict[str, int], detections: list[dict]) -> dict[str, Any]:
    # Lowest cost wins; ties break toward the most-forward (smallest |angle|) zone.
    best = min(ZONES, key=lambda z: (cost[z], abs(ZONE_ANGLE[z])))
    occupied = sorted({d.get("zone") for d in (detections or []) if d.get("zone") in cost})
    occupied_note = (
        f"avoiding {len(detections)} detected person(s) in {', '.join(occupied)}"
        if occupied
        else "no people detected"
    )

    if cost[best] >= HOLD_THRESHOLD:
        return {
            "recommended_direction": "hold",
            "recommended_angle": 0,
            "reason": f"Every zone is blocked or occupied ({occupied_note}); holding position.",
        }

    angle = ZONE_ANGLE[best]
    direction = "forward" if best == "center" else ("left" if angle < 0 else "right")
    return {
        "recommended_direction": direction,
        "recommended_angle": angle,
        "reason": (
            f"{best.replace('_', ' ')} has the lowest cost ({cost[best]}); "
            f"steering {angle}° there, {occupied_note}."
        ),
    }


def build_navigation(vlm_json: Any, detections: list[dict]) -> dict[str, Any]:
    """Merge the VLM's background reading with a code-computed costmap + heading
    into the full navigation JSON the rest of the app expects."""
    vlm = vlm_json if isinstance(vlm_json, dict) else {}
    space_analysis = vlm.get("space_analysis")
    space_analysis = space_analysis if isinstance(space_analysis, dict) else {}
    costmap = build_costmap(detections, space_analysis)
    heading = recommend_heading(costmap, detections)
    return {
        "environment_type": vlm.get("environment_type", "other"),
        "space_analysis": space_analysis,
        "costmap": costmap,
        **heading,
    }
