"""Validation for InternVL's navigation output contract (prompt.txt schema).

People/position/depth now come from the YOLO + Depth sensor stack, not the VLM,
so the VLM's JSON only carries what it alone produces: environment type,
background space per zone, a costmap, and a recommended heading.
"""

from __future__ import annotations

from typing import Any


ENVIRONMENT_TYPES = ("hallway", "open_room", "lobby", "outdoor", "doorway", "other")
SPACE_RATINGS = ("open", "narrow", "blocked")
COSTMAP_ZONES = ("left", "center_left", "center", "center_right", "right")
RECOMMENDED_DIRECTIONS = ("forward", "left", "right", "hold")


def _check_space_analysis(space_analysis: Any, issues: list[str]) -> None:
    if not isinstance(space_analysis, dict):
        issues.append("space_analysis must be an object.")
        return
    for zone in COSTMAP_ZONES:
        field = f"{zone}_space"
        if space_analysis.get(field) not in SPACE_RATINGS:
            issues.append(f"space_analysis.{field} must be one of {SPACE_RATINGS}.")
    if not isinstance(space_analysis.get("space_notes"), str) or not space_analysis.get(
        "space_notes"
    ):
        issues.append("space_analysis.space_notes must be a non-empty string.")


def _check_costmap(costmap: Any, issues: list[str]) -> None:
    if not isinstance(costmap, dict):
        issues.append("costmap must be an object.")
        return
    for zone in COSTMAP_ZONES:
        value = costmap.get(zone)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not (
            0 <= value <= 100
        ):
            issues.append(f"costmap.{zone} must be a number between 0 and 100.")


def evaluate_observation(value: Any) -> dict[str, Any]:
    """Report schema problems without altering the model's original JSON."""
    issues: list[str] = []
    if not isinstance(value, dict):
        return {
            "valid": False,
            "issues": ["The JSON root must be an object."],
        }

    if value.get("environment_type") not in ENVIRONMENT_TYPES:
        issues.append(f"environment_type must be one of {ENVIRONMENT_TYPES}.")

    _check_space_analysis(value.get("space_analysis"), issues)
    _check_costmap(value.get("costmap"), issues)

    if value.get("recommended_direction") not in RECOMMENDED_DIRECTIONS:
        issues.append(f"recommended_direction must be one of {RECOMMENDED_DIRECTIONS}.")

    recommended_angle = value.get("recommended_angle")
    if (
        not isinstance(recommended_angle, (int, float))
        or isinstance(recommended_angle, bool)
        or not (-90 <= recommended_angle <= 90)
    ):
        issues.append("recommended_angle must be a number between -90 and 90.")

    if not isinstance(value.get("reason"), str) or not value.get("reason"):
        issues.append("reason must be a non-empty string.")

    return {
        "valid": not issues,
        "issues": issues,
    }
