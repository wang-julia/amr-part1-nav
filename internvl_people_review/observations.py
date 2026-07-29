"""Validation for InternVL's people-only observation contract."""

from __future__ import annotations

from typing import Any


MOTIONS = ("walking", "standing", "running")
DISTANCES = ("close", "mid", "far")
FACING_DIRECTIONS = (
    "front",
    "left",
    "right",
    "back",
    "frontleft",
    "frontright",
    "backleft",
    "backright",
)
DIRECTION_ANGLES = {
    "front": 0,
    "frontright": 45,
    "right": 90,
    "backright": 135,
    "back": 180,
    "backleft": 225,
    "left": 270,
    "frontleft": 315,
}


def _direction_difference(first: str, second: str) -> int:
    difference = abs(DIRECTION_ANGLES[first] - DIRECTION_ANGLES[second])
    return min(difference, 360 - difference)


def evaluate_observation(value: Any) -> dict[str, Any]:
    """Report schema problems without altering the model's original JSON."""
    issues: list[str] = []
    if not isinstance(value, dict):
        return {
            "valid": False,
            "reported_people_count": None,
            "person_records": 0,
            "complete_records": 0,
            "issues": ["The JSON root must be an object."],
        }

    people_count = value.get("people_count")
    if not isinstance(people_count, int) or isinstance(people_count, bool) or people_count < 0:
        issues.append("people_count must be a non-negative integer.")

    people = value.get("people")
    if not isinstance(people, list):
        issues.append("people must be an array.")
        people = []

    if isinstance(people_count, int) and not isinstance(people_count, bool):
        if people_count != len(people):
            issues.append(
                f"people_count is {people_count}, but people contains {len(people)} records."
            )

    complete_records = 0
    seen_ids: set[int] = set()
    allowed_fields = {
        "person_id",
        "motion",
        "distance",
        "torso_direction",
        "face_direction",
    }
    for index, person in enumerate(people, start=1):
        label = f"people[{index - 1}]"
        person_issues_before = len(issues)
        if not isinstance(person, dict):
            issues.append(f"{label} must be an object.")
            continue

        unexpected = sorted(set(person) - allowed_fields)
        missing = sorted(allowed_fields - set(person))
        if unexpected:
            issues.append(f"{label} has unexpected fields: {', '.join(unexpected)}.")
        if missing:
            issues.append(f"{label} is missing fields: {', '.join(missing)}.")

        person_id = person.get("person_id")
        if not isinstance(person_id, int) or isinstance(person_id, bool) or person_id < 1:
            issues.append(f"{label}.person_id must be a positive integer.")
        elif person_id in seen_ids:
            issues.append(f"{label}.person_id duplicates {person_id}.")
        else:
            seen_ids.add(person_id)

        if person.get("motion") not in MOTIONS:
            issues.append(f"{label}.motion must be walking, standing, or running.")
        if person.get("distance") not in DISTANCES:
            issues.append(f"{label}.distance must be close, mid, or far.")
        torso_direction = person.get("torso_direction")
        face_direction = person.get("face_direction")
        if torso_direction not in FACING_DIRECTIONS:
            issues.append(
                f"{label}.torso_direction must be one of the eight required directions."
            )
        if face_direction not in FACING_DIRECTIONS:
            issues.append(
                f"{label}.face_direction must be one of the eight required directions."
            )
        if torso_direction in DIRECTION_ANGLES and face_direction in DIRECTION_ANGLES:
            difference = _direction_difference(torso_direction, face_direction)
            if difference > 90:
                issues.append(
                    f"{label}.torso_direction and face_direction differ by "
                    f"{difference} degrees; the anatomical limit is 90 degrees."
                )

        if len(issues) == person_issues_before:
            complete_records += 1

    return {
        "valid": not issues,
        "reported_people_count": people_count,
        "person_records": len(people),
        "complete_records": complete_records,
        "issues": issues,
    }
