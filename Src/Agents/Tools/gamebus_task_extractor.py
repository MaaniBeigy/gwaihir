import json
import math
import sys
from datetime import datetime
from typing import Any, Dict


def flatten_condition(condition: Dict[str, Any]) -> Dict[str, str]:
    """Flatten a rule condition to `{property, operator, value}`."""
    prop = condition.get("linkToProperty", {})
    op = condition.get("linkToOperator", {})
    return {
        "property": prop.get("translation_key", "unknown"),
        "operator": op.get("operator", "unknown"),
        "value": condition.get("rhs_value", "unknown"),
    }


def extract_task(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant fields from a single challenge rule (task)."""
    default_gd = rule.get("linkToDefaultGameDescriptor", {}) or {}
    point_mapping = rule.get("linkToPointMappings", [{}])[0] if rule.get("linkToPointMappings") else {}

    # Derive duration/steps from explicit conditions only.
    conditions = rule.get("linkToRuleConditions", [])
    rule_name = str(rule.get("name", ""))
    rule_desc = str(rule.get("description", ""))
    est_duration_min = None
    min_steps = None
    for cond in conditions:
        flat = flatten_condition(cond)
        if flat["property"] == "DURATION" and flat["operator"] == "STRICTLY_GREATER":
            # Convert seconds to minutes plus a small buffer.
            est_duration_min = int(float(flat["value"]) / 60) + 5
        elif flat["property"] == "STEPS" and flat["operator"] == "STRICTLY_GREATER":
            min_steps = int(float(flat["value"])) + 1
            est_duration_min = max(est_duration_min or 0, math.ceil(min_steps / 110) + 5)
        elif flat["property"] == "STEPS_SUM" and flat["operator"] == "STRICTLY_GREATER":
            min_steps = int(float(flat["value"])) + 1
            est_duration_min = max(est_duration_min or 0, math.ceil(min_steps / 110) + 5)

    # Default duration estimates when no explicit condition.
    if est_duration_min is None:
        text = (rule_name + " " + rule_desc).lower()
        if any(word in text for word in ["15-minute", "15 min", "15 minutes"]):
            est_duration_min = 20
        elif any(word in text for word in ["30-minute", "30 min", "30 minutes", "30 seconds", "yoga pose"]):
            est_duration_min = 35 if "30 m" in text else 2
        elif any(word in text for word in ["7-minute", "7 min", "h5p", "workout"]):
            est_duration_min = 10
        elif any(word in text for word in ["photo", "picture", "image"]):
            est_duration_min = 3
        else:
            est_duration_min = 5

    return {
        "task_id": rule["id"],
        "name": rule["name"],
        "description": rule.get("description", ""),
        "est_duration_min": est_duration_min,
        "min_steps": min_steps,
        "cooldown_days": rule.get("min_days_between_fire", 0),
        "max_repeats": rule.get("max_times_fired", 0),
        "image_required": rule.get("image_required", False),
        "game_descriptor_id": default_gd.get("id"),
        "game_descriptor_key": default_gd.get("translation_key", "GENERAL"),
        "points": point_mapping.get("points", 0),
        "conditions": [flatten_condition(c) for c in conditions],
    }


def simplify_challenge(challenge: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify one challenge to core fields + tasks (no lotteries)."""
    rules = challenge.get("linkToChallengeRules", [])

    # Tolerant of missing campaign-Challenge metadata so the slim shape is always usable.
    simplified = {
        "challenge_id": challenge.get("id") or challenge.get("challenge_id"),
        "name": challenge.get("name", ""),
        "description": challenge.get("description", ""),
        "type": challenge.get("challenge_type", "GENERAL"),
        "start_date": challenge.get("start_date"),
        "end_date": challenge.get("end_date"),
        "target": challenge.get("target"),
        "success_next": challenge.get("success_next"),
        "failure_next": challenge.get("failure_next"),
        "tasks": [extract_task(rule) for rule in rules if rule.get("name")],
    }

    return simplified


def process_json(input_file: str, output_file: str) -> None:
    """Read input JSON, simplify, and write slim version."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    simplified_data = [simplify_challenge(challenge) for challenge in data]

    slim_json = {
        "simplified_at": datetime.now().isoformat(),
        "total_challenges": len(simplified_data),
        "challenges": simplified_data,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(slim_json, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(data)} challenges into {len(simplified_data)} slim challenges.")
    print(f"Output saved to {output_file}")
    if simplified_data and simplified_data[0]["tasks"]:
        example = json.dumps(simplified_data[0]["tasks"][0], indent=2, ensure_ascii=False)
        print("Example first task:\n", example)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gamebus_task_extractor.py input.json output.json")
        sys.exit(1)

    process_json(sys.argv[1], sys.argv[2])
