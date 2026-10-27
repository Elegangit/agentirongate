import re
import tomllib
from pathlib import Path

CONDITIONS = {"minimal", "rule", "directed", "neutral", "hint"}


def identity(root):
    with (Path(root) / "project.toml").open("rb") as stream:
        data = tomllib.load(stream)["project"]
    if not re.fullmatch(r"[a-z][a-z0-9-]*", data["slug"]):
        raise ValueError("Project slug must be lowercase letters, digits and hyphens")
    if not isinstance(data["name"], str) or not data["name"].strip():
        raise ValueError("Project name is required")
    return data


def load_config(path):
    with Path(path).open("rb") as stream:
        cfg = tomllib.load(stream)
    required = {"seed", "repeats", "conditions", "prefix_calls", "branch_calls",
                "branch_proposals", "max_api_calls", "feedback_scope", "approver",
                "bootstrap_samples", "models"}
    if set(cfg) != required:
        raise ValueError(f"Configuration fields differ: {set(cfg) ^ required}")
    for key in required - {"conditions", "models", "feedback_scope", "approver"}:
        if type(cfg[key]) is not int or cfg[key] < (0 if key in {"seed", "max_api_calls"} else 1):
            raise ValueError(f"Invalid integer: {key}")
    conditions = cfg["conditions"]
    if (not isinstance(conditions, list) or not conditions or
            not all(isinstance(c, str) and c in CONDITIONS for c in conditions) or
            len(set(conditions)) != len(conditions) or "minimal" not in conditions):
        raise ValueError("Use distinct feedback conditions including minimal")
    if cfg["feedback_scope"] not in {"persistent", "first_only"}:
        raise ValueError("Unknown feedback scope")
    if cfg["approver"] not in {"ideal", "reject_all"}:
        raise ValueError("Unknown approver")
    if not isinstance(cfg["models"], list) or not cfg["models"]:
        raise ValueError("At least one model is required")
    ids = set()
    for model in cfg["models"]:
        if set(model) != {"id", "provider", "model", "temperature", "max_output_tokens"}:
            raise ValueError("Model configuration fields differ")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", model["id"]) or model["id"] in ids:
            raise ValueError("Model IDs must be unique safe identifiers")
        ids.add(model["id"])
        if model["provider"] not in {"offline", "gemini", "openrouter"}:
            raise ValueError("Unknown provider")
        if not isinstance(model["model"], str) or not model["model"].strip():
            raise ValueError("An explicit model version is required")
        if type(model["temperature"]) not in {float, int} or not 0 <= model["temperature"] <= 2:
            raise ValueError("Invalid temperature")
        if type(model["max_output_tokens"]) is not int or model["max_output_tokens"] < 1:
            raise ValueError("Invalid output-token budget")
    return cfg
