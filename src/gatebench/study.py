"""Offline study planning; no provider access or automatic budget changes."""

from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re

from .io import digest

RUNTIME_PACKAGES = ("httpx", "httpcore", "anyio", "certifi", "idna", "h11", "sniffio",
                    "typing_extensions", "matplotlib", "numpy", "contourpy", "cycler",
                    "fonttools", "kiwisolver", "packaging", "pillow", "pyparsing",
                    "python-dateutil", "six")


def dependency_environment():
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version
                 for d in metadata.distributions() if d.metadata["Name"]}
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "installed_packages": dict(sorted(installed.items())),
            "runtime_packages": {name: installed.get(name.replace("_", "-")) for name in RUNTIME_PACKAGES}}


def research_binding(root, catalogs=()):
    root = Path(root).resolve()
    package = root / "src/gatebench"
    if not (package / "scoring.py").is_file() or not (package / "runner.py").is_file():
        raise ValueError("Study binding requires the research source tree")

    def hash_file(path):
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("Bound files must exist inside the repository")
        return hashlib.sha256(resolved.read_bytes()).hexdigest()

    code = {str(p.relative_to(root)): hash_file(p) for p in sorted(package.rglob("*.py"))
            if "viewer" not in p.relative_to(package).parts and
            p.name not in {"branding.py", "cli.py", "__main__.py"}}
    catalog_hashes = {}
    for name in catalogs:
        path = root / name
        relative = str(path.resolve().relative_to(root))
        if relative in catalog_hashes:
            raise ValueError("Duplicate recovery catalog")
        catalog_hashes[relative] = hash_file(path)
    specs = {name: hash_file(root / name) for name in ("pyproject.toml", "environment.yml")}
    return {"research_code": code, "dependency_specs": specs,
            "recovery_catalogs": dict(sorted(catalog_hashes.items())),
            "environment": dependency_environment()}


def binding_changes(expected, actual):
    changes = []
    for field in ("research_code", "dependency_specs", "recovery_catalogs"):
        for path in sorted(expected[field].keys() | actual[field].keys()):
            if expected[field].get(path) != actual[field].get(path):
                changes.append({"component": field, "item": path,
                                "expected": expected[field].get(path), "actual": actual[field].get(path)})
    for field in ("python", "implementation", "runtime_packages"):
        if expected["environment"][field] != actual["environment"][field]:
            changes.append({"component": "environment", "item": field,
                            "expected": expected["environment"][field],
                            "actual": actual["environment"][field]})
    return changes


def preflight(cfg, dataset, *, mode="confirmatory", plan=None, root=None, catalogs=None,
              override_reason=None):
    if mode not in {"confirmatory", "exploratory"}:
        raise ValueError("Use confirmatory or exploratory mode")
    upper = len(dataset["tasks"]) * len(cfg["models"]) * (
        cfg["prefix_calls"] + cfg["repeats"] * len(cfg["conditions"]) * cfg["branch_calls"])
    errors, warnings, deviations = [], [], []
    binding = None
    if override_reason is not None and (not override_reason.strip() or mode != "exploratory" or plan is None):
        errors.append("A nonempty deviation reason is allowed only with an exploratory frozen plan")
    providers = {m["provider"] for m in cfg["models"]}
    if len(providers) != 1:
        errors.append("Do not mix offline and live providers")
    live = bool(providers - {"offline"})
    if live:
        for model in cfg["models"]:
            if "REPLACE" in model["model"].upper():
                errors.append(f"Set an explicit model ID for {model['id']}")
        cap = cfg["max_api_calls"]
        if cap <= 0:
            errors.append("Live runs need a positive request cap")
        elif cap < upper:
            message = f"Request cap {cap} is below worst-case bound {upper}"
            (errors if mode == "confirmatory" else warnings).append(message)
    else:
        warnings.append("Offline fixtures test mechanics, not model effectiveness")
    if cfg["repeats"] < 2:
        warnings.append("Fewer than two repeats cannot estimate within-condition variability")
    if plan is not None:
        required = {"schema_version", "study_id", "mode", "config", "dataset_sha256",
                    "config_sha256", "request_upper_bound", "plan_sha256", "binding"}
        if plan.get("schema_version") == 1:
            raise ValueError("Version 1 plans do not bind implementation; re-freeze a new plan")
        if set(plan) != required or plan["schema_version"] != 2:
            raise ValueError("Invalid study plan schema")
        payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
        if digest(payload) != plan["plan_sha256"]:
            raise ValueError("Study plan integrity mismatch")
        if plan["mode"] != mode:
            errors.append("Study mode differs from the frozen plan")
        if plan["config"] != cfg or plan["config_sha256"] != digest(cfg):
            deviations.append({"component": "Configuration", "expected": plan["config_sha256"],
                               "actual": digest(cfg)})
        if plan["dataset_sha256"] != digest(dataset):
            deviations.append({"component": "Tasks", "expected": plan["dataset_sha256"],
                               "actual": digest(dataset)})
        if plan["request_upper_bound"] != upper:
            deviations.append({"component": "Request bound", "expected": plan["request_upper_bound"],
                               "actual": upper})
        if root is None:
            raise ValueError("Repository root is required to verify a frozen plan")
        selected = plan["binding"]["recovery_catalogs"] if catalogs is None else catalogs
        binding = research_binding(root, selected)
        deviations.extend(binding_changes(plan["binding"], binding))
        for change in deviations:
            message = f"{change['component']} differs from the frozen plan"
            if mode == "exploratory" and override_reason and override_reason.strip():
                warnings.append(message + "; explicit exploratory override recorded")
            else:
                errors.append(message)
    elif root is not None:
        binding = research_binding(root, catalogs or ())
    if binding is not None and not binding["recovery_catalogs"]:
        warnings.append("No recovery catalog is bound; supply --catalog if reference checks are planned")
    return {"mode": mode, "request_upper_bound": upper, "request_cap": cfg["max_api_calls"],
            "config_sha256": digest(cfg), "dataset_sha256": digest(dataset),
            "errors": errors, "warnings": warnings, "ready": not errors,
            "binding": binding, "plan_deviations": deviations, "override_reason": override_reason,
            "scope": "Offline consistency check, not model access, power, or semantic validation"}


def freeze_plan(cfg, dataset, destination, *, study_id, root, mode="confirmatory", catalogs=()):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", study_id):
        raise ValueError("Use a nonempty safe study ID")
    report = preflight(cfg, dataset, mode=mode, root=root, catalogs=catalogs)
    if not report["ready"]:
        raise ValueError("; ".join(report["errors"]))
    plan = {"schema_version": 2, "study_id": study_id, "mode": mode,
            "config": deepcopy(cfg), "config_sha256": digest(cfg),
            "dataset_sha256": digest(dataset), "request_upper_bound": report["request_upper_bound"],
            "binding": report["binding"]}
    plan["plan_sha256"] = digest(plan)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return plan
