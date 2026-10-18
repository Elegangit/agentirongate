"""Run experiments and save each run in a new directory."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import random
import subprocess
import sys

from .config import identity
from .io import append_jsonl, digest, write_json
from .providers import RequestBudget, make_provider
from .runner import collect_prefix, continue_branch, fidelity_suffix, restore_fidelity
from .scoring import score
from .study import preflight
from .model_versions import ModelVersionAudit


def source_hashes(root):
    files = []
    for name in ("src", "configs", "data", "tests"):
        files.extend((Path(root) / name).rglob("*"))
    files += [Path(root) / n for n in ("project.toml", "pyproject.toml", "environment.yml")]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files) if p.is_file() and "__pycache__" not in p.parts}


def environment(root):
    packages = {}
    for name in ("httpx", "matplotlib", "pytest", "ruff", "setuptools"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "git_commit": commit}


def run_experiment(root, cfg, dataset, output, *, allow_network=False, provider_factory=None,
                   study_plan=None, override_reason=None):
    if override_reason is not None and study_plan is None:
        raise ValueError("A deviation override requires an exploratory frozen plan")
    if study_plan is not None:
        check = preflight(cfg, dataset, mode=study_plan.get("mode"), plan=study_plan,
                          root=root, override_reason=override_reason)
        if not check["ready"]:
            raise ValueError("; ".join(check["errors"]))
    providers = {m["provider"] for m in cfg["models"]}
    if len(providers) > 1:
        raise ValueError("Do not mix offline and live providers in one run")
    live = "gemini" in providers
    if live and not allow_network:
        raise ValueError("Live requests require --allow-network")
    if live and cfg["max_api_calls"] == 0:
        raise ValueError("Live requests require a positive API request cap")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    factory = provider_factory or make_provider
    budget = RequestBudget(cfg["max_api_calls"])
    manifest = {"schema_version": 1, "measurement_version": 2,
                "status": "running", "identity": identity(root),
                "evidence": "live_model" if live else "scripted_fixture_only",
                "started_at": datetime.now(timezone.utc).isoformat(), "config": cfg,
                "dataset_hash": digest(dataset), "source_hashes": source_hashes(Path(root)),
                "environment": environment(root), "api_requests": 0}
    if study_plan is not None:
        manifest["study_plan"] = study_plan
        manifest["study_verification"] = check
    write_json(output / "manifest.json", manifest)
    write_json(output / "dataset.json", dataset)
    prefixes, branches = [], []
    rng = random.Random(cfg["seed"])
    version_audit = ModelVersionAudit(cfg["models"])
    try:
        for model in cfg["models"]:
            for task in dataset["tasks"]:
                provider = factory(task, model, budget, allow_network)
                try:
                    prefix = collect_prefix(task, provider, cfg)
                    version_audit.observe(model["id"], prefix.turns)
                    key = f"{model['id']}--{task['id']}"
                    summary = {"task_id": task["id"], "cluster": task["cluster"],
                               "family": task["family"], "model_id": model["id"],
                               "termination": prefix.termination, "model_calls": prefix.calls,
                               "metrics": score(task, prefix.state, prefix.state.events)}
                    prefixes.append(summary)
                    append_jsonl(output / "prefixes.jsonl", summary)
                    write_json(output / "prefixes" / f"{key}.json",
                               {**summary, "history": prefix.history, "turns": prefix.turns,
                                "state": prefix.state.to_dict()})
                    if prefix.checkpoint is None:
                        continue
                    cp = prefix.checkpoint
                    restore_fidelity(cp, fidelity_suffix(task, prefix.state),
                                     original=prefix.state, task=task, approver=cfg["approver"])
                    write_json(output / "checkpoints" / f"{key}.json", cp.to_dict())
                    for repeat in range(cfg["repeats"]):
                        order = list(cfg["conditions"])
                        rng.shuffle(order)
                        for position, condition in enumerate(order):
                            # Each branch gets its own restored history.
                            branch = continue_branch(task, cp, condition, provider, cfg)
                            version_audit.observe(model["id"], branch["turns"])
                            row = {"task_id": task["id"], "cluster": task["cluster"],
                                   "family": task["family"], "model_id": model["id"],
                                   "repeat": repeat, "order": position, "condition": condition,
                                   "termination": branch["termination"], "metrics": branch["metrics"],
                                   "checkpoint_hash": cp.sha256}
                            trace_path = f"branches/{key}--{repeat}--{condition}.json"
                            row["trace"] = trace_path
                            row["trace_sha256"] = digest(branch)
                            write_json(output / trace_path, branch)
                            append_jsonl(output / "branches.jsonl", row)
                            branches.append(row)
                finally:
                    if hasattr(provider, "close"):
                        provider.close()
        exhausted = any(p["termination"] == "api_budget" for p in prefixes + branches)
        manifest["status"] = "budget_exhausted" if exhausted else "complete"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        manifest["api_requests"] = budget.used
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["prefix_terminations"] = dict(Counter(p["termination"] for p in prefixes))
        manifest["prefixes"] = len(prefixes)
        manifest["branches"] = len(branches)
        manifest["model_version_audit"] = version_audit.report()
        write_json(output / "manifest.json", manifest)
    return manifest
