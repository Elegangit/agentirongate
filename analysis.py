"""Task-weighted paired differences with source-cluster bootstrap intervals."""

from collections import Counter, defaultdict
from itertools import combinations
import csv
import json
from pathlib import Path
import random
import statistics

from .io import digest, read_json, write_json

METRICS = ("completion", "prohibited_proposals", "unauthorized_effect", "attacker_success",
           "approved_send", "model_calls", "provider_error", "latency_seconds")
GATE_CHECKS = {"unauthorized_effect", "attacker_success"}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def percentile(values, p):
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = int(idx)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (idx - lo)


def paired(rows, model, condition, metric, *, samples=1000, seed=17, complete_case=False):
    grouped = defaultdict(dict)
    clusters = {}
    for row in rows:
        if row["model_id"] != model or row["condition"] not in {"minimal", condition}:
            continue
        key = (row["task_id"], row["repeat"])
        if row["condition"] in grouped[key]:
            raise ValueError("Duplicate branch in analysis input")
        grouped[key][row["condition"]] = row
        clusters[row["task_id"]] = row["cluster"]
    task_diffs = defaultdict(list)
    missing, errors, completed = 0, 0, 0
    for (task, _), pair in grouped.items():
        if set(pair) != {"minimal", condition}:
            missing += 1
            continue
        if metric == "completion" and any(
                r["metrics"].get("completed_before_checkpoint", 0) for r in pair.values()):
            completed += 1
            continue
        if complete_case and any(r["termination"] in {"provider_error", "api_budget"} for r in pair.values()):
            errors += 1
            continue
        task_diffs[task].append(pair[condition]["metrics"][metric] - pair["minimal"]["metrics"][metric])
    means = {task: statistics.mean(values) for task, values in task_diffs.items()}
    if not means:
        return {"difference": None, "low": None, "high": None, "tasks": 0, "clusters": 0,
                "missing_pairs": missing, "error_pairs_excluded": errors,
                "precompleted_pairs_excluded": completed}
    groups = defaultdict(list)
    for task, value in means.items():
        groups[clusters[task]].append(value)
    values = list(groups.values())
    rng = random.Random(seed)
    estimates = []
    if len(values) >= 2:
        for _ in range(samples):
            draw = [v for _ in values for v in rng.choice(values)]
            estimates.append(statistics.mean(draw))
    return {"difference": statistics.mean(means.values()),
            "low": percentile(estimates, .025) if estimates else None,
            "high": percentile(estimates, .975) if estimates else None,
            "tasks": len(means), "clusters": len(values), "missing_pairs": missing,
            "error_pairs_excluded": errors, "precompleted_pairs_excluded": completed}


def repeat_variability(rows, model, condition, metric):
    """Task-weighted absolute differences among valid same-condition repeats."""
    grouped = defaultdict(list)
    for row in rows:
        if (row["model_id"] != model or row["condition"] != condition or
                row["termination"] in {"provider_error", "api_budget"}):
            continue
        if metric == "completion" and row["metrics"].get("completed_before_checkpoint", 0):
            continue
        grouped[row["task_id"]].append(row["metrics"][metric])
    values, pairs = [], 0
    for observations in grouped.values():
        diffs = [abs(a - b) for a, b in combinations(observations, 2)]
        if diffs:
            values.append(statistics.mean(diffs))
            pairs += len(diffs)
    return {"model": model, "condition": condition, "metric": metric,
            "mean_absolute_repeat_difference": statistics.mean(values) if values else None,
            "tasks": len(values), "repeat_pairs": pairs,
            "interpretation": "Descriptive variability; overlapping pairs are not independent, "
                              "and this is not a significance threshold."}


def verified_run(run_dir):
    """Read a completed run and check the saved evidence without changing files."""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Refusing to analyze an incomplete run as a completed experiment")
    rows = jsonl(run_dir / "branches.jsonl")
    cfg = manifest["config"]
    if len(rows) != manifest["branches"]:
        raise ValueError("Branch count differs from the manifest")
    if digest(read_json(run_dir / "dataset.json")) != manifest["dataset_hash"]:
        raise ValueError("Dataset integrity mismatch")
    prefixes = jsonl(run_dir / "prefixes.jsonl")
    if len(prefixes) != manifest["prefixes"]:
        raise ValueError("Prefix count differs from the manifest")
    expected = {(p["model_id"], p["task_id"], repeat, condition)
                for p in prefixes if p["termination"] == "eligible"
                for repeat in range(cfg["repeats"]) for condition in cfg["conditions"]}
    keys = [(r["model_id"], r["task_id"], r["repeat"], r["condition"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate matched branch grid")
    for row in rows:
        path = (run_dir / row["trace"]).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("Trace path escapes run directory")
        trace = read_json(path)
        if digest(trace) != row["trace_sha256"] or trace["metrics"] != row["metrics"]:
            raise ValueError("Trace integrity or metric mismatch")
        if any(trace[field] != row[field] for field in ("condition", "termination", "checkpoint_hash")):
            raise ValueError("Trace metadata mismatch")
        if manifest.get("measurement_version", 1) >= 2:
            prefix = next(p for p in prefixes if p["model_id"] == row["model_id"] and
                          p["task_id"] == row["task_id"])
            if row["metrics"].get("completed_before_checkpoint") != prefix["metrics"]["completion"]:
                raise ValueError("Checkpoint completion flag differs from prefix")
    return manifest, rows, prefixes


def analyze(run_dir):
    run_dir = Path(run_dir)
    manifest, rows, prefixes = verified_run(run_dir)
    cfg = manifest["config"]
    comparisons, summary, terminations, termination_rows = [], [], [], []
    variability, task_outcomes, eligibility = [], [], []
    for model in cfg["models"]:
        families = sorted({p["family"] for p in prefixes if p["model_id"] == model["id"]})
        for family in families:
            selected = [p for p in prefixes if p["model_id"] == model["id"] and p["family"] == family]
            eligibility.append({"model": model["id"], "family": family, "tasks": len(selected),
                                "eligible": sum(p["termination"] == "eligible" for p in selected),
                                "completed_prefixes": sum(p["metrics"]["completion"] for p in selected),
                                "eligible_already_complete": sum(
                                    p["termination"] == "eligible" and p["metrics"]["completion"]
                                    for p in selected),
                                "terminations": dict(Counter(p["termination"] for p in selected))})
        for condition in cfg["conditions"]:
            subset = [r for r in rows if r["model_id"] == model["id"] and r["condition"] == condition]
            counts = dict(sorted(Counter(r["termination"] for r in subset).items()))
            terminations.append({"model": model["id"], "condition": condition,
                                 "continuations": len(subset), "counts": counts})
            termination_rows.extend({"model": model["id"], "condition": condition,
                                     "termination": ending, "count": count}
                                    for ending, count in counts.items())
            for metric in METRICS:
                by_task = defaultdict(list)
                for row in subset:
                    by_task[row["task_id"]].append(row["metrics"][metric])
                task_outcomes.extend({"model": model["id"], "condition": condition,
                                      "task_id": task, "metric": metric,
                                      "mean": statistics.mean(values), "repeats": len(values)}
                                     for task, values in by_task.items())
                if metric in {"completion", "prohibited_proposals", "model_calls"}:
                    variability.append(repeat_variability(rows, model["id"], condition, metric))
                summary.append({"model": model["id"], "condition": condition, "metric": metric,
                                "tasks": len(by_task), "continuations": len(subset),
                                "mean": statistics.mean(statistics.mean(v) for v in by_task.values())
                                if by_task else None})
                if condition != "minimal" and metric not in GATE_CHECKS:
                    for complete_case in (False, True):
                        result = paired(rows, model["id"], condition, metric,
                                        samples=cfg["bootstrap_samples"], seed=cfg["seed"],
                                        complete_case=complete_case)
                        comparisons.append({"model": model["id"], "condition": condition,
                                            "metric": metric, "complete_case": complete_case,
                                            "estimate": "primary_complete_case" if complete_case
                                            else "all_attempts_sensitivity", **result})
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)
    write_json(output / "summary.json", {"evidence": manifest["evidence"], "summary": summary,
                                         "measurement_version": manifest.get("measurement_version", 1),
                                         "model_version_audit": manifest.get("model_version_audit"),
                                         "eligibility_by_family": eligibility,
                                         "same_condition_variability": variability,
                                         "branch_terminations": terminations,
                                         "paired": comparisons, "prefix_terminations": manifest["prefix_terminations"],
                                         "caution": "Exploratory topic-cluster intervals over reused attack templates; "
                                                    "not benchmark evidence. Completion contrasts exclude already-completed "
                                                    "checkpoints in measurement version 2; legacy records lack that flag."})
    for filename, data in (("summary.csv", summary), ("paired.csv", comparisons),
                           ("task_outcomes.csv", task_outcomes), ("repeat_variability.csv", variability)):
        if data:
            with (output / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    with (output / "terminations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "condition", "termination", "count"])
        writer.writeheader()
        writer.writerows(termination_rows)
    return output


def plot(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    report = read_json(run_dir / "analysis" / "summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    for ax, metric, title in zip(axes, ("completion", "prohibited_proposals"),
                                 ("Task completion", "Prohibited proposals")):
        entries = [r for r in report["paired"] if r["metric"] == metric and
                   r["complete_case"] and r["difference"] is not None]
        for i, row in enumerate(entries):
            x = row["difference"]
            ax.plot(x, i, "o", color="#0072B2")
            if row["low"] is not None:
                ax.plot([row["low"], row["high"]], [i, i], color="#0072B2")
        ax.set_yticks(range(len(entries)), [r["model"] + ": " + r["condition"] for r in entries])
        ax.axvline(0, color="0.6", linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel("Difference from minimal feedback")
        if not entries:
            ax.text(.5, .5, "No eligible paired observations", ha="center", transform=ax.transAxes)
    name = read_json(run_dir / "manifest.json")["identity"]["name"]
    label = "SCRIPTED FIXTURE CHECK — NOT LIVE RESULTS" if report["evidence"] == "scripted_fixture_only" else "Conditional paired effects (exploratory)"
    fig.suptitle(name + "\n" + label, fontsize=10)
    fig.savefig(run_dir / "analysis" / "paired_effects.pdf")
    fig.savefig(run_dir / "analysis" / "paired_effects.png", dpi=180)
    plt.close(fig)
