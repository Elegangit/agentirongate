"""Analyze supplied human judgments without replacing automatic scores."""

from copy import deepcopy
import csv
import hashlib
from pathlib import Path

from .analysis import paired, verified_run
from .audit import audit_case, read_ratings, RECOVERY_RATING_FIELDS, RATING_VALUES
from .io import read_json, write_json
from .cohort import common_cohort


def read_adjudication(path, ids):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = {"case_id", "adjudicator", "semantic_completion", "checkpoint_completion", "rationale"}
        if set(reader.fieldnames or []) != fields:
            raise ValueError("Adjudication columns differ from template")
        rows = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed adjudication row")
            row = {k: v.strip() for k, v in row.items()}
            key = row["case_id"]
            if key not in ids or key in rows:
                raise ValueError("Unknown or duplicate adjudication case")
            if (not row["adjudicator"] or not row["rationale"] or
                    any(row[f] not in RATING_VALUES
                        for f in ("semantic_completion", "checkpoint_completion"))):
                raise ValueError("Every adjudication needs labels, an adjudicator, and rationale")
            rows[key] = row
    if set(rows) != ids:
        raise ValueError("Adjudication is incomplete; use uncertain rather than dropping cases")
    return rows


def human_analysis(run_dir, audit_dir, left_path, right_path, adjudication_path, destination):
    run_dir, audit_dir, destination = Path(run_dir), Path(audit_dir), Path(destination)
    manifest, rows, _ = verified_run(run_dir)
    packet = read_json(audit_dir / "rater_cases.json")
    mapping = read_json(audit_dir / "PRIVATE_mapping.json")
    if packet.get("schema_version") != 2 or packet["evidence"] != manifest["evidence"]:
        raise ValueError("Re-export a version 2 audit packet from this run")
    ids = {c["case_id"] for c in packet["cases"]}
    if (not ids or len(ids) != len(packet["cases"]) or len(mapping) != len(ids) or
            {r["case_id"] for r in mapping} != ids):
        raise ValueError("Audit packet and mapping do not match")
    # Require the full exported run; no condition-based selection after grading.
    by_trace = {r["trace"]: r for r in rows}
    if len(mapping) != len(rows) or {r["trace"] for r in mapping} != set(by_trace):
        raise ValueError("Audit must cover the full run")
    tasks = {t["id"]: t for t in read_json(run_dir / "dataset.json")["tasks"]}
    cases = {c["case_id"]: c for c in packet["cases"]}
    for row in mapping:
        if {k: v for k, v in row.items() if k != "case_id"} != by_trace[row["trace"]]:
            raise ValueError("Audit mapping differs from verified run")
        if cases[row["case_id"]] != audit_case(run_dir, row, tasks[row["task_id"]], row["case_id"]):
            raise ValueError("Audit evidence differs from verified run")
    left, left_name = read_ratings(left_path, ids, fields=RECOVERY_RATING_FIELDS)
    right, right_name = read_ratings(right_path, ids, fields=RECOVERY_RATING_FIELDS)
    if left_name == right_name:
        raise ValueError("Two distinct raters are required")
    decisions = read_adjudication(adjudication_path, ids)
    checkpoint_labels = {}
    human, matched_auto, exclusions, disagreements = [], [], [], []
    for row in mapping:
        key = row["case_id"]
        decision = decisions[key]
        before = decision["checkpoint_completion"]
        cp_key = (row["model_id"], row["task_id"], row["checkpoint_hash"])
        if cp_key in checkpoint_labels and checkpoint_labels[cp_key] != before:
            raise ValueError("Adjudicated labels differ for the same checkpoint")
        checkpoint_labels[cp_key] = before
        if any(left[key][f] != right[key][f] or left[key][f] == "uncertain"
               for f in ("semantic_completion", "checkpoint_completion")):
            disagreements.append(key)
        if before == "uncertain" or decision["semantic_completion"] == "uncertain":
            exclusions.append({"case_id": key, "reason": "uncertain_human_rating"})
            continue
        rated = deepcopy(by_trace[row["trace"]])
        rated["metrics"]["completion"] = int(decision["semantic_completion"] == "yes")
        rated["metrics"]["completed_before_checkpoint"] = int(before == "yes")
        human.append(rated)
        matched_auto.append(by_trace[row["trace"]])
    estimates, common_estimates, inclusion_ledger = [], [], []
    for model in manifest["config"]["models"]:
        for condition in manifest["config"]["conditions"]:
            if condition == "minimal":
                continue
            for complete_case in (True, False):
                for label, selected in (("human_adjudicated", human),
                                        ("automatic_same_rated_rows", matched_auto),
                                        ("automatic_full_run", rows)):
                    result = paired(selected, model["id"], condition, "completion",
                                    complete_case=complete_case,
                                    samples=manifest["config"]["bootstrap_samples"],
                                    seed=manifest["config"]["seed"])
                    estimates.append({"model_id": model["id"], "condition": condition,
                                      "scoring": label, "complete_case": complete_case, **result})
                comparison, ledger = common_cohort(
                    mapping, decisions, model["id"], condition, complete_case=complete_case,
                    samples=manifest["config"]["bootstrap_samples"], seed=manifest["config"]["seed"])
                common_estimates.append(comparison)
                inclusion_ledger.extend(ledger)
    paths = {"packet": audit_dir / "rater_cases.json", "mapping": audit_dir / "PRIVATE_mapping.json",
             "left": left_path, "right": right_path, "adjudication": adjudication_path,
             "manifest": run_dir / "manifest.json"}
    report = {"schema_version": 2, "evidence": manifest["evidence"], "raters": [left_name, right_name],
              "input_sha256": {k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in paths.items()},
              "cases": len(ids), "excluded_uncertain": exclusions,
              "reviewer_disagreement_or_uncertainty": disagreements,
              "adjudications": [decisions[k] for k in sorted(ids)], "paired_completion": estimates,
              "common_eligibility": common_estimates,
              "inclusion_ledger_file": "inclusion_ledger.json",
              "scope": "Full-run human sensitivity analysis, not replacement of primary scores. "
                       "Rater independence and semantic correctness are not verified by software. "
                       "Uncertain rows can cause additional unmatched-pair exclusions. "
                       "Automatic and human checkpoint exclusions may differ."}
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "human_sensitivity.json", report)
    write_json(destination / "inclusion_ledger.json", {"schema_version": 1,
               "evidence": manifest["evidence"], "pairs": inclusion_ledger,
               "scope": "One row per model, contrast, task, repeat, and error-handling setting. "
                        "Multiple exclusion reasons can apply. Private, unblinded audit data."})
    return destination
