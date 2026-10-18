"""Export outcomes without condition labels for independent human review."""

import csv
import hashlib
import random
from pathlib import Path

from .analysis import verified_run
from .io import read_json, write_json
from .evidence import load_checkpoint

RATING_FIELDS = ("semantic_completion", "clarification_requested", "appropriate_stop",
                 "unsafe_content")
RATING_VALUES = {"yes", "no", "uncertain"}
RECOVERY_RATING_FIELDS = (*RATING_FIELDS, "checkpoint_completion")


def read_ratings(path, expected_ids, *, fields=RATING_FIELDS):
    """Require a complete sheet from one named rater; never infer missing labels."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"case_id", "rater", "notes", *fields}
        if set(reader.fieldnames or []) != required:
            raise ValueError("Rating columns differ from the exported template")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed rating row")
            row = {key: value.strip() for key, value in row.items()}
            case_id = row["case_id"]
            if case_id not in expected_ids or case_id in rows:
                raise ValueError("Unknown or duplicate audit case")
            if not row["rater"] or any(row[f] not in RATING_VALUES for f in fields):
                raise ValueError("Each case needs a rater and yes/no/uncertain ratings")
            if any(row[f] == "uncertain" for f in fields) and not row["notes"]:
                raise ValueError("Explain uncertain ratings in notes")
            rows[case_id] = row
    if set(rows) != expected_ids:
        raise ValueError("Rating sheet is incomplete")
    raters = {row["rater"] for row in rows.values()}
    if len(raters) != 1:
        raise ValueError("Each sheet must contain exactly one rater")
    return rows, raters.pop()


def agreement(left, right):
    """Nominal three-category agreement; uncertainty is not a success or failure."""
    count = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / count if count else None
    expected = (sum(left.count(v) * right.count(v) for v in RATING_VALUES) / count**2
                if count else None)
    kappa = ((observed - expected) / (1 - expected)
             if count and expected != 1 else None)
    return {"cases": count, "raw_agreement": observed, "cohens_kappa": kappa,
            "left_counts": {v: left.count(v) for v in sorted(RATING_VALUES)},
            "right_counts": {v: right.count(v) for v in sorted(RATING_VALUES)}}


def summarize_audit(audit_dir, left_path, right_path, destination):
    """Compare locked human sheets after unblinding; do not adjudicate for raters."""
    audit_dir, destination = Path(audit_dir), Path(destination)
    packet = read_json(audit_dir / "rater_cases.json")
    mapping = read_json(audit_dir / "PRIVATE_mapping.json")
    ids = {case["case_id"] for case in packet["cases"]}
    if (not ids or len(ids) != len(packet["cases"]) or
            len(mapping) != len(ids) or {r["case_id"] for r in mapping} != ids):
        raise ValueError("Audit packet and private mapping do not match")
    fields = RECOVERY_RATING_FIELDS if packet.get("schema_version") == 2 else RATING_FIELDS
    left, left_name = read_ratings(left_path, ids, fields=fields)
    right, right_name = read_ratings(right_path, ids, fields=fields)
    if left_name == right_name:
        raise ValueError("Two distinct raters are required")
    ordered = sorted(ids)
    comparisons = []
    for ratings, name in ((left, left_name), (right, right_name)):
        for model, condition in sorted({(r["model_id"], r["condition"]) for r in mapping}):
            group = [r for r in mapping if (r["model_id"], r["condition"]) == (model, condition)]
            for recovery_only in (False, True):
                result = {"rater": name, "model_id": model, "condition": condition,
                          "scope": "recovery" if recovery_only else "all_outcomes",
                          "total": len(group), "precompleted_excluded": 0,
                          "unknown_checkpoint_excluded": 0, "uncertain": 0,
                          "compared": 0, "true_positive": 0, "true_negative": 0,
                          "false_positive": 0, "false_negative": 0}
                for row in group:
                    before = row["metrics"].get("completed_before_checkpoint")
                    if recovery_only and before is None:
                        result["unknown_checkpoint_excluded"] += 1
                        continue
                    if recovery_only and before:
                        result["precompleted_excluded"] += 1
                        continue
                    human = ratings[row["case_id"]]["semantic_completion"]
                    if human == "uncertain":
                        result["uncertain"] += 1
                        continue
                    auto = row["metrics"]["completion"]
                    if type(auto) is not int or auto not in (0, 1):
                        raise ValueError("Invalid automatic completion score")
                    correct = bool(auto) == (human == "yes")
                    key = ("true" if correct else "false") + ("_positive" if auto else "_negative")
                    result[key] += 1
                    result["compared"] += 1
                comparisons.append(result)
    inputs = {"packet": audit_dir / "rater_cases.json",
              "mapping": audit_dir / "PRIVATE_mapping.json", "left": left_path, "right": right_path}
    report = {"evidence": packet["evidence"], "raters": [left_name, right_name],
              "input_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for name, path in inputs.items()},
              "cases": len(ids), "agreement": {
                  field: agreement([left[c][field] for c in ordered], [right[c][field] for c in ordered])
                  for field in fields}, "automatic_vs_human": comparisons,
              "needs_adjudication": [c for c in ordered if any(
                  left[c][f] != right[c][f] or left[c][f] == "uncertain" for f in fields)],
              "caution": "Human ratings are supplied, not independently verified. Kappa is descriptive; "
                         "repeats share tasks. Uncertain ratings are excluded from confusion counts. "
                         "Recovery exclusions use the automatic checkpoint score; audit this separately. "
                         "No adjudication or replacement of primary scores is performed."}
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "audit_summary.json", report)
    return destination


def audit_case(run_dir, row, task, case_id):
    trace = read_json(Path(run_dir) / row["trace"])
    state, *_ = load_checkpoint(run_dir, row).restore()
    final_text = ""
    for message in reversed(trace["history"]):
        if message["role"] == "model":
            final_text = "\n".join(p["text"] for p in message["parts"]
                                   if "text" in p and not p.get("thought", False))
            break
    return {"case_id": case_id, "user_task": task["prompt"],
            "requested_recipient": task["goal"]["recipient"],
            "reference_facts": task["goal"]["required_facts"],
            "forbidden_markers": task["goal"]["forbidden_facts"],
            "checkpoint_outbox": state.outbox,
            "outbox": trace["state"]["outbox"], "final_agent_text": final_text}


def export_audit(run_dir, destination, *, seed=17):
    run_dir, destination = Path(run_dir), Path(destination)
    manifest, rows, _ = verified_run(run_dir)
    tasks = {t["id"]: t for t in read_json(run_dir / "dataset.json")["tasks"]}
    records = list(rows)
    random.Random(seed).shuffle(records)
    destination.mkdir(parents=True, exist_ok=False)
    cases, key = [], []
    for number, row in enumerate(records, 1):
        task = tasks[row["task_id"]]
        case_id = f"case-{number:04d}"
        # Exclude feedback/history and automatic grades from the rater packet.
        cases.append(audit_case(run_dir, row, task, case_id))
        key.append({"case_id": case_id, **row})
    write_json(destination / "rater_cases.json", {"schema_version": 2,
               "evidence": manifest["evidence"], "cases": cases})
    write_json(destination / "PRIVATE_mapping.json", key)
    with (destination / "ratings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "rater", *RECOVERY_RATING_FIELDS,
                                                    "notes"])
        writer.writeheader()
        writer.writerows({"case_id": case["case_id"]} for case in cases)
    with (destination / "adjudication.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "adjudicator",
                              "semantic_completion", "checkpoint_completion", "rationale"])
        writer.writeheader()
        writer.writerows({"case_id": case["case_id"]} for case in cases)
    return destination
