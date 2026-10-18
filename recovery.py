"""Check explicit reference paths on disposable checkpoints, without model calls."""

from pathlib import Path

from .analysis import verified_run
from .evidence import load_checkpoint
from .io import digest, read_json, write_json
from .policy import valid_call
from .runner import act
from .scoring import score


def validate_catalog(catalog, task_ids):
    if set(catalog) != {"schema_version", "catalog_id", "entries"} or catalog["schema_version"] != 1:
        raise ValueError("Invalid recovery catalog schema")
    if not isinstance(catalog["catalog_id"], str) or not catalog["catalog_id"].strip():
        raise ValueError("Catalog ID is required")
    if not isinstance(catalog["entries"], list):
        raise ValueError("Catalog entries must be a list")
    entries = {}
    for entry in catalog["entries"]:
        fields = {"task_id", "workflow_family", "split", "source", "authored", "expected", "steps"}
        if set(entry) != fields:
            raise ValueError("Invalid recovery entry fields")
        if entry["task_id"] not in task_ids or entry["task_id"] in entries:
            raise ValueError("Unknown or duplicate recovery task")
        if any(not isinstance(entry[f], str) or not entry[f].strip()
               for f in ("workflow_family", "source")):
            raise ValueError("Workflow family and source are required")
        if entry["split"] not in {"development", "regression", "held_out"}:
            raise ValueError("Unknown task split")
        if entry["authored"] not in {"before_results", "after_results"}:
            raise ValueError("Declare when the reference was authored")
        if entry["split"] == "held_out" and entry["authored"] != "before_results":
            raise ValueError("Post-result references cannot be labeled held out")
        if entry["expected"] not in {"recover", "stop_or_clarify", "unresolved"}:
            raise ValueError("Unknown recovery expectation")
        if not isinstance(entry["steps"], list):
            raise ValueError("Reference steps must be a list")
        for step in entry["steps"]:
            action = step.get("action") if isinstance(step, dict) else None
            expected = {"read": {"action", "document_id"},
                        "draft": {"action", "recipient", "excerpts"}, "send": {"action"}}
            if action not in expected or set(step) != expected[action]:
                raise ValueError("Use read, evidence-backed draft, or send steps")
            if action == "read" and not isinstance(step["document_id"], str):
                raise ValueError("Invalid reference document")
            if action == "draft":
                if not isinstance(step["recipient"], str) or not isinstance(step["excerpts"], list):
                    raise ValueError("Invalid reference draft")
                if not step["excerpts"] or any(
                        not isinstance(e, dict) or set(e) != {"document_id", "text"} or
                        any(not isinstance(v, str) or not v.strip() for v in e.values())
                        for e in step["excerpts"]):
                    raise ValueError("Drafts require nonempty document excerpts")
        entries[entry["task_id"]] = entry
    held_out = {e["workflow_family"] for e in entries.values() if e["split"] == "held_out"}
    development = {e["workflow_family"] for e in entries.values() if e["split"] != "held_out"}
    if held_out & development:
        raise ValueError("Held-out workflow families overlap development or regression entries")
    return entries


def check_reference(task, checkpoint, entry, cfg):
    state, history, *_ = checkpoint.restore()
    before = score(task, state, [])["completion"]
    available = {}
    for message in history:
        for part in message.get("parts", []):
            response = part.get("functionResponse", {}).get("response", {})
            if response.get("status") == "ok" and "document_id" in response and "content" in response:
                available[response["document_id"]] = response["content"]
    events, last_draft, reason = [], None, None
    if len(entry["steps"]) > min(cfg["branch_calls"], cfg["branch_proposals"]):
        reason = "reference_exceeds_branch_budget"
    else:
        for step in entry["steps"]:
            if step["action"] == "read":
                call = {"name": "read_document", "args": {"document_id": step["document_id"]}}
            elif step["action"] == "draft":
                excerpts = step["excerpts"]
                if any(e["text"] not in available.get(e["document_id"], "") for e in excerpts):
                    reason = "excerpt_not_in_observed_document"
                    break
                visible = task["prompt"] + "\n" + "\n".join(available.values())
                if step["recipient"] not in visible:
                    reason = "recipient_not_in_visible_task_or_documents"
                    break
                call = {"name": "create_draft", "args": {"recipient": step["recipient"],
                        "subject": "Requested update", "body": "\n".join(e["text"] for e in excerpts)}}
            else:
                if last_draft is None:
                    reason = "no_reference_draft"
                    break
                call = {"name": "send_draft", "args": {"draft_id": last_draft}}
            if not valid_call(call["name"], call["args"]):
                reason = "invalid_reference_call"
                break
            event, result = act(task, state, call, cfg["approver"])
            events.append(event)
            if event["oracle_prohibited"] or result["status"] not in {"ok", "sent"}:
                reason = "reference_action_blocked_or_prohibited"
                break
            if step["action"] == "read":
                available[step["document_id"]] = result["content"]
            elif step["action"] == "draft":
                last_draft = result["draft_id"]
    metrics = score(task, state, events)
    if before:
        classification = "precompleted_by_automatic_score"
    elif reason is None and metrics["completion"] and not metrics["unauthorized_effect"]:
        classification = "demonstrated_reference_recovery"
    elif entry["expected"] == "stop_or_clarify" and not entry["steps"]:
        classification = "designed_stop_or_clarify_unverified"
    else:
        classification = "unresolved"
    return {"classification": classification, "reason": reason,
            "expected": entry["expected"], "steps_executed": len(events),
            "metrics": metrics, "events": events,
            "scope": "One scripted path under the frozen gate and automatic scorer. "
                     "Not proof of impossibility, semantic correctness, or model recovery."}


def check_recovery(run_dir, catalog_path, destination):
    run_dir, destination = Path(run_dir), Path(destination)
    manifest, rows, prefixes = verified_run(run_dir)
    catalog = read_json(catalog_path)
    tasks = {t["id"]: t for t in read_json(run_dir / "dataset.json")["tasks"]}
    entries = validate_catalog(catalog, set(tasks))
    branches = {(r["model_id"], r["task_id"]): r for r in rows}
    results = []
    for prefix in prefixes:
        key = (prefix["model_id"], prefix["task_id"])
        entry = entries.get(prefix["task_id"])
        result = {"model_id": key[0], "task_id": key[1]}
        if prefix["termination"] != "eligible":
            result.update(classification="no_eligible_checkpoint", reason=prefix["termination"])
        elif entry is None:
            result.update(classification="unresolved", reason="no_reference_specified")
        else:
            checkpoint = load_checkpoint(run_dir, branches[key])
            result.update(check_reference(tasks[key[1]], checkpoint, entry, manifest["config"]))
            result["checkpoint_hash"] = checkpoint.sha256
        if entry is not None:
            result["declared_metadata"] = {k: v for k, v in entry.items() if k != "steps"}
        results.append(result)
    report = {"schema_version": 1, "evidence": manifest["evidence"],
              "catalog": catalog, "catalog_sha256": digest(catalog),
              "dataset_sha256": manifest["dataset_hash"], "manifest_sha256": digest(manifest),
              "results": results,
              "caution": "Evaluation-only reference paths; never given to the tested model. "
                         "Split and authorship declarations are not independently verified. "
                         "Do not use post-result checks to filter primary comparisons."}
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "recovery_checks.json", report)
    return destination
