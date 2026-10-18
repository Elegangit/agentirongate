from copy import deepcopy
import json

import pytest

from gatebench.analysis import analyze, paired, repeat_variability
from gatebench.audit import export_audit
from gatebench.checkpoint import Checkpoint
from gatebench.experiment import run_experiment
from gatebench.feedback import render
from gatebench.policy import Decision
from gatebench.providers import OfflineProvider, Turn
from gatebench.runner import act, collect_prefix, continue_branch, fidelity_suffix, restore_fidelity
from gatebench.scoring import score
from gatebench.state import State


class Script:
    def __init__(self, calls):
        self.calls = iter(calls)

    def next_turn(self, history):
        call = next(self.calls)
        return Turn({"role": "model", "parts": [{"functionCall": deepcopy(call)}]}, [call])


def test_denial_before_any_read_has_accurate_hint(task, cfg):
    provider = Script([{"name": "read_document", "args": {"document_id": "roster"}}])
    prefix = collect_prefix(task, provider, cfg)
    assert prefix.termination == "eligible" and prefix.state.reads == []
    message = render("directed", "document_not_permitted", task["recovery_hint"])["message"]
    assert "already read" not in message and "if needed" in message


def test_completion_before_denial_is_flagged(task, cfg):
    provider = Script([
        {"name": "read_document", "args": {"document_id": "notes"}},
        {"name": "create_draft", "args": {"recipient": task["goal"]["recipient"],
         "subject": "Update", "body": "; ".join(task["goal"]["required_facts"])}},
        {"name": "send_draft", "args": {"draft_id": "draft-1"}},
        {"name": "read_document", "args": {"document_id": "roster"}},
    ])
    prefix = collect_prefix(task, provider, cfg)
    branch = continue_branch(task, prefix.checkpoint, "minimal", OfflineProvider(task), cfg)
    assert branch["metrics"]["completion"] == 1
    assert branch["metrics"]["completed_before_checkpoint"] == 1
    assert branch["metrics"]["new_completion"] == 0


def rows(values, *, completed=0):
    return [{"task_id": "task", "cluster": "topic", "model_id": "m", "condition": condition,
             "repeat": repeat, "termination": "finished", "metrics": {
                 "completion": value, "completed_before_checkpoint": completed}}
            for condition in ("minimal", "rule") for repeat, value in enumerate(values)]


def test_precompleted_excluded_from_completion_contrast():
    result = paired(rows([1, 1], completed=1), "m", "rule", "completion")
    assert result["difference"] is None and result["precompleted_pairs_excluded"] == 2


@pytest.mark.parametrize("values,expected", [([1, 1, 1], 0), ([0, 1, 1], 2 / 3), ([1], None)])
def test_repeat_variability(values, expected):
    result = repeat_variability(rows(values), "m", "minimal", "completion")
    assert result["mean_absolute_repeat_difference"] == expected


def test_variability_omits_errors_and_precompleted():
    data = rows([0, 1])
    data[0]["termination"] = "provider_error"
    assert repeat_variability(data, "m", "minimal", "completion")["tasks"] == 0
    assert repeat_variability(rows([1, 1], completed=1), "m", "minimal", "completion")["tasks"] == 0


def test_hint_only_has_no_rule_explanation(task):
    result = render("hint", "document_not_permitted", task["recovery_hint"])
    assert result["message"] == "Action denied. " + task["recovery_hint"]
    assert "allowlist" not in result["message"]


def test_capture_detects_serialization_omission(task, monkeypatch):
    state = State.from_task(task)
    state.label = "internal"
    original = State.to_dict
    monkeypatch.setattr(State, "to_dict", lambda self: {k: v for k, v in original(self).items()
                                                      if k != "label"})
    with pytest.raises(ValueError, match="original"):
        Checkpoint.capture(state, [], {}, "test")


def test_original_fidelity_probe_preserves_original(task, cfg):
    task["documents"]["notes"]["label"] = "confidential"
    prefix = collect_prefix(task, OfflineProvider(task), cfg)
    before = deepcopy(prefix.state)
    assert restore_fidelity(prefix.checkpoint, fidelity_suffix(task, prefix.state),
                            original=prefix.state, task=task)
    assert prefix.state == before
    altered = deepcopy(before)
    altered.next_draft += 1
    with pytest.raises(ValueError, match="Restore mismatch"):
        restore_fidelity(prefix.checkpoint, [], original=altered)


def test_real_faulty_gate_is_detected(task, monkeypatch):
    monkeypatch.setattr("gatebench.engine.evaluate", lambda *a, **kw: Decision("allow", "fault"))
    state = State.from_task(task)
    event, _ = act(task, state, {"name": "read_document", "args": {"document_id": "roster"}}, "ideal")
    assert event["decision"] == "allow" and event["oracle_prohibited"]
    assert score(task, state, [event])["unauthorized_effect"] == 1


def test_invalid_proposal_counts_before_stop(task, cfg):
    prefix = collect_prefix(task, OfflineProvider(task), cfg)
    provider = Script([{"name": "read_document", "args": {"document_id": 42}}])
    branch = continue_branch(task, prefix.checkpoint, "minimal", provider, cfg)
    assert branch["termination"] == "invalid_call" and branch["metrics"]["proposals"] == 1


def test_audit_and_analysis_end_to_end(root, task, cfg, tmp_path):
    run = tmp_path / "run"
    config = {**cfg, "repeats": 2, "conditions": ["minimal", "rule", "directed", "hint"]}
    run_experiment(root, config, {"schema_version": 1, "tasks": [task]}, run)
    analyze(run)
    report = json.loads((run / "analysis/summary.json").read_text())
    assert report["measurement_version"] == 2
    assert report["eligibility_by_family"][0]["eligible"] == 1
    assert all(r["mean_absolute_repeat_difference"] == 0 for r in report["same_condition_variability"])
    audit = export_audit(run, tmp_path / "audit")
    packet = json.loads((audit / "rater_cases.json").read_text())
    assert len(packet["cases"]) == 8
    for case in packet["cases"]:
        assert not {"condition", "model_id", "metrics", "feedback", "history"} & case.keys()
        assert case["outbox"]
    assert len(json.loads((audit / "PRIVATE_mapping.json").read_text())) == 8
    with pytest.raises(FileExistsError):
        export_audit(run, audit)
