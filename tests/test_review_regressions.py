from copy import deepcopy

import httpx
import pytest

from gatebench.analysis import analyze, paired
from gatebench.experiment import run_experiment
from gatebench.io import read_json
from gatebench.providers import GeminiProvider, OfflineProvider, RequestBudget, Turn
from gatebench.runner import act, collect_prefix, continue_branch
from gatebench.scoring import prohibited, score
from gatebench.state import State


@pytest.mark.parametrize("candidate", [
    {"finishReason": "STOP"},
    {"finishReason": "STOP", "content": {}},
    {"finishReason": "STOP", "content": {"role": "model"}},
    {"finishReason": "STOP", "content": {"role": "model", "parts": []}},
])
def test_empty_stop_shapes(task, monkeypatch, candidate):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    data = {"candidates": [candidate], "modelVersion": "fixture-version",
            "usageMetadata": {"candidatesTokenCount": 0}}
    original = deepcopy(data)
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=data))) as client:
        provider = GeminiProvider(task, {"model": "fixture", "temperature": 0,
                                        "max_output_tokens": 100}, RequestBudget(1),
                                  allow_network=True, client=client)
        turn = provider.next_turn([])
    assert turn.termination == "empty_response" and not turn.calls
    assert turn.finish_reason == "STOP" and turn.model_version == "fixture-version"
    assert turn.usage == data["usageMetadata"] and data == original


@pytest.mark.parametrize("content", [None, [], "invalid", {"parts": None}, {"parts": {}}])
def test_wrong_content_types_stay_malformed(content):
    with pytest.raises((ValueError, TypeError, KeyError)):
        GeminiProvider.parse({"candidates": [{"finishReason": "STOP", "content": content}]})


def test_unknown_document_is_operational(task):
    state = State.from_task(task)
    before = state.to_dict()
    assert not prohibited(task, before, "read_document", {"document_id": "missing"})
    assert prohibited(task, before, "read_document", {"document_id": "roster"})
    event, result = act(task, state, {"name": "read_document",
                                    "args": {"document_id": "missing"}}, "ideal")
    assert event["rule"] == "unknown_document" and result["status"] == "denied"
    metrics = score(task, state, [event])
    assert metrics["prohibited_proposals"] == 0
    assert metrics["unknown_document_proposals"] == 1


@pytest.mark.parametrize("ending", ["empty_response", "refusal", "invalid_call",
                                    "output_limit", "provider_stop:OTHER"])
def test_model_endings_remain_in_complete_case(ending):
    rows = [{"task_id": "a", "cluster": "c", "model_id": "m", "repeat": 0,
             "condition": condition, "termination": ending, "metrics": {"completion": 0}}
            for condition in ("minimal", "rule")]
    result = paired(rows, "m", "rule", "completion", complete_case=True)
    assert result["tasks"] == 1 and result["error_pairs_excluded"] == 0


def test_termination_counts_by_model_and_arm(root, cfg, task, tmp_path):
    class EndingProvider(OfflineProvider):
        def __init__(self, task, ending):
            super().__init__(task)
            self.ending = ending

        def next_turn(self, history):
            if any(p.get("functionResponse", {}).get("response", {}).get("status") == "denied"
                   for m in history for p in m["parts"]):
                return Turn({"role": "model", "parts": []}, termination=self.ending)
            return super().next_turn(history)

    cfg = {**cfg, "repeats": 2, "bootstrap_samples": 10,
           "models": [{**cfg["models"][0], "id": name} for name in ("empty", "refusal")]}
    output = tmp_path / "endings"
    run_experiment(root, cfg, {"schema_version": 1, "tasks": [task]}, output,
                   provider_factory=lambda task, model, *args: EndingProvider(
                       task, "empty_response" if model["id"] == "empty" else "refusal"))
    analyze(output)
    report = read_json(output / "analysis/summary.json")
    for row in report["branch_terminations"]:
        expected = "empty_response" if row["model"] == "empty" else "refusal"
        assert row["counts"] == {expected: 2} and row["continuations"] == 2
    assert len(report["branch_terminations"]) == 2 * len(cfg["conditions"])
    assert (output / "analysis/terminations.csv").exists()


def test_invalid_branch_stops_without_tool_effect(task, cfg):
    cp = collect_prefix(task, OfflineProvider(task), cfg).checkpoint

    class InvalidProvider:
        def next_turn(self, history):
            call = {"name": "read_document", "args": {"document_id": "notes", "extra": "x"}}
            return Turn({"role": "model", "parts": [{"functionCall": call}]}, [call])

    branch = continue_branch(task, cp, "rule", InvalidProvider(), cfg)
    assert branch["termination"] == "invalid_call"
    assert len(branch["state"]["events"]) == branch["post_checkpoint_event_start"]
    assert branch["metrics"]["provider_error"] == 0


@pytest.mark.parametrize("condition", ["minimal", "rule", "directed", "neutral"])
def test_unknown_reads_get_same_feedback_and_separate_count(task, cfg, condition):
    cp = collect_prefix(task, OfflineProvider(task), cfg).checkpoint

    class UnknownReader:
        def next_turn(self, history):
            call = {"name": "read_document", "args": {"document_id": "missing"}}
            return Turn({"role": "model", "parts": [{"functionCall": call}]}, [call])

    branch = continue_branch(task, cp, condition, UnknownReader(), {**cfg, "branch_calls": 2})
    assert branch["metrics"]["prohibited_proposals"] == 0
    assert branch["metrics"]["unknown_document_proposals"] == 2
    response = branch["history"][-1]["parts"][0]["functionResponse"]["response"]
    assert response == {"status": "denied", "message": "Action denied."}


def test_allowed_draft_is_not_a_prohibited_send(task):
    task = deepcopy(task)
    task["documents"]["notes"]["label"] = "confidential"
    task["policy"]["recipients"].append("sponsor@outside.test")
    state = State.from_task(task)
    act(task, state, {"name": "read_document", "args": {"document_id": "notes"}}, "ideal")
    draft, _ = act(task, state, {"name": "create_draft", "args": {
        "recipient": "sponsor@outside.test", "subject": "Update", "body": "Example"}}, "ideal")
    assert not draft["oracle_prohibited"] and draft["decision"] == "allow"
    send, _ = act(task, state, {"name": "send_draft", "args": {"draft_id": "draft-1"}}, "ideal")
    assert send["oracle_prohibited"] and send["decision"] == "deny"
    assert not state.outbox


def test_missing_parts_ends_branch_as_model_outcome(task, cfg):
    cp = collect_prefix(task, OfflineProvider(task), cfg).checkpoint

    class EmptyProvider:
        def next_turn(self, history):
            return GeminiProvider.parse({"candidates": [{"finishReason": "STOP"}]})

    branch = continue_branch(task, cp, "minimal", EmptyProvider(), cfg)
    assert branch["termination"] == "empty_response"
    assert branch["metrics"]["provider_error"] == 0
    assert branch["turns"][0]["finish_reason"] == "STOP"
