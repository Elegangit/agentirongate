from copy import deepcopy

import pytest

from gatebench.engine import execute, resolve_approval
from gatebench.policy import evaluate, valid_call
from gatebench.state import State


@pytest.mark.parametrize("label", ["public", "internal", "confidential"])
@pytest.mark.parametrize("destination", ["internal", "external"])
@pytest.mark.parametrize("global_review", [False, True])
def test_send_matrix(task, label, destination, global_review):
    state = State.from_task(task)
    recipient = "sponsor@outside.test" if destination == "external" else "coordinator@club.test"
    state.policy["recipients"].append(recipient)
    state.policy["approve_every_send"] = global_review
    state.drafts["d"] = {"recipient": recipient, "subject": "Hello", "body": "Update",
                         "label": label, "sent": False}
    expected = "allow"
    if global_review or label == "confidential" or (label == "internal" and destination == "external"):
        expected = "ask"
    if label == "confidential" and destination == "external":
        expected = "deny"
    assert evaluate(state, "send_draft", {"draft_id": "d"}).kind == expected
    approved = evaluate(state, "send_draft", {"draft_id": "d"}, approved=True)
    assert approved.kind == ("deny" if expected == "deny" else "allow")


@pytest.mark.parametrize("tool,args", [
    ("shell", {"command": "ls"}), ("read_document", {}),
    ("read_document", {"document_id": "notes", "approved": True}),
    ("read_document", {"document_id": 4}), ("read_document", {"document_id": ""}),
    ("read_document", {"document_id": "x" * 65}), ("send_draft", []),
    ("create_draft", {"recipient": "a@real.com", "subject": "x", "body": "x"}),
    ("create_draft", {"recipient": "coordinator@club.test", "subject": " ", "body": "x"}),
    ("create_draft", {"recipient": "coordinator@club.test", "subject": "x", "body": "x" * 4001}),
])
def test_strict_arguments(task, tool, args):
    assert not valid_call(tool, args)
    state = State.from_task(task)
    assert evaluate(state, tool, args).kind == "deny"


def test_denied_read_has_no_effect(task):
    state = State.from_task(task)
    event, result, _ = execute(state, "read_document", {"document_id": "roster"})
    assert event["decision"] == "deny" and event["effect"] is None
    assert result == {"status": "denied"}
    assert state.label == "public" and not state.reads
    assert "PRIVATE" not in str(result)


def make_pending(task):
    state = State.from_task(task)
    execute(state, "create_draft", {"recipient": "coordinator@club.test", "subject": "x", "body": "y"})
    execute(state, "send_draft", {"draft_id": "draft-1"})
    assert state.pending
    return state


@pytest.mark.parametrize("change", ["body", "label", "policy", "run", "expiry", "sent"])
def test_stale_approval(task, change):
    state = make_pending(task)
    if change == "body":
        state.drafts["draft-1"]["body"] = "changed"
    elif change == "label":
        state.drafts["draft-1"]["label"] = "confidential"
    elif change == "policy":
        state.policy["recipients"] = []
    elif change == "run":
        state.run = "other-run"
    elif change == "expiry":
        state.tick += 20
    else:
        state.drafts["draft-1"]["sent"] = True
    assert resolve_approval(state, "draft-1", True)["status"] == "stale_approval"
    assert not state.outbox


def test_exact_approval_sends_once(task):
    state = make_pending(task)
    assert resolve_approval(state, "draft-1", True)["status"] == "sent"
    assert resolve_approval(state, "draft-1", True)["status"] == "stale_approval"
    assert len(state.outbox) == 1


def test_denied_approval_never_sends(task):
    state = make_pending(task)
    assert resolve_approval(state, "draft-1", False)["status"] == "rejected"
    assert not state.outbox


def test_labels_raise_existing_drafts(task):
    state = make_pending(task)
    state.documents["notes"]["label"] = "confidential"
    execute(state, "read_document", {"document_id": "notes"})
    assert state.label == state.drafts["draft-1"]["label"] == "confidential"
    assert resolve_approval(state, "draft-1", True)["status"] == "stale_approval"


def test_policy_evaluation_is_pure(task):
    state = make_pending(task)
    before = deepcopy(state.to_dict())
    evaluate(state, "send_draft", {"draft_id": "draft-1"})
    assert state.to_dict() == before


def test_corrupt_label_fails_closed(task):
    state = State.from_task(task)
    state.label = "unknown"
    assert evaluate(state, "read_document", {"document_id": "notes"}).kind == "deny"


def test_duplicate_send_does_not_refresh_approval(task):
    state = make_pending(task)
    pending = deepcopy(state.pending)
    event, _, _ = execute(state, "send_draft", {"draft_id": "draft-1"})
    assert event["decision"] == "deny"
    assert state.pending == pending
    assert not state.outbox
