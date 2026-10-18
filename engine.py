"""Check permissions before running simulated tools."""

from copy import deepcopy

from .policy import LABELS, evaluate
from .state import draft_hash


def execute(state, name, args):
    state.tick += 1
    before = state.to_dict()
    decision = evaluate(state, name, args)
    event = {"tool": name, "args": deepcopy(args), "decision": decision.kind,
             "rule": decision.rule, "tick": state.tick, "label_before": state.label,
             "effect": None, "approval": None}
    if decision.kind == "deny":
        result = {"status": "denied"}
    elif decision.kind == "ask":
        draft_id = args["draft_id"]
        draft = state.drafts[draft_id]
        state.pending[draft_id] = {
            "hash": draft_hash(draft), "policy": deepcopy(state.policy),
            "run": state.run, "expires": state.tick + 10,
        }
        result = {"status": "awaiting_approval", "draft_id": draft_id}
    else:
        result = _effect(state, name, args)
        event["effect"] = deepcopy(result)
    event["label_after"] = state.label
    state.events.append(event)
    return event, result, before


def _effect(state, name, args):
    if name == "read_document":
        key = args["document_id"]
        doc = state.documents[key]
        state.reads.append(key)
        state.label = max((state.label, doc["label"]), key=LABELS.__getitem__)
        for draft in state.drafts.values():
            if not draft["sent"]:
                draft["label"] = max((draft["label"], state.label), key=LABELS.__getitem__)
        return {"status": "ok", "document_id": key, "content": doc["body"]}
    if name == "create_draft":
        key = f"draft-{state.next_draft}"
        state.next_draft += 1
        state.drafts[key] = {**deepcopy(args), "label": state.label, "sent": False}
        return {"status": "ok", "draft_id": key}
    key = args["draft_id"]
    state.drafts[key]["sent"] = True
    state.outbox.append({"draft_id": key, **deepcopy(state.drafts[key])})
    return {"status": "sent", "draft_id": key}


def resolve_approval(state, draft_id, approve):
    state.tick += 1
    pending = state.pending.pop(draft_id, None)
    draft = state.drafts.get(draft_id)
    if pending is None or draft is None:
        return {"status": "stale_approval"}
    stale = (draft["sent"] or pending["hash"] != draft_hash(draft) or
             pending["policy"] != state.policy or pending["run"] != state.run or
             state.tick > pending["expires"])
    if stale:
        return {"status": "stale_approval"}
    if not approve:
        return {"status": "rejected"}
    decision = evaluate(state, "send_draft", {"draft_id": draft_id}, approved=True)
    if decision.kind != "allow":
        return {"status": "denied"}
    return _effect(state, "send_draft", {"draft_id": draft_id})
