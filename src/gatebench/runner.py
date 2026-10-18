"""Run to the first eligible denial, then branch from the saved checkpoint."""

from copy import deepcopy
from dataclasses import dataclass

from .approval import approve_draft
from .checkpoint import Checkpoint
from .engine import execute, resolve_approval
from .feedback import render
from .policy import SECURITY_RULES, valid_call
from .providers import BudgetExceeded, ProviderError, add_result
from .scoring import prohibited, score
from .state import State


@dataclass
class Prefix:
    checkpoint: Checkpoint | None
    termination: str
    history: list
    state: State
    calls: int
    turns: list


def act(task, state, call, approver):
    event, result, before = execute(state, call["name"], call["args"])
    event["oracle_prohibited"] = bool(valid_call(call["name"], call["args"]) and
                                      prohibited(task, before, call["name"], call["args"]))
    if result["status"] == "awaiting_approval":
        draft_id = result["draft_id"]
        approve = approver == "ideal" and approve_draft(task, state.drafts[draft_id])
        result = resolve_approval(state, draft_id, approve)
        event["approval"] = {"approver": approver, "approved": approve, "result": deepcopy(result)}
        if result["status"] == "sent":
            event["effect"] = deepcopy(result)
    return event, result


def turn_record(turn):
    return {"usage": deepcopy(turn.usage), "latency_seconds": turn.latency,
            "model_version": turn.model_version, "termination": turn.termination,
            "finish_reason": turn.finish_reason}


def collect_prefix(task, provider, cfg):
    state = State.from_task(task)
    history = [{"role": "user", "parts": [{"text": task["prompt"]}]}]
    turns = []
    termination = "prefix_budget"
    calls = 0
    for _ in range(cfg["prefix_calls"]):
        calls += 1
        try:
            turn = provider.next_turn(deepcopy(history))
        except BudgetExceeded:
            calls -= 1
            termination = "api_budget"
            break
        except ProviderError:
            termination = "provider_error"
            break
        turns.append(turn_record(turn))
        history.append(turn.content)
        if turn.termination:
            termination = turn.termination
            break
        if len(turn.calls) != 1:
            termination = "excluded_batch"
            break
        call = turn.calls[0]
        if not valid_call(call["name"], call["args"]):
            termination = "invalid_call"
            break
        event, result = act(task, state, call, cfg["approver"])
        if event["decision"] == "deny" and event["rule"] in SECURITY_RULES:
            cp = Checkpoint.capture(state, history, call, event["rule"])
            return Prefix(cp, "eligible", history, state, calls, turns)
        if event["decision"] == "deny":
            result = {"status": "denied", "message": "Action denied."}
        add_result(history, call, result)
    return Prefix(None, termination, history, state, calls, turns)


def restore_fidelity(checkpoint, suffix, *, original=None, task=None, approver="ideal"):
    """Compare original/restored states and scripted effects; never call a model."""
    left = deepcopy(original) if original is not None else checkpoint.restore()[0]
    right, *_ = checkpoint.restore()
    if left != right:
        raise ValueError("Restore mismatch")
    for call in suffix:
        if task is None:
            left_result = execute(left, call["name"], call["args"])
            right_result = execute(right, call["name"], call["args"])
        else:
            left_result = act(task, left, call, approver)
            right_result = act(task, right, call, approver)
        if left_result != right_result or left != right:
            raise ValueError("Scripted replay differs after restoration")
    return True


def fidelity_suffix(task, state):
    """Exercise allowed reads, a draft, and its send on disposable copies."""
    return ([{"name": "read_document", "args": {"document_id": key}}
             for key in task["policy"]["reads"]] + [
        {"name": "create_draft", "args": {"recipient": task["goal"]["recipient"],
         "subject": "Fidelity probe", "body": "; ".join(task["goal"]["required_facts"])}},
        {"name": "send_draft", "args": {"draft_id": f"draft-{state.next_draft}"}},
    ])


def continue_branch(task, checkpoint, condition, provider, cfg):
    state, history, denied_call, rule = checkpoint.restore()
    completed_before = score(task, state, [])['completion']
    add_result(history, denied_call, render(condition, rule, task["recovery_hint"]))
    start_event = len(state.events)
    turns, calls, proposals = [], 0, 0
    termination = "call_budget"
    for _ in range(cfg["branch_calls"]):
        if proposals >= cfg["branch_proposals"]:
            termination = "proposal_budget"
            break
        calls += 1
        try:
            turn = provider.next_turn(deepcopy(history))
        except BudgetExceeded:
            calls -= 1
            termination = "api_budget"
            break
        except ProviderError:
            termination = "provider_error"
            break
        turns.append(turn_record(turn))
        history.append(turn.content)
        if turn.termination:
            termination = turn.termination
            break
        if len(turn.calls) != 1:
            termination = "batch_stop"
            break
        call = turn.calls[0]
        proposals += 1
        if not valid_call(call["name"], call["args"]):
            termination = "invalid_call"
            break
        event, result = act(task, state, call, cfg["approver"])
        if event["decision"] == "deny":
            style = condition if (cfg["feedback_scope"] == "persistent" and
                                  event["rule"] in SECURITY_RULES) else "minimal"
            result = render(style, event["rule"], task["recovery_hint"])
        add_result(history, call, result)
    events = state.events[start_event:]
    metrics = score(task, state, events)
    metrics.update({"model_calls": calls, "proposals": proposals,
                    "completed_before_checkpoint": completed_before,
                    "new_completion": int(not completed_before and metrics["completion"]),
                    "provider_error": int(termination == "provider_error"),
                    "latency_seconds": sum(t["latency_seconds"] for t in turns)})
    return {"condition": condition, "termination": termination, "metrics": metrics,
            "state": state.to_dict(), "history": history, "turns": turns,
            "checkpoint_hash": checkpoint.sha256, "post_checkpoint_event_start": start_event}
