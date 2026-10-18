"""Check permissions without running tools."""

from dataclasses import dataclass
import re

LABELS = {"public": 0, "internal": 1, "confidential": 2}
FIELDS = {"read_document": {"document_id"},
          "create_draft": {"recipient", "subject", "body"}, "send_draft": {"draft_id"}}
SECURITY_RULES = {"document_not_permitted", "recipient_not_permitted", "confidential_external"}
REASONS = {
    "document_not_permitted": "This document is not on the read allowlist.",
    "recipient_not_permitted": "This recipient is not on the recipient allowlist.",
    "confidential_external": "Confidential information cannot be sent outside the organization.",
}


@dataclass(frozen=True)
class Decision:
    kind: str
    rule: str


def valid_call(name, args):
    if name not in FIELDS or not isinstance(args, dict) or set(args) != FIELDS[name]:
        return False
    if any(not isinstance(v, str) for v in args.values()):
        return False
    if name == "create_draft":
        return (bool(re.fullmatch(r"[a-z0-9._+-]+@[a-z0-9.-]+\.test", args["recipient"]))
                and 0 < len(args["subject"].strip()) <= 120 and len(args["body"]) <= 4000)
    return all(0 < len(v) <= 64 for v in args.values())


def evaluate(state, name, args, approved=False):
    if not valid_call(name, args):
        return Decision("deny", "invalid_arguments")
    if state.label not in LABELS:
        return Decision("deny", "invalid_label")
    if name == "read_document":
        doc = state.documents.get(args["document_id"])
        if doc is None:
            return Decision("deny", "unknown_document")
        if doc["label"] not in LABELS:
            return Decision("deny", "invalid_label")
        if args["document_id"] not in state.policy["reads"]:
            return Decision("deny", "document_not_permitted")
        return Decision("allow", "document_permitted")
    if name == "create_draft":
        recipient = args["recipient"]
        if recipient not in state.policy["recipients"] or recipient not in state.contacts:
            return Decision("deny", "recipient_not_permitted")
        if state.label not in LABELS or state.contacts[recipient] not in {"internal", "external"}:
            return Decision("deny", "invalid_label")
        return Decision("allow", "draft_permitted")
    draft = state.drafts.get(args["draft_id"])
    if draft is None:
        return Decision("deny", "unknown_draft")
    if draft["sent"]:
        return Decision("deny", "already_sent")
    if args["draft_id"] in state.pending and not approved:
        return Decision("deny", "already_pending")
    recipient, label = draft["recipient"], draft["label"]
    if recipient not in state.policy["recipients"] or recipient not in state.contacts:
        return Decision("deny", "recipient_not_permitted")
    kind = state.contacts[recipient]
    if label not in LABELS or kind not in {"internal", "external"}:
        return Decision("deny", "invalid_label")
    if label == "confidential" and kind == "external":
        return Decision("deny", "confidential_external")
    if (not approved and (state.policy["approve_every_send"] or label == "confidential" or
                         (label == "internal" and kind == "external"))):
        return Decision("ask", "review_required")
    return Decision("allow", "send_permitted")
