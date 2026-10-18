from copy import deepcopy
from dataclasses import asdict, dataclass, field
from uuid import uuid4

from .io import digest


@dataclass
class State:
    documents: dict
    contacts: dict
    policy: dict
    label: str = "public"
    drafts: dict = field(default_factory=dict)
    outbox: list = field(default_factory=list)
    reads: list = field(default_factory=list)
    pending: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    tick: int = 0
    next_draft: int = 1
    run: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_task(cls, task):
        return cls(**deepcopy({k: task[k] for k in ("documents", "contacts", "policy")}))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def restore(cls, data):
        return cls(**deepcopy(data))


def draft_hash(draft):
    return digest({k: draft[k] for k in ("recipient", "subject", "body", "label")})
