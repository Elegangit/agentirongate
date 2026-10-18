from copy import deepcopy
from dataclasses import dataclass
import json

from .io import canonical, digest
from .state import State


@dataclass(frozen=True)
class Checkpoint:
    payload: str
    sha256: str

    @classmethod
    def capture(cls, state, history, call, rule):
        if state.pending:
            raise ValueError("Checkpoint has unresolved approval")
        data = {"schema_version": 1, "state": state.to_dict(), "history": deepcopy(history),
                "call": deepcopy(call), "rule": rule}
        checkpoint = cls(canonical(data), digest(data))
        restored, saved_history, saved_call, saved_rule = checkpoint.restore()
        if (restored != state or saved_history != history or
                saved_call != call or saved_rule != rule):
            raise ValueError("Checkpoint differs from original state or conversation")
        return checkpoint

    def restore(self):
        data = json.loads(self.payload)
        if data.get("schema_version") != 1 or digest(data) != self.sha256:
            raise ValueError("Checkpoint integrity check failed")
        state = State.restore(data["state"])
        if state.pending:
            raise ValueError("Checkpoint has pending approval")
        if canonical(state.to_dict()) != canonical(data["state"]):
            raise ValueError("Checkpoint state restoration differs")
        return state, data["history"], data["call"], data["rule"]

    def to_dict(self):
        return {"payload": json.loads(self.payload), "sha256": self.sha256}
