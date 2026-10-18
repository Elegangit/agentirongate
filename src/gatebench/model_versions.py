"""Summarize provider-reported versions without inferring missing identifiers."""

from collections import Counter


class ModelVersionAudit:
    def __init__(self, models):
        self.models = models
        self.versions = {m["id"]: Counter() for m in models}
        self.missing = Counter()

    def observe(self, model_id, turns):
        for turn in turns:
            value = turn.get("model_version")
            if isinstance(value, str) and value.strip():
                self.versions[model_id][value.strip()] += 1
            else:
                self.missing[model_id] += 1

    def report(self):
        records = []
        for model in self.models:
            versions = dict(sorted(self.versions[model["id"]].items()))
            missing = self.missing[model["id"]]
            total = sum(versions.values()) + missing
            flags = []
            if len(versions) > 1:
                flags.append("mixed_model_versions")
            if missing:
                flags.append("missing_model_version")
            if total == 0:
                flags.append("no_returned_turns")
            records.append({"model_id": model["id"], "configured_model": model["model"],
                            "reported_versions": versions, "missing_identifiers": missing,
                            "returned_turns": total, "flags": flags})
        return {"schema_version": 1, "models": records,
                "flagged": any(r["flags"] for r in records),
                "scope": "Returned turns only; transport failures have no version evidence. "
                         "Identifiers are provider claims, not proof of a fixed backend. "
                         "Flags do not silently exclude outcomes or alter run status."}
