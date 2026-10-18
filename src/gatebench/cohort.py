"""Compare final scores on the same human-reviewed recovery opportunities."""

from collections import Counter, defaultdict
from copy import deepcopy

from .analysis import paired
from .io import digest


def common_cohort(mapping, decisions, model, condition, *, complete_case, samples, seed):
    if condition == "minimal":
        raise ValueError("Common-cohort contrasts need a non-minimal condition")
    groups = defaultdict(dict)
    for row in mapping:
        if row["model_id"] == model and row["condition"] in {"minimal", condition}:
            key = (row["task_id"], row["repeat"])
            if row["condition"] in groups[key]:
                raise ValueError("Duplicate branch in common-cohort input")
            groups[key][row["condition"]] = row
    ledger, human, automatic, retained = [], [], [], []
    for (task, repeat), pair in sorted(groups.items()):
        missing = set(pair) != {"minimal", condition}
        if not missing and len({r["checkpoint_hash"] for r in pair.values()}) != 1:
            raise ValueError("Paired branches do not share a checkpoint")
        if not missing and len({r["cluster"] for r in pair.values()}) != 1:
            raise ValueError("Paired branches have different task clusters")
        ratings = {arm: decisions[row["case_id"]] for arm, row in pair.items()}
        for arm, row in pair.items():
            value = row["metrics"]["completion"]
            if type(value) is not int or value not in (0, 1):
                raise ValueError("Automatic completion must be a binary score")
            if any(ratings[arm][f] not in {"yes", "no", "uncertain"}
                   for f in ("semantic_completion", "checkpoint_completion")):
                raise ValueError("Invalid human completion rating")
        if len({r["checkpoint_completion"] for r in ratings.values()}) > 1:
            raise ValueError("Paired branches have inconsistent human checkpoint labels")
        technical_error = any(r["termination"] in {"provider_error", "api_budget"}
                              for r in pair.values())
        common_reasons = []
        if missing:
            common_reasons.append("missing_branch")
        if any(r["checkpoint_completion"] == "uncertain" for r in ratings.values()):
            common_reasons.append("uncertain_checkpoint")
        if any(r["checkpoint_completion"] == "yes" for r in ratings.values()):
            common_reasons.append("human_precompleted")
        if any(r["semantic_completion"] == "uncertain" for r in ratings.values()):
            common_reasons.append("uncertain_final_score")
        if complete_case and technical_error:
            common_reasons.append("provider_error")
        automatic_reasons = ["missing_branch"] if missing else []
        if any(r["metrics"].get("completed_before_checkpoint", 0) for r in pair.values()):
            automatic_reasons.append("automatic_precompleted")
        if complete_case and technical_error:
            automatic_reasons.append("provider_error")
        rated_reasons = list(automatic_reasons)
        for reason in ("uncertain_checkpoint", "uncertain_final_score"):
            if reason in common_reasons:
                rated_reasons.append(reason)
        record = {"model_id": model, "condition": condition, "task_id": task, "repeat": repeat,
                  "complete_case": complete_case, "retained": not common_reasons,
                  "common_exclusion_reasons": common_reasons,
                  "automatic_full_exclusion_reasons": automatic_reasons,
                  "automatic_rated_rows_exclusion_reasons": rated_reasons,
                  "branches": {arm: {"case_id": row["case_id"], "checkpoint_hash": row["checkpoint_hash"],
                                      "termination": row["termination"],
                                      "automatic_completion": row["metrics"]["completion"],
                                      "automatic_checkpoint_completion": row["metrics"].get("completed_before_checkpoint"),
                                      "human_completion": ratings[arm]["semantic_completion"],
                                      "human_checkpoint_completion": ratings[arm]["checkpoint_completion"]}
                               for arm, row in pair.items()}}
        ledger.append(record)
        if common_reasons:
            continue
        retained.append({"task_id": task, "repeat": repeat,
                         "baseline_case": pair["minimal"]["case_id"],
                         "condition_case": pair[condition]["case_id"]})
        for arm in ("minimal", condition):
            auto = deepcopy(pair[arm])
            # Only the added comparison uses human-reviewed checkpoint eligibility.
            auto["metrics"]["completed_before_checkpoint"] = 0
            rated = deepcopy(auto)
            rated["metrics"]["completion"] = int(ratings[arm]["semantic_completion"] == "yes")
            automatic.append(auto)
            human.append(rated)
    auto_result = paired(automatic, model, condition, "completion", complete_case=complete_case,
                         samples=samples, seed=seed)
    human_result = paired(human, model, condition, "completion", complete_case=complete_case,
                          samples=samples, seed=seed)
    counts = Counter(reason for row in ledger for reason in row["common_exclusion_reasons"])
    report = {"model_id": model, "condition": condition, "complete_case": complete_case,
              "candidate_pairs": len(ledger), "retained_pairs": len(retained),
              "excluded_pairs": len(ledger) - len(retained), "exclusion_reason_counts": dict(counts),
              "retained_pair_sha256": digest(retained), "automatic": auto_result, "human": human_result,
              "human_minus_automatic_difference": (
                  human_result["difference"] - auto_result["difference"] if retained else None),
              "scope": "Both scores use the same pairs and human checkpoint eligibility. "
                       "Differences describe final-score disagreement, not a new treatment effect. "
                       "Reason counts overlap; exclusions are recorded before estimation."}
    return report, ledger
