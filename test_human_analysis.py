import csv
from copy import deepcopy

import pytest

from gatebench.audit import export_audit, RECOVERY_RATING_FIELDS, summarize_audit
from gatebench.experiment import run_experiment
from gatebench.human_analysis import human_analysis
from gatebench.io import read_json, write_json


def csv_file(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def review(root, cfg, task, tmp_path, request):
    run = tmp_path / "run"
    kwargs = {}
    if getattr(request, "param", False):
        from gatebench.providers import OfflineProvider, ProviderError

        class OnceError(OfflineProvider):
            count = 0

            def next_turn(self, history):
                self.count += 1
                if self.count == 3:
                    raise ProviderError("Synthetic first-branch failure")
                return super().next_turn(history)

        kwargs["provider_factory"] = lambda *args: OnceError(task)
    run_experiment(root, {**cfg, "repeats": 2, "bootstrap_samples": 10},
                   {"schema_version": 1, "tasks": [task]}, run, **kwargs)
    audit = export_audit(run, tmp_path / "audit")
    mapping = read_json(audit / "PRIVATE_mapping.json")
    for name in ("left", "right"):
        csv_file(tmp_path / f"{name}.csv", ["case_id", "rater", *RECOVERY_RATING_FIELDS, "notes"],
                 [{"case_id": r["case_id"], "rater": name, "notes": "Synthetic test rating",
                   **{f: "no" for f in RECOVERY_RATING_FIELDS}} for r in mapping])
    decisions = [{"case_id": r["case_id"], "adjudicator": "test adjudicator",
                  "semantic_completion": "no" if r["condition"] == "minimal" else "yes",
                  "checkpoint_completion": "no", "rationale": "Synthetic test only"}
                 for r in mapping]
    return run, audit, mapping, decisions


def analyze_review(review, tmp_path, decisions=None, output="report"):
    run, audit, _, defaults = review
    selected = defaults if decisions is None else decisions
    path = tmp_path / "decisions.csv"
    csv_file(path, ["case_id", "adjudicator", "semantic_completion", "checkpoint_completion", "rationale"],
             selected)
    return human_analysis(run, audit, tmp_path / "left.csv", tmp_path / "right.csv", path,
                          tmp_path / output)


def test_separate_human_estimates_and_unchanged_inputs(review, tmp_path):
    run, audit, _, _ = review
    before = (run / "branches.jsonl").read_bytes()
    report = read_json(analyze_review(review, tmp_path) / "human_sensitivity.json")
    assert (run / "branches.jsonl").read_bytes() == before
    assert all(r["difference"] == 1 for r in report["paired_completion"]
               if r["scoring"] == "human_adjudicated")
    assert all(r["difference"] == 0 for r in report["paired_completion"]
               if r["scoring"] == "automatic_full_run")
    assert report["schema_version"] == 2
    assert all(r["automatic"]["difference"] == 0 and r["human"]["difference"] == 1
               for r in report["common_eligibility"])
    ledger = read_json(tmp_path / "report/inclusion_ledger.json")
    assert len(ledger["pairs"]) == 8 and all(p["retained"] for p in ledger["pairs"])
    packet = read_json(audit / "rater_cases.json")
    assert all(c["checkpoint_outbox"] == [] for c in packet["cases"])
    assert all("requested_recipient" in c for c in packet["cases"])
    summarize_audit(audit, tmp_path / "left.csv", tmp_path / "right.csv", tmp_path / "summary")
    from gatebench.cli import main
    main(["human-analysis", str(run), str(audit), "--left", str(tmp_path / "left.csv"),
          "--right", str(tmp_path / "right.csv"), "--adjudication", str(tmp_path / "decisions.csv"),
          "--output", str(tmp_path / "cli-report")])
    assert (tmp_path / "cli-report/human_sensitivity.json").exists()
    with pytest.raises(FileExistsError):
        analyze_review(review, tmp_path)


def test_uncertainty_and_human_checkpoint_exclusions(review, tmp_path):
    decisions = deepcopy(review[3])
    for d in decisions:
        d["semantic_completion"] = "uncertain"
    report = read_json(analyze_review(review, tmp_path, decisions) / "human_sensitivity.json")
    assert len(report["excluded_uncertain"]) == len(decisions)
    assert all(r["retained_pairs"] == 0 for r in report["common_eligibility"])
    assert all(r["difference"] is None for r in report["paired_completion"]
               if r["scoring"] == "human_adjudicated")
    for d in decisions:
        d["semantic_completion"] = "yes"
        d["checkpoint_completion"] = "yes"
    report = read_json(analyze_review(review, tmp_path, decisions, "precompleted") / "human_sensitivity.json")
    assert all(r["precompleted_pairs_excluded"] == 2 for r in report["paired_completion"]
               if r["scoring"] == "human_adjudicated")
    ledger = read_json(tmp_path / "precompleted/inclusion_ledger.json")
    assert all("human_precompleted" in p["common_exclusion_reasons"] for p in ledger["pairs"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "reason", "checkpoint"])
def test_bad_adjudications_rejected(review, tmp_path, change):
    decisions = deepcopy(review[3])
    if change == "missing":
        decisions.pop()
    elif change == "duplicate":
        decisions.append(decisions[0])
    elif change == "reason":
        decisions[0]["rationale"] = ""
    else:
        decisions[0]["checkpoint_completion"] = "yes"
    with pytest.raises(ValueError):
        analyze_review(review, tmp_path, decisions)


def test_mismatched_packet_rejected(review, tmp_path):
    path = review[1] / "rater_cases.json"
    packet = read_json(path)
    packet["cases"][0]["outbox"] = []
    write_json(path, packet)
    with pytest.raises(ValueError, match="evidence differs"):
        analyze_review(review, tmp_path)


@pytest.mark.parametrize("review", [True], indirect=True)
def test_provider_errors_follow_primary_and_sensitivity_rules(review, tmp_path):
    report = read_json(analyze_review(review, tmp_path) / "human_sensitivity.json")
    human = [r for r in report["paired_completion"] if r["scoring"] == "human_adjudicated"]
    assert sum(r["error_pairs_excluded"] for r in human if r["complete_case"]) >= 1
    assert all(r["error_pairs_excluded"] == 0 for r in human if not r["complete_case"])


def test_disagreement_preserved_and_uncertain_pair_reported(review, tmp_path):
    decisions = deepcopy(review[3])
    decisions[0]["semantic_completion"] = "uncertain"
    mapping = review[2]
    csv_file(tmp_path / "right.csv", ["case_id", "rater", *RECOVERY_RATING_FIELDS, "notes"],
             [{"case_id": r["case_id"], "rater": "right", "notes": "Synthetic disagreement",
               **{f: "no" for f in RECOVERY_RATING_FIELDS}, "semantic_completion": "yes"}
              for r in mapping])
    report = read_json(analyze_review(review, tmp_path, decisions) / "human_sensitivity.json")
    assert len(report["reviewer_disagreement_or_uncertainty"]) == len(mapping)
    assert len(report["excluded_uncertain"]) == 1
    assert sum(r["missing_pairs"] for r in report["paired_completion"]
               if r["scoring"] == "human_adjudicated") >= 1
