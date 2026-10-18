from copy import deepcopy

import pytest

from gatebench.analysis import paired
from gatebench.cohort import common_cohort


def add_pair(rows, ratings, task, repeat=0, *, auto=(0, 1), human=("no", "yes"),
             auto_before=0, human_before="no", termination="finished"):
    for n, arm in enumerate(("minimal", "directed")):
        case = f"{task}-{repeat}-{arm}"
        rows.append({"case_id": case, "model_id": "m", "task_id": task, "cluster": task,
                     "repeat": repeat, "condition": arm, "checkpoint_hash": "cp-" + task,
                     "termination": termination if n else "finished",
                     "metrics": {"completion": auto[n], "completed_before_checkpoint": auto_before}})
        ratings[case] = {"semantic_completion": human[n], "checkpoint_completion": human_before}


def compare(rows, ratings, complete_case=True):
    return common_cohort(rows, ratings, "m", "directed", complete_case=complete_case, samples=20, seed=17)


def test_only_final_score_disagreement():
    rows, ratings = [], {}
    add_pair(rows, ratings, "a", auto=(1, 1))
    original = deepcopy(rows)
    report, ledger = compare(rows, ratings)
    assert report["automatic"]["difference"] == 0
    assert report["human"]["difference"] == 1
    assert report["human_minus_automatic_difference"] == 1
    assert report["retained_pairs"] == 1 and ledger[0]["retained"]
    assert rows == original


def test_only_checkpoint_judgments_change_the_population():
    rows, ratings = [], {}
    add_pair(rows, ratings, "a", auto=(1, 1), human=("yes", "yes"), human_before="yes")
    add_pair(rows, ratings, "b")
    original_auto = paired(rows, "m", "directed", "completion")
    report, ledger = compare(rows, ratings)
    assert original_auto["difference"] == .5
    assert report["automatic"]["difference"] == report["human"]["difference"] == 1
    assert report["human_minus_automatic_difference"] == 0
    assert ledger[0]["common_exclusion_reasons"] == ["human_precompleted"]
    assert ledger[0]["automatic_full_exclusion_reasons"] == []
    assert report["retained_pairs"] == 1


def test_both_checkpoint_and_outcome_differences():
    rows, ratings = [], {}
    add_pair(rows, ratings, "a", auto=(1, 1), human=("yes", "yes"), human_before="yes")
    add_pair(rows, ratings, "b", auto=(1, 1), auto_before=1)
    report, ledger = compare(rows, ratings)
    assert report["automatic"]["difference"] == 0 and report["human"]["difference"] == 1
    assert ledger[1]["automatic_full_exclusion_reasons"] == ["automatic_precompleted"]
    assert ledger[1]["retained"]


def test_equal_task_weights_with_unequal_repeat_counts():
    rows, ratings = [], {}
    for repeat in range(3):
        add_pair(rows, ratings, "a", repeat)
    add_pair(rows, ratings, "b", auto=(1, 1), human=("yes", "yes"))
    report, ledger = compare(rows, ratings)
    assert report["human"]["difference"] == report["automatic"]["difference"] == .5
    assert report["human"]["tasks"] == report["automatic"]["tasks"] == 2
    assert report["retained_pairs"] == len(ledger) == 4


def test_uncertainty_errors_and_empty_sets():
    rows, ratings = [], {}
    add_pair(rows, ratings, "a", human=("no", "uncertain"))
    add_pair(rows, ratings, "b", human_before="uncertain")
    add_pair(rows, ratings, "c", termination="provider_error")
    report, ledger = compare(rows, ratings)
    assert report["retained_pairs"] == 0
    assert report["automatic"]["difference"] is report["human"]["difference"] is None
    assert report["human_minus_automatic_difference"] is None
    assert report["exclusion_reason_counts"] == {"uncertain_final_score": 1,
                                                  "uncertain_checkpoint": 1, "provider_error": 1}
    sensitivity, sensitivity_ledger = compare(rows, ratings, complete_case=False)
    assert sensitivity["retained_pairs"] == 1 and sensitivity_ledger[-1]["retained"]
    assert sensitivity["human"]["difference"] == 1


def test_incomplete_duplicate_and_mismatched_pairs():
    rows, ratings = [], {}
    add_pair(rows, ratings, "a")
    report, ledger = compare(rows[:1], ratings)
    assert report["retained_pairs"] == 0 and ledger[0]["common_exclusion_reasons"] == ["missing_branch"]
    with pytest.raises(ValueError, match="Duplicate"):
        compare(rows + [rows[0]], ratings)
    rows[1]["checkpoint_hash"] = "different"
    with pytest.raises(ValueError, match="checkpoint"):
        compare(rows, ratings)


def test_other_models_and_arms_do_not_change_cohort():
    rows, ratings = [], {}
    add_pair(rows, ratings, "a")
    expected, _ = compare(rows, ratings)
    rows.extend([{**rows[0], "model_id": "other"}, {**rows[1], "condition": "neutral"}])
    report, _ = compare(rows, ratings)
    assert report == expected
