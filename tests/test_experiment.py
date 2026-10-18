from copy import deepcopy
import json
import shutil

import pytest

from gatebench.analysis import analyze, paired
from gatebench.branding import latex_text, render_identity
from gatebench.config import identity
from gatebench.experiment import run_experiment
from gatebench.io import read_json
from gatebench.tasks import load_tasks, validate_task


def test_end_to_end(root, cfg, tmp_path):
    dataset = load_tasks(root / "data/pilot.json")
    cfg = {**cfg, "repeats": 1, "bootstrap_samples": 30}
    output = tmp_path / "run"
    manifest = run_experiment(root, cfg, dataset, output)
    assert manifest["status"] == "complete"
    assert manifest["branches"] == 36 and manifest["api_requests"] == 0
    assert manifest["evidence"] == "scripted_fixture_only"
    analyze(output)
    report = read_json(output / "analysis/summary.json")
    assert all(r["mean"] == 1 for r in report["summary"] if r["metric"] == "completion")
    assert all(r["difference"] == 0 for r in report["paired"])
    assert all(r["clusters"] == 4 for r in report["paired"])
    with pytest.raises(FileExistsError):
        run_experiment(root, cfg, dataset, output)


def test_benign_no_manufactured_denial(root, cfg, tmp_path):
    output = tmp_path / "benign"
    manifest = run_experiment(root, cfg, load_tasks(root / "data/benign.json"), output)
    assert manifest["branches"] == 0 and manifest["prefixes"] == 4
    rows = [json.loads(s) for s in (output / "prefixes.jsonl").read_text().splitlines()]
    assert all(r["metrics"]["completion"] == 1 for r in rows)
    analyze(output)
    report = read_json(output / "analysis/summary.json")
    assert report["branch_terminations"]
    assert all(r["counts"] == {} and r["continuations"] == 0
               for r in report["branch_terminations"])
    assert (output / "analysis/terminations.csv").read_text().strip() == (
        "model,condition,termination,count")


def test_incomplete_run_analysis_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "interrupted"}))
    with pytest.raises(ValueError, match="incomplete"):
        analyze(tmp_path)


def test_trace_tampering_is_detected(root, cfg, task, tmp_path):
    output = tmp_path / "tampered"
    run_experiment(root, {**cfg, "repeats": 1}, {"schema_version": 1, "tasks": [task]}, output)
    row = json.loads((output / "branches.jsonl").read_text().splitlines()[0])
    path = output / row["trace"]
    trace = read_json(path)
    trace["metrics"]["completion"] = 42
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="integrity"):
        analyze(output)


def test_api_budget_marks_run_incomplete(root, cfg, task, tmp_path):
    from gatebench.providers import BudgetExceeded

    class Exhausted:
        def next_turn(self, history):
            raise BudgetExceeded("test request limit")

    output = tmp_path / "exhausted"
    manifest = run_experiment(root, cfg, {"schema_version": 1, "tasks": [task]}, output,
                              provider_factory=lambda *args: Exhausted())
    assert manifest["status"] == "budget_exhausted"
    with pytest.raises(ValueError, match="incomplete"):
        analyze(output)


def test_initialization_failure_preserves_manifest(root, cfg, task, tmp_path):
    def broken(*args):
        raise RuntimeError("fixture initialization failure")

    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError):
        run_experiment(root, cfg, {"schema_version": 1, "tasks": [task]}, output,
                       provider_factory=broken)
    assert read_json(output / "manifest.json")["status"] == "interrupted"


@pytest.mark.parametrize("change", ["duplicate", "condition", "termination", "dataset"])
def test_analysis_rejects_mismatched_evidence(root, cfg, task, tmp_path, change):
    output = tmp_path / "mismatch"
    run_experiment(root, {**cfg, "repeats": 1}, {"schema_version": 1, "tasks": [task]}, output)
    path = output / "branches.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif change == "condition":
        rows[0]["condition"] = "invented-arm"
    elif change == "termination":
        rows[0]["termination"] = "invented-status"
    else:
        (output / "dataset.json").write_text("{}")
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError):
        analyze(output)


def test_cluster_pairing_and_errors():
    rows = []
    for task, cluster in [("a", "x"), ("b", "x"), ("c", "y")]:
        for repeat in range(3):
            for condition in ("minimal", "rule"):
                rows.append({"task_id": task, "cluster": cluster, "model_id": "m",
                             "repeat": repeat, "condition": condition, "termination": "finished",
                             "metrics": {"completion": int(condition == "rule")}})
    result = paired(rows, "m", "rule", "completion", samples=30)
    assert result["difference"] == result["low"] == result["high"] == 1
    assert result["tasks"] == 3 and result["clusters"] == 2
    rows[0]["termination"] = "provider_error"
    assert paired(rows, "m", "rule", "completion", complete_case=True)["error_pairs_excluded"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        paired(rows + [rows[0]], "m", "rule", "completion")


def test_rename_propagation(root, tmp_path):
    shutil.copy(root / "project.toml", tmp_path)
    shutil.copy(root / "README.md", tmp_path)
    original_body = (tmp_path / "README.md").read_text().split("<!-- project-identity:end -->")[1]
    path = tmp_path / "project.toml"
    old_name = identity(root)["name"]
    path.write_text(path.read_text().replace(old_name, "New Research Name"))
    render_identity(tmp_path, latex=tmp_path / "project-name.tex")
    macros = (tmp_path / "project-name.tex").read_text()
    assert r"\newcommand{\systemname}{New Research Name}" in macros
    assert identity(tmp_path)["paper_title"] in macros
    assert identity(tmp_path)["name"] == "New Research Name"
    assert "New Research Name" in (tmp_path / "README.md").read_text()
    assert not (tmp_path / "CITATION.cff").exists()
    assert original_body == (tmp_path / "README.md").read_text().split("<!-- project-identity:end -->")[1]
    assert old_name not in (tmp_path / "README.md").read_text()


def test_latex_name_escaping():
    assert latex_text("Gate & Co_2%") == r"Gate \& Co\_2\%"
    assert latex_text(r"\input{x}") == r"\textbackslash{}input\{x\}"


@pytest.mark.parametrize("kind", ["label", "goal", "recipient", "field"])
def test_task_validation(task, kind):
    task = deepcopy(task)
    if kind == "label":
        task["documents"]["notes"]["label"] = "secret-ish"
    elif kind == "goal":
        task["goal"]["required_facts"] = []
    elif kind == "recipient":
        task["contacts"]["person@example.com"] = "external"
    else:
        task["surprise"] = True
    with pytest.raises(ValueError):
        validate_task(task)
