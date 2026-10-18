from copy import deepcopy
import shutil

import pytest

from gatebench.experiment import run_experiment
from gatebench.io import digest
from gatebench.study import freeze_plan, preflight


@pytest.fixture
def workspace(root, tmp_path):
    target = tmp_path / "repo"
    shutil.copytree(root / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "environment.yml", "project.toml", "README.md"):
        shutil.copy(root / name, target / name)
    shutil.copytree(root / "data", target / "data")
    return target


def plan_for(workspace, cfg, task, tmp_path, mode="confirmatory"):
    dataset = {"schema_version": 1, "tasks": [task]}
    plan = freeze_plan(cfg, dataset, tmp_path / "plan.json", root=workspace,
                       study_id="binding-test", mode=mode, catalogs=["data/recovery_checks.json"])
    return dataset, plan


@pytest.mark.parametrize("name", ["scoring.py", "feedback.py", "runner.py", "analysis.py"])
def test_research_edits_rejected_before_running(workspace, cfg, task, tmp_path, name):
    dataset, plan = plan_for(workspace, cfg, task, tmp_path)
    path = workspace / "src/gatebench" / name
    path.write_text(path.read_text() + "\n# Test-only change\n")
    result = preflight(cfg, dataset, plan=plan, root=workspace)
    assert not result["ready"]
    assert any(d["item"].endswith(name) for d in result["plan_deviations"])
    with pytest.raises(ValueError, match="research_code"):
        run_experiment(workspace, cfg, dataset, tmp_path / "run", study_plan=plan)
    assert not (tmp_path / "run").exists()


def test_unrelated_docs_and_viewer_do_not_invalidate(workspace, cfg, task, tmp_path):
    dataset, plan = plan_for(workspace, cfg, task, tmp_path)
    (workspace / "README.md").write_text("Updated documentation\n")
    (workspace / "src/gatebench/viewer/app.py").write_text("# Changed viewer\n")
    assert preflight(cfg, dataset, plan=plan, root=workspace)["ready"]


@pytest.mark.parametrize("target", ["data/recovery_checks.json", "environment.yml", "pyproject.toml"])
def test_catalog_and_dependency_specs_bound(workspace, cfg, task, tmp_path, target):
    dataset, plan = plan_for(workspace, cfg, task, tmp_path)
    path = workspace / target
    path.write_text(path.read_text() + "\n")
    assert not preflight(cfg, dataset, plan=plan, root=workspace)["ready"]


def test_runtime_dependency_change(workspace, cfg, task, tmp_path, monkeypatch):
    dataset, plan = plan_for(workspace, cfg, task, tmp_path)
    env = deepcopy(plan["binding"]["environment"])
    env["runtime_packages"]["httpx"] = "different-version"
    monkeypatch.setattr("gatebench.study.dependency_environment", lambda: env)
    assert not preflight(cfg, dataset, plan=plan, root=workspace)["ready"]
    env["runtime_packages"] = plan["binding"]["environment"]["runtime_packages"]
    env["installed_packages"]["unrelated-dev-tool"] = "new"
    assert preflight(cfg, dataset, plan=plan, root=workspace)["ready"]


def test_added_and_removed_research_modules(workspace, cfg, task, tmp_path):
    dataset, plan = plan_for(workspace, cfg, task, tmp_path)
    module = workspace / "src/gatebench/new_measurement.py"
    module.write_text("# Test fixture\n")
    assert not preflight(cfg, dataset, plan=plan, root=workspace)["ready"]
    module.unlink()
    (workspace / "src/gatebench/feedback.py").unlink()
    assert not preflight(cfg, dataset, plan=plan, root=workspace)["ready"]


def test_override_requires_exploratory_plan_and_reason(workspace, cfg, task, tmp_path):
    dataset, plan = plan_for(workspace, cfg, task, tmp_path, mode="exploratory")
    changed = {**cfg, "repeats": 1}
    assert not preflight(changed, dataset, plan=plan, root=workspace, mode="exploratory")["ready"]
    manifest = run_experiment(workspace, changed, dataset, tmp_path / "run", study_plan=plan,
                              override_reason="Shorter diagnostic run")
    assert manifest["study_plan"] == plan
    assert manifest["config"]["repeats"] == 1
    assert manifest["study_verification"]["override_reason"] == "Shorter diagnostic run"
    assert manifest["study_verification"]["plan_deviations"]
    assert not preflight(changed, dataset, plan=plan, root=workspace, mode="exploratory",
                         override_reason=" ")["ready"]
    with pytest.raises(ValueError, match="requires"):
        run_experiment(workspace, cfg, dataset, tmp_path / "bad", override_reason="No plan")


def test_confirmatory_override_and_legacy_plan_rejected(workspace, cfg, task, tmp_path):
    dataset, plan = plan_for(workspace, cfg, task, tmp_path)
    assert not preflight(cfg, dataset, plan=plan, root=workspace, override_reason="Excuse")["ready"]
    legacy = {k: v for k, v in plan.items() if k not in {"binding", "plan_sha256"}}
    legacy["schema_version"] = 1
    legacy["plan_sha256"] = digest(legacy)
    with pytest.raises(ValueError, match="re-freeze"):
        preflight(cfg, dataset, plan=legacy, root=workspace)


def test_catalog_cannot_escape_repository(workspace, cfg, task, tmp_path):
    with pytest.raises(ValueError):
        freeze_plan(cfg, {"tasks": [task]}, tmp_path / "plan.json", root=workspace,
                    study_id="bad", catalogs=["../private.json"])
