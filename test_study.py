from copy import deepcopy

import pytest

from gatebench.config import load_config
from gatebench.experiment import run_experiment
from gatebench.study import freeze_plan, preflight
from gatebench.tasks import load_tasks


def test_live_bounds_and_placeholder(root):
    cfg = load_config(root / "configs/gemini.toml")
    tasks = load_tasks(root / "data/pilot.json")
    assert not preflight(cfg, tasks)["ready"]
    cfg["repeats"] = 5
    cfg["models"][0]["model"] = "explicit-test-model"
    report = preflight(cfg, tasks)
    assert report["request_upper_bound"] == 1152 and not report["ready"]
    assert preflight(cfg, tasks, mode="exploratory")["ready"]
    cfg["max_api_calls"] = 1152
    assert preflight(cfg, tasks)["ready"]
    benign = load_tasks(root / "data/benign.json")
    assert preflight(cfg, benign)["request_upper_bound"] == 384
    cfg["conditions"].append("neutral")
    cfg["models"].append({**cfg["models"][0], "id": "second"})
    assert preflight(cfg, tasks)["request_upper_bound"] == 3024
    cfg["max_api_calls"] = 0
    assert not preflight(cfg, tasks, mode="exploratory")["ready"]


def test_freeze_verify_and_run(root, cfg, task, tmp_path):
    dataset = {"schema_version": 1, "tasks": [task]}
    path = tmp_path / "plan.json"
    plan = freeze_plan(cfg, dataset, path, study_id="test-plan", root=root)
    assert preflight(cfg, dataset, plan=plan, root=root)["ready"]
    manifest = run_experiment(root, cfg, dataset, tmp_path / "run", study_plan=plan)
    assert manifest["study_plan"] == plan
    with pytest.raises(FileExistsError):
        freeze_plan(cfg, dataset, path, study_id="test-plan", root=root)
    for field, value in (("repeats", 5), ("conditions", ["minimal", "directed"])):
        changed = {**cfg, field: value}
        assert not preflight(changed, dataset, plan=plan, root=root)["ready"]
        with pytest.raises(ValueError, match="Configuration"):
            run_experiment(root, changed, dataset, tmp_path / "bad", study_plan=plan)
        assert not (tmp_path / "bad").exists()
    changed_tasks = deepcopy(dataset)
    changed_tasks["tasks"][0]["prompt"] += " changed"
    assert not preflight(cfg, changed_tasks, plan=plan, root=root)["ready"]
    with pytest.raises(ValueError, match="integrity"):
        preflight(cfg, dataset, plan={**plan, "study_id": "modified"}, root=root)


def test_cli_plan(root, tmp_path, capsys):
    from gatebench.cli import main
    from gatebench.io import read_json
    main(["--root", str(root), "preflight", "--tasks", "data/structural_checks.json",
          "--catalog", "data/recovery_checks.json", "--freeze", str(tmp_path / "plan.json")])
    assert '"ready": true' in capsys.readouterr().out
    plan = read_json(tmp_path / "plan.json")
    assert plan["schema_version"] == 2 and "data/recovery_checks.json" in plan["binding"]["recovery_catalogs"]
    main(["--root", str(root), "run", "--tasks", "data/structural_checks.json",
          "--plan", str(tmp_path / "plan.json"), "--output", str(tmp_path / "run")])
    assert read_json(tmp_path / "run/manifest.json")["study_verification"]["ready"]
    with pytest.raises(SystemExit) as exc:
        main(["--root", str(root), "preflight", "--config", "configs/gemini.toml"])
    assert exc.value.code == 2
