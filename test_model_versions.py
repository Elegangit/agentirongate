from gatebench.analysis import analyze
from gatebench.experiment import run_experiment
from gatebench.io import read_json
from gatebench.model_versions import ModelVersionAudit
from gatebench.providers import OfflineProvider


def test_missing_mixed_and_empty_versions(cfg):
    audit = ModelVersionAudit(cfg["models"])
    assert audit.report()["models"][0]["flags"] == ["no_returned_turns"]
    audit.observe("fixture", [{"model_version": "v1"}, {"model_version": "v2"},
                              {"model_version": None}, {}, {"model_version": " "}])
    record = audit.report()["models"][0]
    assert record["missing_identifiers"] == 3 and record["returned_turns"] == 5
    assert record["reported_versions"] == {"v1": 1, "v2": 1}
    assert record["flags"] == ["mixed_model_versions", "missing_model_version"]


def test_versions_are_separate_by_model(cfg):
    audit = ModelVersionAudit([cfg["models"][0], {**cfg["models"][0], "id": "other"}])
    audit.observe("fixture", [{"model_version": "v1"}])
    audit.observe("other", [{"model_version": "v2"}])
    assert not audit.report()["flagged"]


def test_manifest_and_analysis_flag_without_filtering(root, cfg, task, tmp_path):
    class Mixed(OfflineProvider):
        calls = 0

        def next_turn(self, history):
            self.calls += 1
            turn = super().next_turn(history)
            turn.model_version = "v1" if self.calls == 1 else "v2"
            return turn

    run = tmp_path / "run"
    manifest = run_experiment(root, cfg, {"tasks": [task]}, run,
                              provider_factory=lambda *args: Mixed(task))
    assert manifest["status"] == "complete" and manifest["model_version_audit"]["flagged"]
    assert manifest["branches"] == 9
    analyze(run)
    report = read_json(run / "analysis/summary.json")
    assert report["model_version_audit"] == manifest["model_version_audit"]
    assert report["paired"]
