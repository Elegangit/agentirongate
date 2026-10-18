from copy import deepcopy
from http.client import HTTPConnection
import json
from threading import Thread

import pytest

from gatebench.experiment import run_experiment
from gatebench.io import digest, read_json
from gatebench.viewer.app import build_data, bundle, export, inside, make_server


@pytest.fixture
def saved_run(root, cfg, task, tmp_path):
    output = tmp_path / "run"
    run_experiment(root, {**cfg, "repeats": 1},
                   {"schema_version": 1, "tasks": [task]}, output)
    return output


def test_loads_recorded_feedback_without_writing(root, saved_run):
    before = {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    data = build_data(root, saved_run)
    assert data["evidence"] == "scripted_fixture_only"
    assert len(data["branches"]) == 3
    branches = {b["condition"]: b for b in data["branches"]}
    assert branches["minimal"]["feedback"]["message"] == "Action denied."
    assert len(branches["directed"]["feedback"]["message"]) > len(
        branches["rule"]["feedback"]["message"])
    assert all(b["metrics"]["completion"] == 1 for b in data["branches"])
    assert all(len(b["events"]) == 2 for b in data["branches"])
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not (saved_run / "analysis").exists()
    assert "environment" not in data and "source_hashes" not in data


def test_benign_no_branches(root, cfg, tmp_path):
    output = tmp_path / "benign"
    run_experiment(root, cfg, read_json(root / "data/benign.json"), output)
    data = build_data(root, output)
    assert data["branches"] == [] and data["checkpoints"] == {}
    assert len(data["prefixes"]) == 4


def test_checkpoint_tampering_rejected(root, saved_run):
    path = next((saved_run / "checkpoints").glob("*.json"))
    cp = read_json(path)
    cp["payload"]["state"]["label"] = "confidential"
    path.write_text(json.dumps(cp))
    with pytest.raises(ValueError, match="integrity"):
        build_data(root, saved_run)


def test_trace_tampering_rejected(root, saved_run):
    path = next((saved_run / "branches").glob("*.json"))
    trace = read_json(path)
    trace["metrics"]["completion"] = 42
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="integrity"):
        build_data(root, saved_run)


def test_export_is_portable_and_does_not_overwrite(root, saved_run, tmp_path):
    data = build_data(root, saved_run)
    output = export(data, tmp_path / "site")
    assert {p.name for p in output.iterdir()} == {"index.html", "style.css", "app.js", "data.json"}
    assert read_json(output / "data.json") == data
    assert 'src="./app.js"' in (output / "index.html").read_text()
    with pytest.raises(FileExistsError):
        export(data, output)


def test_boundary_and_symlinks(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "escape").symlink_to(tmp_path)
    for relative in ("../secret", "escape/secret", str(tmp_path / "secret")):
        with pytest.raises(ValueError, match="escapes"):
            inside(run, relative)


def test_display_name_comes_from_current_project(root, saved_run, tmp_path):
    project = root.joinpath("project.toml").read_text()
    old = build_data(root, saved_run)["name"]
    tmp_path.joinpath("project.toml").write_text(project.replace(old, "Renamed Gate"))
    assert build_data(tmp_path, saved_run)["name"] == "Renamed Gate"


def test_fixed_server_routes(root, saved_run):
    server = make_server(build_data(root, saved_run), 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        for path, expected in (("/", 200), ("/data.json", 200), ("/../project.toml", 404),
                               ("/.env", 404), ("/outputs/", 404)):
            conn.request("GET", path)
            response = conn.getresponse()
            assert response.status == expected
            if expected == 200:
                assert response.getheader("Content-Security-Policy")
            response.read()
        conn.request("POST", "/data.json", body="{}")
        response = conn.getresponse()
        assert response.status == 501
        response.read()
        conn.request("GET", "/data.json", headers={"Host": "untrusted.example"})
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_bundle_renders_data_as_json_not_inline_script(root, saved_run):
    data = deepcopy(build_data(root, saved_run))
    data["name"] = "</script><script>alert(1)</script>"
    files = bundle(data)
    assert data["name"].encode() not in files["/"][1]
    assert json.loads(files["/data.json"][1])["name"] == data["name"]


def test_prefix_mismatch_rejected_even_with_updated_trace_hash(root, saved_run):
    rows_path = saved_run / "branches.jsonl"
    rows = [json.loads(s) for s in rows_path.read_text().splitlines()]
    path = saved_run / rows[0]["trace"]
    trace = read_json(path)
    trace["post_checkpoint_event_start"] = 0
    path.write_text(json.dumps(trace))
    rows[0]["trace_sha256"] = digest(trace)
    rows_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="prefix differs"):
        build_data(root, saved_run)
