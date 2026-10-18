"""Serve a selected run locally, or export the same view as a static site."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from ..analysis import verified_run
from ..checkpoint import Checkpoint
from ..config import identity
from ..io import canonical, read_json

ASSETS = Path(__file__).parent / "static"
FILES = {"/": ("index.html", "text/html"),
         "/index.html": ("index.html", "text/html"),
         "/app.js": ("app.js", "text/javascript"),
         "/style.css": ("style.css", "text/css")}


def inside(run_dir, relative):
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("Artifact path escapes run directory")
    return path


def build_data(root, run_dir):
    run_dir = Path(run_dir).resolve()
    for name in ("manifest.json", "dataset.json", "branches.jsonl", "prefixes.jsonl"):
        inside(run_dir, name)
    manifest, rows, prefixes = verified_run(run_dir)
    dataset = read_json(inside(run_dir, "dataset.json"))
    checkpoints, branches = {}, []
    for row in rows:
        trace = read_json(inside(run_dir, row["trace"]))
        key = f"{row['model_id']}--{row['task_id']}"
        if key not in checkpoints:
            saved = read_json(inside(run_dir, f"checkpoints/{key}.json"))
            cp = Checkpoint(canonical(saved["payload"]), saved["sha256"])
            state, history, call, rule = cp.restore()
            checkpoints[key] = {"hash": cp.sha256, "history_length": len(history),
                                "events": state.events, "policy": state.policy,
                                "label": state.label, "call": call, "rule": rule}
        cp = checkpoints[key]
        if trace["checkpoint_hash"] != cp["hash"]:
            raise ValueError("Branch does not match its checkpoint")
        start = trace["post_checkpoint_event_start"]
        if start != len(cp["events"]) or trace["state"]["events"][:start] != cp["events"]:
            raise ValueError("Branch prefix differs from checkpoint")
        # Show the response actually recorded, not a freshly rendered template.
        response = trace["history"][cp["history_length"]]["parts"][0]["functionResponse"]
        branches.append({"task_id": row["task_id"], "model_id": row["model_id"],
                         "repeat": row["repeat"], "condition": row["condition"],
                         "termination": row["termination"], "order": row["order"],
                         "metrics": trace["metrics"], "feedback": response["response"],
                         "events": trace["state"]["events"][start:],
                         "outbox": trace["state"]["outbox"],
                         "label": trace["state"]["label"]})
    cfg = manifest["config"]
    return {"name": identity(root)["name"], "run_name": run_dir.name,
            "recorded_name": manifest["identity"]["name"], "evidence": manifest["evidence"],
            "started_at": manifest["started_at"], "status": manifest["status"],
            "models": cfg["models"], "conditions": cfg["conditions"],
            "repeats": cfg["repeats"], "scope": cfg["feedback_scope"],
            "approver": cfg["approver"], "call_budget": cfg["branch_calls"],
            "proposal_budget": cfg["branch_proposals"],
            "tasks": [{"id": t["id"], "prompt": t["prompt"], "family": t["family"],
                       "cluster": t["cluster"]} for t in dataset["tasks"]],
            "prefixes": prefixes, "checkpoints": checkpoints, "branches": branches}


def bundle(data):
    files = {url: (mime, (ASSETS / name).read_bytes()) for url, (name, mime) in FILES.items()}
    files["/data.json"] = ("application/json", json.dumps(data, allow_nan=False).encode())
    return files


def make_server(data, port):
    files = bundle(data)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            host = self.headers.get("Host", "").split(":", 1)[0]
            if host not in {"127.0.0.1", "localhost"}:
                self.send_error(403)
                return
            # Never expose a filesystem directory or accept a filename from a URL.
            item = files.get(self.path.split("?", 1)[0])
            if item is None:
                self.send_error(404)
                return
            mime, body = item
            self.send_response(200)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; object-src 'none'; base-uri 'none'; "
                             "frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(data, port=8765):
    server = make_server(data, port)
    print(f"Research viewer: http://127.0.0.1:{server.server_port}", flush=True)
    print("Read-only saved run. No model calls. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def export(data, destination):
    destination = Path(destination)
    files = bundle(data)
    destination.mkdir(parents=True, exist_ok=False)
    for url, (_, body) in files.items():
        if url != "/":
            (destination / url[1:]).write_bytes(body)
    return destination
