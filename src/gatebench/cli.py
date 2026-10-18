import argparse
import json
from pathlib import Path

from .analysis import analyze, plot
from .branding import render_identity
from .config import identity, load_config
from .experiment import run_experiment
from .providers import ProviderError
from .tasks import load_tasks
from .io import read_json
from .study import freeze_plan, preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description="Controlled post-denial agent experiments")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    sub = parser.add_subparsers(dest="command", required=True)
    branding = sub.add_parser("identity", help="Update the README identity block from project.toml")
    branding.add_argument("--latex", type=Path, help="Also write paper naming macros to this file")
    viewer = sub.add_parser("view", help="Inspect a saved run in a read-only graphical viewer")
    viewer.add_argument("run_dir")
    viewer.add_argument("--port", type=int, default=8765)
    viewer.add_argument("--export", type=Path, help="Write a static site to a new directory")
    audit = sub.add_parser("audit", help="Export condition-blinded outcomes for human review")
    audit.add_argument("run_dir")
    audit.add_argument("--output", required=True, type=Path, help="New private audit directory")
    audit_summary = sub.add_parser("audit-summary", help="Compare two completed human-rating sheets")
    audit_summary.add_argument("audit_dir")
    audit_summary.add_argument("--left", required=True, type=Path)
    audit_summary.add_argument("--right", required=True, type=Path)
    audit_summary.add_argument("--output", required=True, type=Path)
    human = sub.add_parser("human-analysis", help="Analyze adjudicated ratings separately from automatic scores")
    human.add_argument("run_dir")
    human.add_argument("audit_dir")
    for flag in ("left", "right", "adjudication", "output"):
        human.add_argument("--" + flag, required=True, type=Path)
    recovery = sub.add_parser("check-recovery", help="Check reference paths offline on saved checkpoints")
    recovery.add_argument("run_dir")
    recovery.add_argument("--catalog", required=True, type=Path)
    recovery.add_argument("--output", required=True, type=Path)
    for name in ("validate", "run", "preflight"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="configs/smoke.toml")
        cmd.add_argument("--tasks", default="data/pilot.json")
        if name in {"run", "preflight"}:
            cmd.add_argument("--plan", type=Path, help="Frozen study plan to verify")
            cmd.add_argument("--override-reason", help="Explain a deviation from an exploratory plan")
        if name == "preflight":
            cmd.add_argument("--mode", choices=["confirmatory", "exploratory"],
                             default="confirmatory")
            cmd.add_argument("--freeze", type=Path, help="Write a new plan; never overwrite")
            cmd.add_argument("--study-id", default="pilot")
            cmd.add_argument("--catalog", type=Path, action="append", help="Bind a recovery catalog; repeatable")
        if name == "run":
            cmd.add_argument("--output", required=True, help="New output directory; never overwritten")
            cmd.add_argument("--allow-network", action="store_true")
    for name in ("analyze", "plot"):
        cmd = sub.add_parser(name)
        cmd.add_argument("run_dir")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        project = identity(root)
        if args.command == "identity":
            print(render_identity(root, latex=args.latex))
        elif args.command == "human-analysis":
            from .human_analysis import human_analysis
            print(human_analysis(root / args.run_dir, root / args.audit_dir,
                                 root / args.left, root / args.right, root / args.adjudication,
                                 root / args.output))
        elif args.command == "check-recovery":
            from .recovery import check_recovery
            print(check_recovery(root / args.run_dir, root / args.catalog, root / args.output))
        elif args.command == "audit-summary":
            from .audit import summarize_audit
            print(summarize_audit(root / args.audit_dir, root / args.left,
                                  root / args.right, root / args.output))
        elif args.command == "audit":
            from .audit import export_audit
            print(export_audit(root / args.run_dir, root / args.output))
            print("Give raters only rater_cases.json and separate copies of ratings.csv. "
                  "Keep PRIVATE_mapping.json withheld and do not publish it.")
        elif args.command == "view":
            from .viewer.app import build_data, export, serve
            data = build_data(root, root / args.run_dir)
            if args.export:
                print(export(data, root / args.export))
                print("Review data.json for sensitive content before publishing.")
            else:
                if not 1 <= args.port <= 65535:
                    raise ValueError("Port must be between 1 and 65535")
                serve(data, args.port)
        elif args.command in {"analyze", "plot"}:
            path = root / args.run_dir
            print(analyze(path) if args.command == "analyze" else plot(path) or "Plots saved")
        else:
            cfg = load_config(root / args.config)
            dataset = load_tasks(root / args.tasks)
            if args.command == "preflight":
                plan = read_json(root / args.plan) if args.plan else None
                report = preflight(cfg, dataset, mode=args.mode, plan=plan, root=root,
                                   catalogs=args.catalog, override_reason=args.override_reason)
                print(json.dumps(report, indent=2))
                if not report["ready"]:
                    parser.exit(2)
                if args.freeze:
                    if args.plan:
                        raise ValueError("Verify a plan or freeze a new one, not both")
                    freeze_plan(cfg, dataset, root / args.freeze,
                                study_id=args.study_id, mode=args.mode, root=root,
                                catalogs=args.catalog or ())
                return
            upper = len(dataset["tasks"]) * len(cfg["models"]) * (
                cfg["prefix_calls"] + cfg["repeats"] * len(cfg["conditions"]) * cfg["branch_calls"])
            print(f"{project['name']}: {len(dataset['tasks'])} cases; "
                  f"{len({t['cluster'] for t in dataset['tasks']})} source clusters")
            print(f"Core call upper bound: {upper}; API request cap: {cfg['max_api_calls']}")
            if args.command == "run":
                manifest = run_experiment(root, cfg, dataset, root / args.output,
                                          allow_network=args.allow_network,
                                          study_plan=read_json(root / args.plan) if args.plan else None,
                                          override_reason=args.override_reason)
                print(f"{manifest['status']}: {manifest['branches']} continuations; "
                      f"evidence={manifest['evidence']}")
                if manifest["model_version_audit"]["flagged"]:
                    print("Warning: missing or mixed model-version evidence; inspect the manifest.")
    except (ValueError, OSError, ProviderError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
