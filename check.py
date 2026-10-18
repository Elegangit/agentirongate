"""Run local checks with the active Python environment."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from gatebench.experiment import environment


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    evidence = root / "validation/local" / stamp
    evidence.mkdir(parents=True)
    smoke = "outputs/validation-" + stamp
    benign = smoke + "-benign"
    structural = smoke + "-structural"
    plan = smoke + "-plan.json"
    commands = [
        ["-m", "ruff", "check", "src", "tests", "validation"],
        ["-m", "pytest", "-q"],
        ["-m", "pip", "check"],
        ["-m", "gatebench", "identity"],
        ["-m", "gatebench", "validate"],
        ["-m", "gatebench", "validate", "--config", "configs/gemini.toml"],
        ["-m", "gatebench", "preflight", "--freeze", plan, "--study-id", "offline-validation"],
        ["-m", "gatebench", "run", "--plan", plan, "--output", smoke],
        ["-m", "gatebench", "analyze", smoke],
        ["-m", "gatebench", "plot", smoke],
        ["-m", "gatebench", "run", "--tasks", "data/benign.json", "--output", benign],
        ["-m", "gatebench", "analyze", benign],
        ["-m", "gatebench", "plot", benign],
        ["-m", "gatebench", "run", "--tasks", "data/structural_checks.json", "--output", structural],
        ["-m", "gatebench", "check-recovery", structural, "--catalog", "data/recovery_checks.json",
         "--output", structural + "-references"],
    ]
    results = []
    for index, args in enumerate(commands, 1):
        result = subprocess.run([sys.executable, *args], cwd=root, text=True,
                                capture_output=True, timeout=120)
        log = result.stdout + result.stderr
        (evidence / f"{index:02d}.txt").write_text(log)
        results.append({"command": args, "returncode": result.returncode, "output": log})
        print(index, "PASS" if result.returncode == 0 else "FAIL", " ".join(args), flush=True)
        if result.returncode:
            break
    report = {"timestamp_utc": stamp, "python_executable": sys.executable,
              "environment": environment(root), "checks": results,
              "smoke": smoke, "benign": benign,
              "all_passed": len(results) == len(commands) and all(r["returncode"] == 0 for r in results),
              "evidence_scope": "Offline mechanics and mocked API tests, not live model effectiveness."}
    (root / "validation/latest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved validation/latest.json")
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
