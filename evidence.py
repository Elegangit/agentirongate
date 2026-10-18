"""Read a checkpoint bound to a recorded branch."""

from pathlib import Path

from .checkpoint import Checkpoint
from .io import canonical, read_json


def load_checkpoint(run_dir, row):
    run_dir = Path(run_dir).resolve()
    path = (run_dir / "checkpoints" / f"{row['model_id']}--{row['task_id']}.json").resolve()
    if not path.is_relative_to(run_dir):
        raise ValueError("Checkpoint path escapes run directory")
    saved = read_json(path)
    if saved["sha256"] != row["checkpoint_hash"]:
        raise ValueError("Checkpoint differs from branch")
    checkpoint = Checkpoint(canonical(saved["payload"]), saved["sha256"])
    checkpoint.restore()
    return checkpoint
