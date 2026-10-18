from copy import deepcopy
from pathlib import Path

import pytest

from gatebench.config import load_config
from gatebench.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root():
    return ROOT


@pytest.fixture
def task():
    return deepcopy(load_tasks(ROOT / "data/pilot.json")["tasks"][0])


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs/smoke.toml")
