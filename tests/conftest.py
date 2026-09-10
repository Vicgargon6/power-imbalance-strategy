import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omie_imbalance import data  # noqa: E402


@pytest.fixture(scope="session")
def df():
    frame, _ = data.load(ROOT / "data" / "raw" / "market_data.csv")
    return frame
