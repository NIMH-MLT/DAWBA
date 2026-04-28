# conftest.py
# Partially written by an LLM
import pytest
import numpy as np
import pandas as pd
from src.config import SEED
from src.utils import col_config_sensitivity


IMPACT_TYPE = "self"
QUEST_COLS = col_config_sensitivity[f"{IMPACT_TYPE.lower()}_quest"]
N_OUTCOMES_SHOT = 31
N_OUTCOMES_LR = 28


def _make_prob_vector(n_outcomes, size):
    """Return a list of lists — each row is a probability vector."""
    return [list(row) for row in np.random.rand(size, n_outcomes)]


@pytest.fixture
def base_df():
    """Minimal DataFrame that satisfies ResidualRegressor._build_X_blocks()."""
    np.random.seed(SEED)
    n = 40
    quest_col_dict = {quest: np.random.randint(0, 4, n) for quest in QUEST_COLS}

    df = pd.DataFrame(
        {
            "eligible_group": np.tile([0, 1], n // 2 + 1)[:n],
            "DAWBA_ID": np.arange(1000, 1000 + n),
            **quest_col_dict,
            # vector columns used by different agg_types
            "prob_vector_avg": _make_prob_vector(N_OUTCOMES_LR, n),
            "prob_vector_max": _make_prob_vector(N_OUTCOMES_LR, n),
            "prob_vector_block": _make_prob_vector(N_OUTCOMES_LR, n),
            "open_text_block_0_shot_binary": _make_prob_vector(N_OUTCOMES_SHOT, n),
            "open_text_block_0_shot": _make_prob_vector(N_OUTCOMES_SHOT, n),
            # target
            # "impact_score_self": np.random.randint(0, 15, n),
            # CV split column (populated by add_split_col in real use)
            "split_id": np.tile(np.arange(4), n // 4 + 1)[:n],
        }
    )
    df["impact_score_self"] = df["DAWBA_ID"] * 10
    return df


@pytest.fixture
def base_df_no_splits(base_df):
    return base_df.drop("split_id", axis=1)


@pytest.fixture
def y(base_df):
    return base_df["impact_score_self"].values.ravel()
