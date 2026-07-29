#conftest.py

import numpy as np
import pandas as pd
import pytest

import src.Regressor as reg_mod
from src.Regressor import IndependentRegressor, ResidualRegressor


# shared column vocabulary
QUEST_COLS = ["q1", "q2", "q3"]
PROB_OUTCOMES = ["t1", "t2", "t3"]
PROB_COL = "open_text_block_0_shot"


@pytest.fixture
def quest_cols():
    return list(QUEST_COLS)


@pytest.fixture
def prob_outcomes():
    return list(PROB_OUTCOMES)


@pytest.fixture(autouse=True)
def patch_prob_outcomes(monkeypatch):
    """
    The Regressor resolves its text-feature columns via
    get_prob_col_outcomes(agg_type, desc_seg_type). Pin them to a fixed set so
    the model logic is decoupled from src.utils' real config tables. Autouse so
    every Regressor fit in the suite sees the same text block; harmless for
    tests (e.g. the CV splitter) that never touch src.Regressor.
    """
    monkeypatch.setattr(
        reg_mod,
        "get_prob_col_outcomes",
        lambda agg_type, desc_seg_type: (PROB_COL, list(PROB_OUTCOMES)),
    )


# Regressor feature df
@pytest.fixture
def make_xy():
    """
    Factory for a synthetic feature frame + target.

    X columns: DAWBA_ID, quest cols, prob-outcome cols, plus an extra column the
    regressor must ignore.

    y_from:
        'random' -> noise (default)
        'sum'    -> exact affine function of the questionnaire sum
        'text'   -> exact affine function of t1 only
    quest_constant: make every questionnaire column constant.
    """

    def _make_xy(n=12, seed=0, quest_constant=False, y_from="random"):
        rng = np.random.default_rng(seed)
        ids = np.arange(100, 100 + n)
        if quest_constant:
            q = {c: np.full(n, 2.0) for c in QUEST_COLS}
        else:
            q = {c: rng.integers(0, 4, n).astype(float) for c in QUEST_COLS}
        t = {c: rng.random(n) for c in PROB_OUTCOMES}
        df = pd.DataFrame({"DAWBA_ID": ids, **q, **t, "ignored": rng.random(n)})

        sum_score = df[QUEST_COLS].sum(axis=1).to_numpy()
        if y_from == "sum":
            y = 2.0 + 3.0 * sum_score
        elif y_from == "text":
            y = 1.5 + 5.0 * df["t1"].to_numpy()
        else:
            y = rng.random(n) * 10.0
        return df, pd.Series(y, name="target")

    return _make_xy


# Regressor builders
@pytest.fixture
def make_resid():
    def _make(
        num_layers,
        two_layer_base=None,
        alpha2=2.5,
        alpha3=5.0,
        outcome_sub_list=None,
        seed=42,
        linear_type="ridge",
    ):
        return ResidualRegressor(
            num_layers=num_layers,
            two_layer_base=two_layer_base,
            impact_type="self",
            desc_seg_type="original",
            quest_cols=list(QUEST_COLS),
            agg_type="max",
            alpha2=alpha2,
            alpha3=alpha3,
            seed=seed,
            linear_type=linear_type,
            outcome_sub_list=outcome_sub_list,
            max_iter=5000,
        )

    return _make


@pytest.fixture
def make_independent():
    def _make(alpha3=5.0, outcome_sub_list=None, seed=42):
        return IndependentRegressor(
            num_layers=1,
            impact_type="self",
            desc_seg_type="original",
            quest_cols=list(QUEST_COLS),
            agg_type="max",
            alpha2=2.5,
            alpha3=alpha3,
            seed=seed,
            linear_type="ridge",
            outcome_sub_list=outcome_sub_list,
            max_iter=5000,
        )

    return _make


@pytest.fixture
def stub_linear():
    """Returns a constructor for a fitted-model stand-in (coef_/intercept_)."""

    class _StubLinear:
        def __init__(self, coef, intercept):
            self.coef_ = np.asarray(coef, dtype=float)
            self.intercept_ = intercept

    return _StubLinear


# --- Stratified-CV frames ---------------------------------------------------- #
@pytest.fixture
def make_stratified_df():
    """
    Factory for a df fed to add_split_col: balanced binary eligible_group,
    DAWBA_ID, and impact_score columns. Rows are shuffled so eligible_group is
    not index-ordered going into the splitter.
    """

    def _make(n_per_group=15, groups=(0, 1), seed=0, impact_types=("self", "parent")):
        rng = np.random.default_rng(seed)
        eg = np.concatenate([np.full(n_per_group, g) for g in groups])
        n = len(eg)
        eg = eg[rng.permutation(n)]
        data = {"DAWBA_ID": np.arange(n), "eligible_group": eg}
        for it in impact_types:
            data[f"impact_score_{it}"] = rng.random(n)
        return pd.DataFrame(data)

    return _make


@pytest.fixture
def make_split_frame():
    """
    Factory for a df with an explicit split_id / eligible_group composition,
    so check_ratios can be exercised in isolation from the splitter.

    spec: {split_id: {group: count, ...}, ...}
    """

    def _make(spec, impact_type="self"):
        recs = []
        for split_id, comp in spec.items():
            for group, count in comp.items():
                recs.extend(
                    {"split_id": split_id, "eligible_group": group}
                    for _ in range(count)
                )
        df = pd.DataFrame(recs)
        df[f"impact_score_{impact_type}"] = np.arange(len(df), dtype=float)
        return df

    return _make