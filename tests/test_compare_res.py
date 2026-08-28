# test_compare_res.py
# Run w/ uv run pytest tests/test_compare_res.py -v
# Written in collaboration w/ Claude Opus 5

import numpy as np
import pandas as pd
import pytest

import src.compare_res as cmp_mod
from src.compare_res import (
    compute_r2s,
    export_data,
    get_avg_r2,
    get_layer1,
    get_layer2,
    get_layer3,
    get_score,
    get_top_models,
)

PARAMS = [
    "num_layers",
    "agg_type",
    "llm_model",
    "desc_seg",
    "alpha",
    "two_layer_base",
    "temp",
]


######## Fixtures ########
@pytest.fixture(autouse=True)
def stub_plots(monkeypatch):
    """Nothing here should open a figure."""
    monkeypatch.setattr(cmp_mod.plt, "show", lambda *a, **kw: None)


@pytest.fixture
def make_pred_df():
    """Prediction rows as aggregate_and_export writes them: one row per
    participant per fold per repeat, tagged with the full config."""

    def _make(n_ids=6, n_splits=2, n_repeats=2, configs=None, seed=0):
        rng = np.random.default_rng(seed)
        configs = configs or [
            dict(num_layers=3, alpha="1.0_5.0", two_layer_base="None"),
        ]
        rows = []
        for cfg in configs:
            for repeat in range(n_repeats):
                for i in range(n_ids):
                    true = float(i)
                    rows.append(
                        {
                            "DAWBA_ID": i + 1,
                            "repeat": repeat,
                            "split_id": i % n_splits,
                            "true": true,
                            "pred_stage1": true + rng.normal(0, 1.0),
                            "pred_stage2_total": true + rng.normal(0, 0.5),
                            "pred_stage3_total": true + rng.normal(0, 0.2),
                            "agg_type": "scale-new",
                            "llm_model": "gemini-2.5-pro",
                            "desc_seg": "shot_few",
                            "temp": 0.4,
                            **cfg,
                        }
                    )
        return pd.DataFrame(rows)

    return _make


@pytest.fixture
def make_r2_df():
    """A get_avg_r2-shaped frame: one row per config, flattened mean/std cols."""

    def _make(rows):
        base = dict(
            agg_type="scale-new",
            llm_model="gemini-2.5-pro",
            desc_seg="shot_few",
            temp=0.4,
            r2_S_std=0.01,
            r2_Q_std=0.01,
            r2_T_std=0.01,
        )
        return pd.DataFrame([{**base, **r} for r in rows])

    return _make


######## compute_r2s / get_avg_r2 ########
class TestComputeR2s:
    def test_scores_each_stage_against_truth(self):
        g = pd.DataFrame(
            {
                "true": [1.0, 2.0, 3.0, 4.0],
                "pred_stage1": [1.0, 2.0, 3.0, 4.0],  # perfect
                "pred_stage2_total": [1.1, 2.1, 3.1, 4.1],
                "pred_stage3_total": [2.5, 2.5, 2.5, 2.5],  # mean-only
            }
        )
        out = compute_r2s(g)
        assert list(out.index) == ["r2_S", "r2_Q", "r2_T"]
        assert out["r2_S"] == pytest.approx(1.0)
        assert out["r2_T"] == pytest.approx(0.0)
        assert out["r2_Q"] > out["r2_T"]


class TestGetAvgR2:
    def test_one_row_per_config_with_flattened_columns(self, make_pred_df):
        """Scores are computed within each repeat, then averaged across repeats --
        so the output is one row per config, not per repeat."""
        df = make_pred_df(n_repeats=3)
        out = get_avg_r2(df)
        assert len(out) == 1
        for stage in ("S", "Q", "T"):
            assert f"r2_{stage}_mean" in out.columns
            assert f"r2_{stage}_std" in out.columns
        assert "repeat" not in out.columns

    def test_separate_rows_per_config(self, make_pred_df):
        df = make_pred_df(
            configs=[
                dict(num_layers=3, alpha="1.0_5.0", two_layer_base="None"),
                dict(num_layers=3, alpha="5.0_1.0", two_layer_base="None"),
                dict(num_layers=2, alpha="1.0", two_layer_base="S"),
            ]
        )
        out = get_avg_r2(df)
        assert len(out) == 3
        assert set(out["alpha"]) == {"1.0_5.0", "5.0_1.0", "1.0"}


######## get_layer1 / get_layer2 / get_layer3 ########
class TestGetLayer1:
    def test_diff_is_against_the_single_best_Q(self, make_r2_df):
        """The 1-layer T configs are all compared to ONE questionnaire baseline --
        the best Q across every alpha -- not to their own row's Q."""
        r2_df = make_r2_df(
            [
                dict(
                    num_layers=1,
                    alpha="1.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.30,
                    r2_T_mean=0.50,
                ),
                dict(
                    num_layers=1,
                    alpha="5.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.40,
                    r2_T_mean=0.45,
                ),
            ]
        )
        out = get_layer1(r2_df)
        assert out["diff"].tolist() == [0.1, 0.05]  # both vs 0.40
        assert set(out["best_Q_alpha"]) == {"0.4_5.0"}

    def test_returns_none_when_no_one_layer_rows(self, make_r2_df):
        r2_df = make_r2_df(
            [
                dict(
                    num_layers=3,
                    alpha="1.0_5.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.2,
                    r2_T_mean=0.3,
                )
            ]
        )
        assert get_layer1(r2_df) is None

    def test_ignores_other_layer_counts(self, make_r2_df):
        r2_df = make_r2_df(
            [
                dict(
                    num_layers=1,
                    alpha="1.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.2,
                    r2_T_mean=0.3,
                ),
                dict(
                    num_layers=3,
                    alpha="1.0_5.0",
                    two_layer_base="None",
                    r2_S_mean=0.9,
                    r2_Q_mean=0.9,
                    r2_T_mean=0.9,
                ),
            ]
        )
        out = get_layer1(r2_df)
        assert len(out) == 1 and (out["num_layers"] == 1).all()


class TestGetLayer2:
    @pytest.fixture
    def two_layer(self, make_r2_df):
        return make_r2_df(
            [
                dict(
                    num_layers=2,
                    alpha="1.0",
                    two_layer_base="S",
                    r2_S_mean=0.20,
                    r2_Q_mean=0.0,
                    r2_T_mean=0.50,
                ),
                dict(
                    num_layers=2,
                    alpha="5.0",
                    two_layer_base="S",
                    r2_S_mean=0.20,
                    r2_Q_mean=0.0,
                    r2_T_mean=0.30,
                ),
                dict(
                    num_layers=2,
                    alpha="1.0_1.0",
                    two_layer_base="Q",
                    r2_S_mean=0.0,
                    r2_Q_mean=0.30,
                    r2_T_mean=0.60,
                ),
                dict(
                    num_layers=2,
                    alpha="5.0_1.0",
                    two_layer_base="Q",
                    r2_S_mean=0.0,
                    r2_Q_mean=0.45,
                    r2_T_mean=0.55,
                ),
            ]
        )

    def test_S_diff_is_row_wise_against_its_own_sum_baseline(self, two_layer):
        """The S base reuses the sum stage, so each row already carries the
        baseline it should be compared to."""
        s_base, _ = get_layer2(two_layer, filter_best_Q=False)
        assert s_base["diff"].tolist() == [0.30, 0.10]
        assert (s_base["two_layer_base"] == "S").all()

    def test_Q_diff_is_against_the_single_best_Q(self, two_layer):
        """The Q base fits an independent questionnaire model per alpha2, so the
        comparison is against the best of those, not each row's own."""
        _, q_base = get_layer2(two_layer, filter_best_Q=False)
        assert q_base["diff"].tolist() == [0.15, 0.10]  # both vs 0.45
        assert set(q_base["best_Q"]) == {0.45}

    def test_filter_best_Q_keeps_only_the_winning_baseline(self, two_layer):
        """Filtering holds the questionnaire stage fixed so the remaining rows
        differ only in the text stage."""
        _, q_base = get_layer2(two_layer, filter_best_Q=True)
        assert len(q_base) == 1
        assert q_base["r2_Q_mean"].iloc[0] == 0.45
        assert "best_Q" not in q_base.columns  # only set on the unfiltered path

    def test_both_frames_sorted_by_diff(self, two_layer):
        s_base, q_base = get_layer2(two_layer, filter_best_Q=False)
        for frame in (s_base, q_base):
            assert frame["diff"].tolist() == sorted(frame["diff"], reverse=True)


class TestGetLayer3:
    @pytest.fixture
    def three_layer(self, make_r2_df):
        return make_r2_df(
            [
                dict(
                    num_layers=3,
                    alpha="1.0_1.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.30,
                    r2_T_mean=0.50,
                ),
                dict(
                    num_layers=3,
                    alpha="1.0_5.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.30,
                    r2_T_mean=0.35,
                ),
                dict(
                    num_layers=3,
                    alpha="5.0_1.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.45,
                    r2_T_mean=0.60,
                ),
            ]
        )

    def test_diff_against_best_Q_and_sorted(self, three_layer):
        out = get_layer3(three_layer, filter_best_Q=False)
        assert out["diff"].tolist() == [0.15, 0.05, -0.10]  # all vs 0.45 (SORTED)
        assert set(out["best_Q_alpha"]) == {"0.45_5.0"}

    def test_filter_keeps_rows_sharing_the_best_Q(self, three_layer):
        """alpha2 drives the questionnaire stage, so filtering on the best Q keeps
        every alpha3 that shares that alpha2 -- here just one."""
        out = get_layer3(three_layer, filter_best_Q=True)
        assert len(out) == 1
        assert out["alpha"].iloc[0] == "5.0_1.0"

    def test_filter_keeps_all_alpha3_for_the_winning_alpha2(self, make_r2_df):
        r2_df = make_r2_df(
            [
                dict(
                    num_layers=3,
                    alpha="5.0_1.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.45,
                    r2_T_mean=0.60,
                ),
                dict(
                    num_layers=3,
                    alpha="5.0_5.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.45,
                    r2_T_mean=0.55,
                ),
                dict(
                    num_layers=3,
                    alpha="1.0_1.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.30,
                    r2_T_mean=0.70,
                ),
            ]
        )
        out = get_layer3(r2_df, filter_best_Q=True)
        assert set(out["alpha"]) == {"5.0_1.0", "5.0_5.0"}

    def test_best_Q_matching_is_exact_float_equality(self, make_r2_df):
        """
        r^2 values should be rounded when get_layer3 is called, but
        checks for float point errors just in case.
        """
        r2_df = make_r2_df(
            [
                dict(
                    num_layers=3,
                    alpha="5.0_1.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.450,
                    r2_T_mean=0.60,
                ),
                dict(
                    num_layers=3,
                    alpha="5.0_5.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.4500001,
                    r2_T_mean=0.55,
                ),
            ]
        )
        out = get_layer3(r2_df, filter_best_Q=True)
        assert set(out["alpha"]) == {"5.0_1.0", "5.0_5.0"}


######## get_top_models ########
class TestGetTopModels:
    @pytest.fixture
    def mixed(self, make_r2_df):
        rows = []
        for i, a in enumerate(["1.0", "5.0"]):
            rows.append(
                dict(
                    num_layers=1,
                    alpha=a,
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.2,
                    r2_T_mean=0.3 + 0.1 * i,
                )
            )
            rows.append(
                dict(
                    num_layers=2,
                    alpha=a,
                    two_layer_base="S",
                    r2_S_mean=0.2,
                    r2_Q_mean=0.0,
                    r2_T_mean=0.4 + 0.1 * i,
                )
            )
            rows.append(
                dict(
                    num_layers=2,
                    alpha=f"{a}_1.0",
                    two_layer_base="Q",
                    r2_S_mean=0.0,
                    r2_Q_mean=0.3,
                    r2_T_mean=0.5 + 0.1 * i,
                )
            )
            rows.append(
                dict(
                    num_layers=3,
                    alpha=f"{a}_1.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.3,
                    r2_T_mean=0.6 + 0.1 * i,
                )
            )
        return make_r2_df(rows)

    @pytest.mark.parametrize(
        "num_layers,two_layer_base,expected_base",
        [(1, None, "None"), (2, "S", "S"), (2, "Q", "Q"), (3, None, "None")],
    )
    def test_routes_to_the_right_layer(
        self, mixed, num_layers, two_layer_base, expected_base
    ):
        top, _ = get_top_models(
            mixed, PARAMS, num_layers=num_layers, two_layer_base=two_layer_base
        )
        assert (top["num_layers"] == num_layers).all()
        assert (top["two_layer_base"] == expected_base).all()

    def test_two_layer_requires_a_base(self, mixed):
        with pytest.raises(ValueError, match="two_layer_base"):
            get_top_models(mixed, PARAMS, num_layers=2, two_layer_base=None)

    def test_top_row_is_the_largest_diff(self, mixed):
        top, best_params = get_top_models(mixed, PARAMS, num_layers=3)
        assert top["diff"].iloc[0] == top["diff"].max()
        assert best_params["alpha"] == top["alpha"].iloc[0]

    def test_top_n_truncates(self, mixed):
        top, _ = get_top_models(mixed, PARAMS, top_n=2, num_layers=3)
        assert len(top) == 2

    def test_returned_params_are_exactly_param_list(self, mixed):
        """top_model_params feeds get_score's mask, so its keys must match the
        prediction frame's config columns."""
        _, best_params = get_top_models(mixed, PARAMS, num_layers=3)
        assert list(best_params.index) == PARAMS

    def test_unhandled_num_layers_raises_value_error(self, mixed):
        with pytest.raises(ValueError, match="num_layers"):
            get_top_models(mixed, PARAMS, num_layers=4)

    def test_missing_one_layer_rows_fail_opaquely(self, make_r2_df):
        # TODO(bug C): get_layer1 returns None when no 1-layer configs exist, and
        # get_top_models calls .sort_values on it immediately -- so an incomplete
        # aggregate surfaces as AttributeError on NoneType.
        r2_df = make_r2_df(
            [
                dict(
                    num_layers=3,
                    alpha="1.0_1.0",
                    two_layer_base="None",
                    r2_S_mean=0.1,
                    r2_Q_mean=0.2,
                    r2_T_mean=0.3,
                )
            ]
        )
        with pytest.raises(AttributeError):
            get_top_models(r2_df, PARAMS, num_layers=1)


######## get_score ########
class TestGetScore:
    def test_returns_fold_and_repeat_level_for_both_stages(self, make_pred_df):
        """Four frames: stage-3 per fold/repeat, stage-3 per repeat, then the same
        two for stage 2."""
        df = make_pred_df(n_ids=6, n_splits=2, n_repeats=3)
        rf, r, rf_q, r_q = get_score(df, "r2")
        assert len(rf) == 3 * 2 and list(rf.columns) == ["repeat", "split_id", "r2"]
        assert len(r) == 3 and list(r.columns) == ["repeat", "r2"]
        assert len(rf_q) == 3 * 2
        assert len(r_q) == 3

    @pytest.mark.parametrize("score_type", ["r2", "mae"])
    def test_both_metrics_supported(self, make_pred_df, score_type):
        rf, _, _, _ = get_score(make_pred_df(), score_type)
        assert rf.columns[-1] == score_type
        if score_type == "mae":
            assert (rf["mae"] >= 0).all()

    def test_unknown_metric_raises(self, make_pred_df):
        with pytest.raises(KeyError):
            get_score(make_pred_df(), "rmse")

    def test_param_vals_filters_to_one_config(self, make_pred_df):
        """The scores must come from the selected config alone -- an unfiltered
        frame would average several models together and look plausible."""
        df = make_pred_df(
            n_repeats=1,
            configs=[
                dict(num_layers=3, alpha="1.0_5.0", two_layer_base="None"),
                dict(num_layers=3, alpha="5.0_1.0", two_layer_base="None"),
            ],
        )
        params = pd.Series(
            {"num_layers": 3, "alpha": "1.0_5.0", "two_layer_base": "None"}
        )
        rf, _, _, _ = get_score(df, "r2", param_vals=params)
        expected = df[df["alpha"] == "1.0_5.0"]
        assert len(rf) == expected["split_id"].nunique()

    def test_param_vals_none_placeholder_matches(self, make_pred_df):
        """get_avg_r2 writes the string "None" into two_layer_base; the mask
        fillna makes a NaN in the param series match it."""
        df = make_pred_df(
            configs=[dict(num_layers=3, alpha="1.0_5.0", two_layer_base="None")]
        )
        params = pd.Series({"num_layers": 3, "two_layer_base": np.nan})
        rf, _, _, _ = get_score(df, "r2", param_vals=params)
        assert len(rf) > 0

    def test_nonmatching_params_raises_error(self, make_pred_df):
        params = pd.Series({"num_layers": 3, "alpha": "does_not_exist"})
        with pytest.raises(
            ValueError, match="No matching rows found for the given parameter values."
        ):
            get_score(make_pred_df(), "r2", param_vals=params)

    def test_fold_scores_differ_from_pooled_repeat_scores(self, make_pred_df):
        """The two granularities are not interchangeable -- r2 over pooled folds
        is not the mean of the per-fold r2s, which is why both are reported."""
        df = make_pred_df(n_ids=8, n_splits=2, n_repeats=2)
        rf, r, _, _ = get_score(df, "r2")
        pooled = r.set_index("repeat")["r2"]
        per_fold_mean = rf.groupby("repeat")["r2"].mean()
        assert not np.allclose(pooled, per_fold_mean)
