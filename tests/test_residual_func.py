# test_residual_func.py
# Run w/ uv run pytest tests/test_residual_func.py -v
# Written in collab w/ Claude Opus 5

import numpy as np
import pandas as pd
import pytest

import src.Regressor as reg_mod
import src.residual_func as res_mod
from src.residual_func import (
    build_model,
    get_lr_output,
    get_saliency_map,
    prep_train_test,
)

PROB_OUTCOMES = ["t1", "t2", "t3", "t4"]
QUEST = {
    "self": ["s_dep_q1", "s_dep_q2", "s_dep_q3"],
    "parent": ["p_dep_q1", "p_dep_q2", "p_dep_q3"],
}


######## Fixtures ########
@pytest.fixture(autouse=True)
def stub_registry(monkeypatch):
    """get_prob_col_outcomes is only imported into both Regressor. col_config_sensitivity is only imported into residual_func."""

    monkeypatch.setattr(
        reg_mod,
        "get_prob_col_outcomes",
        lambda agg_type, desc_seg_type: ("prob_col", list(PROB_OUTCOMES)),
    )
    monkeypatch.setattr(
        res_mod,
        "col_config_sensitivity",
        {"self_quest": QUEST["self"], "parent_quest": QUEST["parent"]},
    )


@pytest.fixture(autouse=True)
def stub_expand(monkeypatch):
    """The fixtures already carry exploded probability columns, so expand_text is
    a pass-through."""
    monkeypatch.setattr(
        res_mod,
        "expand_text",
        lambda df, agg_type, desc_seg_type: (
            df.copy(),
            list(PROB_OUTCOMES),
        ),
    )


@pytest.fixture(autouse=True)
def stub_heatmap(monkeypatch):
    monkeypatch.setattr(res_mod, "plot_heatmap", lambda df, **kw: None)


@pytest.fixture
def make_df():
    def _make(n=24, n_splits=3, seed=0):
        rng = np.random.default_rng(seed)
        data = {
            "DAWBA_ID": np.arange(1, n + 1),
            "split_id": np.tile(np.arange(n_splits), n // n_splits + 1)[:n],
        }
        for cols in QUEST.values():
            for c in cols:
                data[c] = rng.integers(0, 3, n) * 1.0
        for c in PROB_OUTCOMES:
            data[c] = rng.random(n)
        df = pd.DataFrame(data)
        for impact in ("self", "parent"):
            df[f"impact_score_{impact}"] = (
                df[QUEST[impact]].sum(axis=1) + 2.0 * df["t1"] + rng.normal(0, 0.1, n)
            )
        return df

    return _make


@pytest.fixture
def model_config():
    """Mirrors the notebook: BEST_PARAM_DICT entry minus llm_model/temp, plus
    cohort fields and shared estimator settings."""

    def _make(impact_type="self", **overrides):
        cfg = dict(
            impact_type=impact_type,
            quest_cols=list(QUEST[impact_type]),
            agg_type="scale-new",
            desc_seg_type="shot_few",
            num_layers=3,
            alpha2=15.0,
            alpha3=75.0,
            max_iter=50000,
            seed=0,
            linear_type="ridge",
        )
        cfg.update(overrides)
        return cfg

    return _make


@pytest.fixture
def on_disk(make_df, tmp_path, monkeypatch):
    def _write(df=None):
        df = make_df() if df is None else df
        path = tmp_path / "train.parquet"
        df.to_parquet(path)
        monkeypatch.setattr(res_mod, "get_df_path", lambda **kw: path)
        return df

    return _write


######## prep_train_test ########
class TestPrepTrainTest:
    def test_split_is_a_partition(self, make_df):
        df = make_df(n=24, n_splits=3)
        X_tr, X_te, _, _ = prep_train_test(df, 1, "self", "scale-new", "original")

        assert len(X_te) == (df["split_id"] == 1).sum()
        assert len(X_tr) + len(X_te) == len(df)
        assert not (set(X_te["DAWBA_ID"]) & set(X_tr["DAWBA_ID"]))

    def test_target_removed_from_X(self, make_df):
        df = make_df()
        X_tr, X_te, _, _ = prep_train_test(df, 0, "self", "scale-new", "original")
        for X in (X_tr, X_te):
            assert "impact_score_self" not in X.columns
            assert "DAWBA_ID" in X.columns

    def test_X_and_y_share_row_order(self, make_df):
        """X.iloc[i] and y.iloc[i] must be the same participant"""
        df = make_df()
        X_tr, X_te, y_tr, y_te = prep_train_test(df, 2, "self", "scale-new", "original")
        truth = df.set_index("DAWBA_ID")["impact_score_self"]
        for X, y in ((X_tr, y_tr), (X_te, y_te)):
            np.testing.assert_allclose(
                y.to_numpy(), truth.loc[X["DAWBA_ID"]].to_numpy()
            )

    def test_empty_frame_raises(self):
        empty = pd.DataFrame(
            columns=["DAWBA_ID", "split_id", "impact_score_self"]
            + QUEST["self"]
            + PROB_OUTCOMES
        )
        with pytest.raises(AssertionError):
            prep_train_test(empty, 0, "self", "scale-new", "original")


class TestBuildModel:
    def test_coef_index_strips_cohort_prefix(self, model_config, on_disk):
        """build_model strips s_dep/p_dep so self and parent coefs can be joined;
        get_saliency_map reattaches it from the leading underscore."""
        on_disk()
        cfg = model_config("self")
        model, coef_df, _ = build_model(
            cfg, data_dir="ignored", temp=0.0, llm_model="ignored"
        )
        assert coef_df.index.name == "stage_feat"
        assert "stage2__q1" in coef_df.index
        assert not any("s_dep" in i for i in coef_df.index)


######## get_lr_output ########
class TestGetLrOutput:
    @pytest.fixture
    def call(self, make_df):
        def _call(num_layers=3, two_layer_base=None, split_id=0, df=None, **kw):
            return get_lr_output(
                df=make_df() if df is None else df,
                agg_type="scale-new",
                desc_seg_type="original",
                num_layers=num_layers,
                selected_alphas=kw.pop("selected_alphas", (15.0, 75.0)),
                split_id=split_id,
                impact_type=kw.pop("impact_type", "self"),
                max_iter=50000,
                seed=0,
                perm_dir=None,
                two_layer_base=two_layer_base,
                **kw,
            )

        return _call

    @pytest.mark.parametrize(
        "num_layers,two_layer_base,expected",
        [
            (
                1,
                None,
                [
                    "DAWBA_ID",
                    "pred_stage1",
                    "pred_stage2_total",
                    "pred_stage3_total",
                    "true",
                    "split_id",
                ],
            ),
            (
                3,
                None,
                [
                    "DAWBA_ID",
                    "pred_stage1",
                    "pred_stage2_increment",
                    "pred_stage2_total",
                    "pred_stage3_increment",
                    "pred_stage3_total",
                    "true",
                    "split_id",
                ],
            ),
        ],
    )
    def test_output_columns(self, call, num_layers, two_layer_base, expected):
        _, output_df, _ = call(num_layers=num_layers, two_layer_base=two_layer_base)
        assert list(output_df.columns) == expected

    @pytest.mark.parametrize("split_id", [0, 1, 2])
    def test_true_and_ids_stay_row_aligned(self, call, make_df, split_id):
        """`true` is assigned positionally onto predict_all's frame, so any index
        drift silently corrupts every downstream R²."""
        df = make_df(n=24, n_splits=3)
        _, output_df, _ = call(df=df, split_id=split_id)
        truth = df.set_index("DAWBA_ID")["impact_score_self"]
        np.testing.assert_allclose(
            output_df["true"].to_numpy(),
            truth.loc[output_df["DAWBA_ID"]].to_numpy(),
        )

    def test_output_rows_are_the_test_fold(self, call, make_df):
        df = make_df(n=24, n_splits=3)
        _, output_df, _ = call(df=df, split_id=1)
        assert set(output_df["DAWBA_ID"]) == set(
            df.loc[df["split_id"] == 1, "DAWBA_ID"]
        )

    def test_split_id_tagged_on_outputs(self, call):
        perm_id_df, output_df, coef_df = call(split_id=2)
        assert (output_df["split_id"] == 2).all()
        assert (coef_df["split_id"] == 2).all()
        assert list(perm_id_df.columns) == ["perm_id_split_2"]

    def test_coef_stages_match_the_layer_config(self, call):
        _, _, coef3 = call(num_layers=3)
        assert set(coef3["stage"]) == {"stage1", "stage2", "stage3"}

        _, _, coef_s = call(num_layers=2, two_layer_base="S")
        assert set(coef_s["stage"]) == {"stage1", "stage3"}

        _, _, coef_q = call(num_layers=2, two_layer_base="Q")
        assert set(coef_q["stage"]) == {"stage2", "stage3"}


######## get_saliency_map ########
class TestGetSaliencyMap:
    @pytest.fixture
    def inputs(self, model_config, make_df):
        """coef_df shaped the way get_saliency_map reads it: indexed by stage_feat
        with a coef_<impact_type> column, i.e. post-rename."""

        def _make(impact_type="self", coef_col=None):
            cfg = model_config(impact_type)

            class StubModel:
                quest_cols = list(QUEST[impact_type])

            rows = {"stage1_sum_score": 0.5, "stage1_intercept": 99.0}
            for i in range(1, len(QUEST[impact_type]) + 1):
                rows[f"stage2__q{i}"] = 0.1 * i
            for i, c in enumerate(PROB_OUTCOMES):
                rows[f"stage3_{c}"] = -0.2 * (i + 1)

            coef_df = pd.DataFrame({coef_col or f"coef_{impact_type}": pd.Series(rows)})
            coef_df.index.name = "stage_feat"
            return cfg, StubModel(), make_df(n=10), coef_df.sort_index()

        return _make

    @pytest.mark.parametrize("impact_type", ["self", "parent"])
    def test_all_features_present_and_intercepts_excluded(self, inputs, impact_type):
        """Intercepts are not per-participant contributions, so they carry no
        saliency. The questionnaire prefix must be reattached from the leading
        underscore that build_model left behind."""
        cfg, model, df, coef_df = inputs(impact_type)
        summary = get_saliency_map(cfg, model, df, coef_df)
        assert set(summary.columns) == {
            "sum_score",
            *QUEST[impact_type],
            *PROB_OUTCOMES,
        }

    def test_values_are_coef_times_observation(self, inputs):
        cfg, model, df, coef_df = inputs()
        summary = get_saliency_map(cfg, model, df, coef_df)
        expected = (df["t1"] * coef_df.loc["stage3_t1", "coef_self"]).mean()
        assert summary.loc["mean", "t1"] == pytest.approx(expected)
