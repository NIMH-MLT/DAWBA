# test_regressor.py
# Run w/ uv run pytest tests/test_regressor.py -v
# Written in collaboration w/ Claude Opus 5
#
# NOTE: shuffle_vals / permutation-null coverage is deliberately absent -- to be
# added separately. The two fit()-level shuffle tests below only guard the
# default (shuffle=None) path and the argument validation.

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge

import src.Regressor as reg_mod
from src.Regressor import BaseRegressor, IndependentRegressor, ResidualRegressor

QUEST_COLS = ["q1", "q2", "q3"]
PROB_OUTCOMES = ["t1", "t2", "t3", "t4"]


######## Fixtures ########
@pytest.fixture(autouse=True)
def stub_prob_col_outcomes(monkeypatch):
    """get_prob_col_outcomes reads the real outcome registry; pin it so the
    feature block is a known width in every test."""
    monkeypatch.setattr(
        reg_mod,
        "get_prob_col_outcomes",
        lambda agg_type, desc_seg_type: ("prob_col", list(PROB_OUTCOMES)),
    )


@pytest.fixture
def quest_cols():
    return list(QUEST_COLS)


@pytest.fixture
def prob_outcomes():
    return list(PROB_OUTCOMES)


@pytest.fixture
def base_kwargs():
    return dict(
        impact_type="self",
        desc_seg_type="original",
        quest_cols=list(QUEST_COLS),
        agg_type="avg",
        alpha2=1.0,
        alpha3=1.0,
        seed=0,
        linear_type="ridge",
    )


@pytest.fixture
def make_resid(base_kwargs):
    def _make(num_layers=3, two_layer_base=None, **overrides):
        kwargs = {**base_kwargs, "num_layers": num_layers, **overrides}
        return ResidualRegressor(two_layer_base=two_layer_base, **kwargs)

    return _make


@pytest.fixture
def make_independent(base_kwargs):
    def _make(num_layers=1, **overrides):
        kwargs = {**base_kwargs, "num_layers": num_layers, **overrides}
        return IndependentRegressor(**kwargs)

    return _make


@pytest.fixture
def make_xy():
    def _make(n=12, y_from="mixed", quest_constant=False, seed=0):
        rng = np.random.default_rng(seed)
        data = {"DAWBA_ID": np.arange(1, n + 1)}
        for c in QUEST_COLS:
            data[c] = np.full(n, 2.0) if quest_constant else rng.integers(0, 3, n) * 1.0
        for c in PROB_OUTCOMES:
            data[c] = rng.random(n)
        df = pd.DataFrame(data)

        # Optionally make y linear combination of sum, quest, text, or a mix
        if y_from == "sum":
            y = 3.0 + 2.0 * df[QUEST_COLS].sum(axis=1)
        elif y_from == "text":
            y = 5.0 * df["t1"] - 3.0 * df["t2"]
        elif y_from == "quest":
            y = 1.0 * df["q1"] - 2.0 * df["q2"]
        else:
            y = df[QUEST_COLS].sum(axis=1) + 2.0 * df["t1"] + rng.normal(0, 0.1, n)
        return df, pd.Series(np.asarray(y, dtype=float), name="impact_score")

    return _make


@pytest.fixture
def stub_linear():
    def _make(coef, intercept):
        class _Stub:
            pass

        s = _Stub()
        s.coef_ = np.asarray(coef, dtype=float)
        s.intercept_ = np.asarray(intercept, dtype=float)
        return s

    return _make


######## Construction invariants ########
class TestResidualRegressorConstruction:
    def test_num_layers_one_rejected(self, make_resid):
        with pytest.raises(ValueError, match="Num layers"):
            make_resid(num_layers=1)

    def test_two_layers_requires_base(self, make_resid):
        with pytest.raises(ValueError, match="two_layer_base"):
            make_resid(num_layers=2, two_layer_base=None)

    @pytest.mark.parametrize("base", ["S", "Q"])
    def test_two_layers_with_base_ok(self, make_resid, base):
        m = make_resid(num_layers=2, two_layer_base=base)
        assert m.num_layers == 2
        assert m.two_layer_base == base

    def test_three_layers_no_base_ok(self, make_resid):
        m = make_resid(num_layers=3, two_layer_base=None)
        assert m.num_layers == 3
        assert m.two_layer_base is None

    @pytest.mark.parametrize("base", ["S", "Q"])
    def test_three_layers_with_base_rejected(self, make_resid, base):
        with pytest.raises(ValueError, match="two_layer_base"):
            make_resid(num_layers=3, two_layer_base=base)


######## _make_model ########
class TestMakeModel:
    def test_ridge_receives_alpha_and_seed(self, make_resid):
        m = make_resid(num_layers=3)
        model = m._make_model(alpha=3.0, model_type="ridge")
        assert isinstance(model, Ridge)
        assert model.alpha == 3.0
        assert model.random_state == m.seed

    def test_regular_ignores_alpha(self, make_resid):
        m = make_resid(num_layers=3)
        model = m._make_model(alpha=99.0, model_type="regular")
        assert isinstance(model, LinearRegression)

    def test_unknown_model_type_raises(self, make_resid):
        m = make_resid(num_layers=3)
        with pytest.raises(ValueError, match="Unrecognized model_type"):
            m._make_model(alpha=1.0, model_type="lasso")


######## _build_X_blocks ########
class TestBuildXBlocks:
    def test_non_dataframe_raises(self, make_resid):
        m = make_resid(num_layers=3)
        with pytest.raises(TypeError, match="DataFrame"):
            m._build_X_blocks(np.zeros((4, 3)))

    def test_block_shapes_and_sum_relationship(
        self, make_resid, make_xy, quest_cols, prob_outcomes
    ):
        m = make_resid(num_layers=3)
        df, _ = make_xy(n=10)
        quest_sum, quest_np, prob_vec = m._build_X_blocks(df)

        assert quest_sum.shape == (10, 1)
        assert quest_np.shape == (10, len(quest_cols))
        assert prob_vec.shape == (10, len(prob_outcomes))

        np.testing.assert_allclose(
            quest_sum[:, 0], df[quest_cols].sum(axis=1).to_numpy()
        )
        np.testing.assert_allclose(prob_vec, df[prob_outcomes].to_numpy())

    def test_column_order_follows_registry_not_frame(self, make_resid, make_xy):
        """Block column order comes from get_prob_col_outcomes, so a reordered
        input frame must not reorder the text block."""
        m = make_resid(num_layers=3)
        df, _ = make_xy(n=6)
        shuffled_cols = df[["DAWBA_ID"] + QUEST_COLS + PROB_OUTCOMES[::-1]]
        _, _, prob_vec = m._build_X_blocks(shuffled_cols)
        np.testing.assert_allclose(prob_vec, df[PROB_OUTCOMES].to_numpy())

    def test_outcome_sub_list_subsets_text_block(self, make_resid, make_xy):
        m = make_resid(num_layers=3, outcome_sub_list=["t1", "t3"])
        df, _ = make_xy(n=6)
        _, _, prob_vec = m._build_X_blocks(df)
        assert prob_vec.shape == (6, 2)
        np.testing.assert_allclose(prob_vec, df[["t1", "t3"]].to_numpy())

    def test_outcome_sub_list_preserves_caller_order(self, make_resid, make_xy):
        """Unlike the full-registry path, the subset is used verbatim -- so the
        caller's ordering, not the registry's, determines column order."""
        m = make_resid(num_layers=3, outcome_sub_list=["t3", "t1"])
        df, _ = make_xy(n=6)
        _, _, prob_vec = m._build_X_blocks(df)
        np.testing.assert_allclose(prob_vec, df[["t3", "t1"]].to_numpy())

    def test_missing_quest_col_raises(self, make_resid, make_xy):
        m = make_resid(num_layers=3)
        df, _ = make_xy(n=6)
        with pytest.raises(KeyError):
            m._build_X_blocks(df.drop(columns=["q2"]))

    def test_non_numeric_quest_col_raises(self, make_resid, make_xy):
        m = make_resid(num_layers=3)
        df, _ = make_xy(n=6)
        df["q1"] = "not_a_number"
        with pytest.raises(ValueError):
            m._build_X_blocks(df)


######## _extract_coefs ########
class TestExtractCoefs:
    def test_intercept_first_then_features_in_order(self, make_resid, stub_linear):
        m = make_resid(num_layers=3)
        stub = stub_linear(coef=[0.1, -0.2, 0.3], intercept=4.0)
        out = m._extract_coefs(stub, ["a", "b", "c"], "stageX")

        assert list(out["stage"]) == ["stageX"] * 4
        assert list(out["feature"]) == ["intercept", "a", "b", "c"]
        np.testing.assert_allclose(out["coef"].to_numpy(), [4.0, 0.1, -0.2, 0.3])

    def test_array_shaped_coef_and_intercept_flattened(self, make_resid, stub_linear):
        m = make_resid(num_layers=3)
        stub = stub_linear(coef=np.array([[1.0, 2.0]]), intercept=np.array([7.0]))
        out = m._extract_coefs(stub, ["a", "b"], "s")
        assert out.loc[0, "feature"] == "intercept"
        assert out.loc[0, "coef"] == 7.0
        np.testing.assert_allclose(out["coef"].to_numpy()[1:], [1.0, 2.0])


######## _reported_stages / get_coef_df ########
class TestReportedStages:
    def test_base_class_is_abstract(self, base_kwargs):
        m = BaseRegressor(num_layers=3, **base_kwargs)
        with pytest.raises(NotImplementedError):
            m._reported_stages()

    def test_three_layer_reports_all_stages(self, make_resid, make_xy):
        m = make_resid(num_layers=3)
        df, y = make_xy()
        m.fit(df, y)
        assert set(m.get_coef_df()["stage"]) == {"stage1", "stage2", "stage3"}

    def test_three_layer_stage_order_is_cascade_order(self, make_resid, make_xy):
        m = make_resid(num_layers=3)
        df, y = make_xy()
        m.fit(df, y)
        seen = list(dict.fromkeys(m.get_coef_df()["stage"]))
        assert seen == ["stage1", "stage2", "stage3"]

    @pytest.mark.parametrize(
        "base,expected",
        [("S", {"stage1", "stage3"}), ("Q", {"stage2", "stage3"})],
    )
    def test_two_layer_reports_contributing_stages(
        self, make_resid, make_xy, base, expected
    ):
        """A stage is reported iff predict_all adds its prediction. For base="S"
        stage2_ IS stage1_, so reporting both would double-count one set of
        coefficients; for base="Q" stage1_ is fit but zeroed in predict_all."""
        m = make_resid(num_layers=2, two_layer_base=base)
        df, y = make_xy()
        m.fit(df, y)
        assert set(m.get_coef_df()["stage"]) == expected

    def test_Q_base_omits_the_zeroed_sum_stage(self, make_resid, make_xy):
        m = make_resid(num_layers=2, two_layer_base="Q")
        df, y = make_xy()
        m.fit(df, y)
        assert "stage1" not in set(m.get_coef_df()["stage"])
        np.testing.assert_allclose(m.predict_all(df)["pred_stage1"], 0.0)

    def test_S_base_reports_the_shared_model_once(self, make_resid, make_xy):
        m = make_resid(num_layers=2, two_layer_base="S")
        df, y = make_xy()
        m.fit(df, y)
        cdf = m.get_coef_df()
        assert m.stage2_ is m.stage1_
        assert len(cdf[cdf["feature"] == "sum_score"]) == 1

    def test_independent_reports_all_three_stages(self, make_independent, make_xy):
        """fit() builds S, Q and T regardless of num_layers, and each is a
        standalone model on raw y -- so all three are reported."""
        m = make_independent(num_layers=1)
        df, y = make_xy()
        m.fit(df, y)
        assert set(m.get_coef_df()["stage"]) == {"stage1", "stage2", "stage3"}

    def test_feature_names_match_their_blocks(
        self, make_resid, make_xy, quest_cols, prob_outcomes
    ):
        m = make_resid(num_layers=3)
        df, y = make_xy()
        m.fit(df, y)
        cdf = m.get_coef_df()

        def feats(stage):
            return list(cdf[cdf["stage"] == stage]["feature"])

        assert feats("stage1") == ["intercept", "sum_score"]
        assert feats("stage2") == ["intercept", *quest_cols]
        assert feats("stage3") == ["intercept", *prob_outcomes]

    def test_before_fit_raises(self, make_resid):
        m = make_resid(num_layers=3)
        with pytest.raises(AttributeError):
            m.get_coef_df()


class TestOutcomeSubListCoefLabelling:
    @pytest.mark.parametrize("kind", ["resid", "indep"])
    def test_stage3_names_follow_the_subset(
        self, kind, make_resid, make_independent, make_xy
    ):
        sub = ["t1", "t3"]
        m = (
            make_resid(num_layers=3, outcome_sub_list=sub)
            if kind == "resid"
            else make_independent(outcome_sub_list=sub)
        )
        df, y = make_xy(n=12)
        m.fit(df, y)
        cdf = m.get_coef_df()
        stage3 = list(cdf[cdf["stage"] == "stage3"]["feature"])
        assert stage3 == ["intercept", *sub]

    def test_stage3_coefs_align_with_the_fitted_model(self, make_resid, make_xy):
        m = make_resid(num_layers=3, outcome_sub_list=["t1", "t3"])
        df, y = make_xy(n=12)
        m.fit(df, y)
        stage3 = m.get_coef_df().query("stage == 'stage3'")
        np.testing.assert_allclose(stage3["coef"].to_numpy()[1:], m.stage3_.coef_)


######## predict_all additive identities ########
class TestResidualPredictIdentities:
    def test_predict_all_column_set(self, make_resid, make_xy):
        m = make_resid(num_layers=3)
        df, y = make_xy()
        m.fit(df, y)
        assert list(m.predict_all(df).columns) == [
            "pred_stage1",
            "pred_stage2_increment",
            "pred_stage2_total",
            "pred_stage3_increment",
            "pred_stage3_total",
        ]

    def test_three_layer_increments_sum_to_totals(self, make_resid, make_xy):
        m = make_resid(num_layers=3)
        df, y = make_xy(n=14)
        m.fit(df, y)
        p = m.predict_all(df)
        np.testing.assert_allclose(
            p["pred_stage2_total"], p["pred_stage1"] + p["pred_stage2_increment"]
        )
        np.testing.assert_allclose(
            p["pred_stage3_total"],
            p["pred_stage2_total"] + p["pred_stage3_increment"],
        )

    def test_two_layer_S_zeroes_stage2_increment(self, make_resid, make_xy):
        m = make_resid(num_layers=2, two_layer_base="S")
        df, y = make_xy(n=14)
        m.fit(df, y)
        p = m.predict_all(df)
        np.testing.assert_allclose(p["pred_stage2_increment"], 0.0)
        np.testing.assert_allclose(p["pred_stage2_total"], p["pred_stage1"])
        np.testing.assert_allclose(
            p["pred_stage3_total"], p["pred_stage1"] + p["pred_stage3_increment"]
        )

    def test_two_layer_Q_zeroes_stage1(self, make_resid, make_xy):
        m = make_resid(num_layers=2, two_layer_base="Q")
        df, y = make_xy(n=14)
        m.fit(df, y)
        p = m.predict_all(df)
        np.testing.assert_allclose(p["pred_stage1"], 0.0)
        np.testing.assert_allclose(p["pred_stage2_total"], p["pred_stage2_increment"])

    def test_predict_returns_stage3_total(self, make_resid, make_xy):
        m = make_resid(num_layers=3)
        df, y = make_xy()
        m.fit(df, y)
        np.testing.assert_allclose(
            m.predict(df), m.predict_all(df)["pred_stage3_total"].to_numpy()
        )

    def test_train_time_predictions_match_predict_all(self, make_resid, make_xy):
        """pred1_train_/pred2_train_/pred3_train_ are cached during fit and used to
        form the residual targets; they must agree with predict_all on the same X."""
        m = make_resid(num_layers=3)
        df, y = make_xy(n=14)
        m.fit(df, y)
        p = m.predict_all(df)
        np.testing.assert_allclose(m.pred1_train_, p["pred_stage1"])
        np.testing.assert_allclose(m.pred2_train_, p["pred_stage2_increment"])
        np.testing.assert_allclose(m.pred3_train_, p["pred_stage3_increment"])

    def test_predict_row_order_follows_input(self, make_resid, make_xy):
        """predict_all returns a fresh RangeIndex, so callers align by position --
        pin that a reordered X produces correspondingly reordered predictions."""
        m = make_resid(num_layers=3)
        df, y = make_xy(n=10)
        m.fit(df, y)
        p = m.predict_all(df)
        rev = df.iloc[::-1].reset_index(drop=True)
        p_rev = m.predict_all(rev)
        np.testing.assert_allclose(
            p_rev["pred_stage3_total"].to_numpy(),
            p["pred_stage3_total"].to_numpy()[::-1],
        )


######## Residual cascade: signal routes to the stage owning the feature block ########
class TestResidualCascade:
    def test_signal_in_sum_collapses_downstream_increments(self, make_resid, make_xy):
        m = make_resid(num_layers=3, alpha2=1.0, alpha3=1.0)
        df, y = make_xy(n=16, y_from="sum")
        m.fit(df, y)
        p = m.predict_all(df)
        np.testing.assert_allclose(p["pred_stage1"], y.to_numpy(), atol=1e-6)
        np.testing.assert_allclose(p["pred_stage2_increment"], 0.0, atol=1e-6)
        np.testing.assert_allclose(p["pred_stage3_increment"], 0.0, atol=1e-6)

    def test_signal_in_text_surfaces_in_stage3_increment(self, make_resid, make_xy):
        # constant questionnaire -> no sum/quest signal; y depends only on text
        m = make_resid(num_layers=3, alpha2=1e-3, alpha3=1e-3)
        df, y = make_xy(n=16, y_from="text", quest_constant=True)
        m.fit(df, y)
        p = m.predict_all(df)
        np.testing.assert_allclose(p["pred_stage2_increment"], 0.0, atol=1e-6)
        assert np.std(p["pred_stage3_increment"]) > 0.1  # ensure no collapse to 0

    def test_two_layer_Q_residualizes_on_quest_not_sum(self, make_resid, make_xy):
        """Base="Q" must regress the text block on residuals from the *granular*
        questionnaire fit, not from the sum fit."""
        m = make_resid(num_layers=2, two_layer_base="Q", alpha2=1e-6, alpha3=1e-3)
        df, y = make_xy(n=20, y_from="quest")
        m.fit(df, y)
        p = m.predict_all(df)
        np.testing.assert_allclose(p["pred_stage2_total"], y.to_numpy(), atol=1e-4)
        np.testing.assert_allclose(p["pred_stage3_increment"], 0.0, atol=1e-4)


######## Alpha routing — which penalty reaches which stage ########
class TestAlphaRouting:
    def test_three_layer_alpha_assignment(self, make_resid, make_xy):
        m = make_resid(num_layers=3, alpha2=2.0, alpha3=5.0)
        df, y = make_xy()
        m.fit(df, y)
        assert isinstance(m.stage1_, LinearRegression)
        assert isinstance(m.stage2_, Ridge) and m.stage2_.alpha == 2.0
        assert isinstance(m.stage3_, Ridge) and m.stage3_.alpha == 5.0

    def test_two_layer_S_reuses_stage1_as_stage2(self, make_resid, make_xy):
        m = make_resid(num_layers=2, two_layer_base="S", alpha3=5.0)
        df, y = make_xy()
        m.fit(df, y)
        assert m.stage2_ is m.stage1_
        assert isinstance(m.stage3_, Ridge) and m.stage3_.alpha == 5.0

    def test_two_layer_Q_stage2_uses_alpha2(self, make_resid, make_xy):
        """The Q branch takes alpha2, consistent with the 3-layer questionnaire
        stage."""
        m = make_resid(num_layers=2, two_layer_base="Q", alpha2=2.0, alpha3=5.0)
        df, y = make_xy()
        m.fit(df, y)
        assert isinstance(m.stage2_, Ridge) and m.stage2_.alpha == 2.0
        assert isinstance(m.stage3_, Ridge) and m.stage3_.alpha == 5.0

    def test_independent_uses_alpha3_for_both_ridge_stages(
        self, make_independent, make_xy
    ):
        # documented in the fit() docstring: alpha3 covers quest AND text here,
        # so alpha2 is inert for this class
        m = make_independent(alpha2=2.0, alpha3=5.0)
        df, y = make_xy()
        m.fit(df, y)
        assert isinstance(m.stage1_, LinearRegression)
        assert isinstance(m.stage2_, Ridge) and m.stage2_.alpha == 5.0
        assert isinstance(m.stage3_, Ridge) and m.stage3_.alpha == 5.0

    def test_linear_type_regular_bypasses_penalties(self, make_resid, make_xy):
        m = make_resid(num_layers=3, linear_type="regular", alpha2=99.0, alpha3=99.0)
        df, y = make_xy()
        m.fit(df, y)
        assert all(
            isinstance(s, LinearRegression) for s in (m.stage1_, m.stage2_, m.stage3_)
        )

    def test_unknown_linear_type_raises_at_fit(self, make_resid, make_xy):
        m = make_resid(num_layers=3, linear_type="elastic")
        df, y = make_xy()
        with pytest.raises(ValueError, match="Unrecognized model_type"):
            m.fit(df, y)


######## IndependentRegressor specifics ########
class TestIndependentRegressor:
    def test_predict_all_columns(self, make_independent, make_xy):
        m = make_independent()
        df, y = make_xy()
        m.fit(df, y)
        p = m.predict_all(df)
        assert list(p.columns) == [
            "pred_stage1",
            "pred_stage2_total",
            "pred_stage3_total",
        ]

    def test_stages_regress_raw_target_not_residuals(self, make_independent, make_xy):
        # stage1 is OLS on the sum; when y is affine in the sum it recovers y
        # exactly (unlike the residual cascade, where increments carry it).
        m = make_independent()
        df, y = make_xy(n=16, y_from="sum")
        m.fit(df, y)
        p = m.predict_all(df)
        np.testing.assert_allclose(p["pred_stage1"], y.to_numpy(), atol=1e-6)

    def test_totals_are_not_cumulative(self, make_independent, make_xy):
        """Despite the _total suffix, each stage is a standalone model -- so the
        columns must NOT satisfy the residual-cascade additive identity."""
        m = make_independent()
        df, y = make_xy(n=16)
        m.fit(df, y)
        p = m.predict_all(df)
        assert not np.allclose(
            p["pred_stage3_total"], p["pred_stage1"] + p["pred_stage2_total"]
        )

    def test_all_three_stages_fit_regardless_of_num_layers(
        self, make_independent, make_xy
    ):
        m = make_independent(num_layers=1)
        df, y = make_xy()
        m.fit(df, y)
        assert all(hasattr(m, f"stage{i}_") for i in (1, 2, 3))

    def test_predict_returns_stage3_total(self, make_independent, make_xy):
        # NOTE: this is the text-only model for EVERY num_layers, including the
        # 1-layer S and Q configurations. Anything routing through the sklearn
        # predict() interface therefore scores T no matter which single-block
        # config was requested; callers wanting S or Q must read the
        # corresponding column out of predict_all themselves.
        m = make_independent(num_layers=1)
        df, y = make_xy()
        m.fit(df, y)
        np.testing.assert_allclose(
            m.predict(df), m.predict_all(df)["pred_stage3_total"].to_numpy()
        )
