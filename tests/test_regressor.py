# test_regressor.py

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, Ridge

import src.Regressor as reg_mod


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


######## _build_X_blocks ########
class TestBuildXBlocks:
    def test_non_dataframe_raises(self, make_resid):
        m = make_resid(num_layers=3)
        with pytest.raises(TypeError, match="DataFrame"):
            m._build_X_blocks(np.zeros((4, 3)))

    def test_block_shapes_and_sum_relationship(self, make_resid, make_xy,
                                                quest_cols, prob_outcomes):
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

    def test_outcome_sub_list_subsets_text_block(self, make_resid, make_xy):
        m = make_resid(num_layers=3, outcome_sub_list=["t1", "t3"])
        df, _ = make_xy(n=6)
        _, _, prob_vec = m._build_X_blocks(df)
        assert prob_vec.shape == (6, 2)
        np.testing.assert_allclose(prob_vec, df[["t1", "t3"]].to_numpy())

    def test_outcome_sub_list_not_a_subset_raises(self, make_resid, make_xy):
        m = make_resid(num_layers=3, outcome_sub_list=["t1", "NOT_AN_OUTCOME"])
        df, _ = make_xy(n=6)
        with pytest.raises(AssertionError):
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

######## get_coef_df ########
class TestGetCoefDf:
    def test_one_layer_reports_only_stage1(self, make_independent, make_xy):
        m = make_independent()
        df, y = make_xy()
        m.fit(df, y)
        assert set(m.get_coef_df()["stage"]) == {"stage1"}

    @pytest.mark.parametrize("base", ["S", "Q"])
    def test_two_layer_reports_stage1_and_stage3(self, make_resid, make_xy, base):
        # TODO(bug A): get_coef_df currently returns {stage1, stage3} for BOTH
        # bases. That is correct for base="S" (stage2_ is stage1_), but for
        # base="Q" stage2_ is an independently fit questionnaire model whose
        # coefficients are silently dropped. This pins CURRENT behavior.
        # Intended once fixed:
        #   1-layer -> the single fitted stage (S | Q | T)
        #   2-layer -> {stage1, stage3} (S base)  OR  {stage2, stage3} (Q base)
        #   3-layer -> {stage1, stage2, stage3}
        m = make_resid(num_layers=2, two_layer_base=base)
        df, y = make_xy()
        m.fit(df, y)
        assert set(m.get_coef_df()["stage"]) == {"stage1", "stage3"}

    def test_three_layer_reports_all_stages(self, make_resid, make_xy):
        m = make_resid(num_layers=3)
        df, y = make_xy()
        m.fit(df, y)
        assert set(m.get_coef_df()["stage"]) == {"stage1", "stage2", "stage3"}

    def test_feature_names_match_their_blocks(self, make_resid, make_xy,
                                              quest_cols, prob_outcomes):
        m = make_resid(num_layers=3)
        df, y = make_xy()
        m.fit(df, y)
        cdf = m.get_coef_df()

        def feats(stage):
            return list(cdf[cdf["stage"] == stage]["feature"])

        assert feats("stage1") == ["intercept", "sum_score"]
        assert feats("stage2") == ["intercept", *quest_cols]
        assert feats("stage3") == ["intercept", *prob_outcomes]

######## predict_all additive identities ########
class TestResidualPredictIdentities:
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
        np.testing.assert_allclose(
            p["pred_stage2_total"], p["pred_stage2_increment"]
        )

    def test_predict_returns_stage3_total(self, make_resid, make_xy):
        m = make_resid(num_layers=3)
        df, y = make_xy()
        m.fit(df, y)
        np.testing.assert_allclose(
            m.predict(df), m.predict_all(df)["pred_stage3_total"].to_numpy()
        )

######## Residual cascade: signal routes to the stage that owns the feature block ########
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
        assert np.std(p["pred_stage3_increment"]) > 0.1


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

    def test_two_layer_Q_stage2_uses_alpha3_not_alpha2(self, make_resid, make_xy):
        m = make_resid(num_layers=2, two_layer_base="Q", alpha2=2.0, alpha3=5.0)
        df, y = make_xy()
        m.fit(df, y)
        # the "Q" branch deliberately fits stage2 with alpha3
        assert isinstance(m.stage2_, Ridge) and m.stage2_.alpha == 5.0
        assert isinstance(m.stage3_, Ridge) and m.stage3_.alpha == 5.0

    def test_independent_uses_alpha3_for_both_ridge_stages(self, make_independent,
                                                          make_xy):
        m = make_independent(alpha3=5.0)
        df, y = make_xy()
        m.fit(df, y)
        assert isinstance(m.stage1_, LinearRegression)
        assert isinstance(m.stage2_, Ridge) and m.stage2_.alpha == 5.0
        assert isinstance(m.stage3_, Ridge) and m.stage3_.alpha == 5.0

######## shuffle_vals — the permutation that builds the significance-testing null ########
class TestShuffleVals:
    def test_text_block_follows_permuted_ids(self, make_resid, monkeypatch):
        m = make_resid(num_layers=3)
        ids = np.array([30, 10, 20])  # deliberately out of sorted order
        # each row's text values encode its DAWBA_ID, so we can trace which
        # participant's text ends up in each output row
        col = np.array([[30.0, 30.5], [10.0, 10.5], [20.0, 20.5]])
        X = pd.DataFrame({"DAWBA_ID": ids})

        monkeypatch.setattr(reg_mod, "get_perm_seed", lambda perm_dir: 123)
        # canonical (sorted) order is [10, 20, 30]; perm maps output pos -> ID
        perm = np.array([20, 30, 10])
        monkeypatch.setattr(
            reg_mod, "create_perms", lambda perm_seed, df_to_permute: perm
        )

        out = m.shuffle_vals(X=X, col_to_shuffle_np=col, perm_dir="ignored")

        expected = np.array([[20.0, 20.5], [30.0, 30.5], [10.0, 10.5]])
        np.testing.assert_allclose(out, expected)
        np.testing.assert_array_equal(m.perm_ids, perm)
        # the null is only valid if this is a genuine reordering, not identity
        assert not np.allclose(out, col)

    def test_create_perms_receives_seed_and_id_sorted_frame(self, make_resid,
                                                           monkeypatch):
        m = make_resid(num_layers=3)
        X = pd.DataFrame({"DAWBA_ID": np.array([30, 10, 20])})
        col = np.array([[1.0], [2.0], [3.0]])

        captured = {}
        monkeypatch.setattr(reg_mod, "get_perm_seed", lambda perm_dir: 777)

        def fake_create_perms(perm_seed, df_to_permute):
            captured["seed"] = perm_seed
            captured["ids"] = df_to_permute["DAWBA_ID"].tolist()
            return np.array([10, 20, 30])

        monkeypatch.setattr(reg_mod, "create_perms", fake_create_perms)
        m.shuffle_vals(X=X, col_to_shuffle_np=col, perm_dir="ignored")

        assert captured["seed"] == 777
        # the frame handed to create_perms is sorted by DAWBA_ID
        assert captured["ids"] == [10, 20, 30]

######## fit() shuffle wiring (shared across both regressor classes) ########
class TestFitShuffleWiring:
    @pytest.mark.parametrize("kind", ["resid", "indep"])
    def test_shuffle_none_sets_perm_ids_to_dawba_ids(self, kind, make_resid,
                                                     make_independent, make_xy):
        m = make_resid(num_layers=3) if kind == "resid" else make_independent()
        df, y = make_xy(n=8)
        m.fit(df, y, shuffle=None)
        np.testing.assert_array_equal(m.perm_ids, df["DAWBA_ID"].to_numpy())

    @pytest.mark.parametrize("kind", ["resid", "indep"])
    def test_unrecognized_shuffle_raises(self, kind, make_resid, make_independent,
                                        make_xy):
        m = make_resid(num_layers=3) if kind == "resid" else make_independent()
        df, y = make_xy(n=8)
        with pytest.raises(ValueError, match="shuffle"):
            m.fit(df, y, shuffle="rows")

    def test_shuffle_text_invokes_shuffle_vals_with_perm_dir(self, make_resid,
                                                            make_xy, monkeypatch):
        m = make_resid(num_layers=3)
        df, y = make_xy(n=8)
        seen = {}

        def fake_shuffle(X, col_to_shuffle_np, perm_dir):
            seen["called"] = True
            seen["perm_dir"] = perm_dir
            m.perm_ids = X["DAWBA_ID"].to_numpy()  # mimic the real side effect
            return col_to_shuffle_np  # pass through so the rest of fit proceeds

        monkeypatch.setattr(m, "shuffle_vals", fake_shuffle)
        m.fit(df, y, shuffle="text", perm_dir="some/dir")

        assert seen["called"] is True
        assert seen["perm_dir"] == "some/dir"

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

    def test_predict_returns_stage3_total(self, make_independent, make_xy):
        m = make_independent()
        df, y = make_xy()
        m.fit(df, y)
        np.testing.assert_allclose(
            m.predict(df), m.predict_all(df)["pred_stage3_total"].to_numpy()
        )

######## Latent edge case: bad two_layer_base survives __init__ but trips fit() ########
class TestInvalidTwoLayerBaseAtFit:
    def test_unknown_base_passes_init_but_fails_in_fit(self, make_resid, make_xy):
        # __init__ only checks None-ness, so a bogus base survives construction
        # for a 2-layer model; the S/Q membership check fires inside fit().
        m = make_resid(num_layers=2, two_layer_base="X")
        df, y = make_xy()
        with pytest.raises(ValueError, match="two_layer_base"):
            m.fit(df, y)