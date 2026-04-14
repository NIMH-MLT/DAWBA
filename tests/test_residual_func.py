# test_residual_func.py
# This script was partially written by an LLM
from src.config import SEED
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression

from src.residual_func import ResidualRegressor, get_lr_output
from .conftest import QUEST_COLS


def make_regressor(**overrides):
    """Helper function that returns a ResidualRegressor. Override as needed"""
    defaults = dict(
        quest_cols=QUEST_COLS,
        agg_type="avg",
        num_layers=3,
        alpha2=0.1,
        l1_ratio2=0.5,
        alpha3=0.1,
        l1_ratio3=0.5,
        seed=SEED,
        impactme_outcome=None,
        linear_type="elastic",
    )
    defaults.update(overrides)
    return ResidualRegressor(**defaults)


class TestInit:
    """
    Test initialization of ResidualRegressor
    """

    @pytest.mark.parametrize("num_layers", [0, 1, 4])
    def test_invalid_num_layers_raises(self, num_layers):
        with pytest.raises(ValueError, match="Num layers"):
            make_regressor(num_layers=num_layers)

    @pytest.mark.parametrize("num_layers", [2, 3])
    def test_valid_num_layers_accepted(self, num_layers):
        r = make_regressor(num_layers=num_layers)
        assert r.num_layers == num_layers


class TestGetProbCol:
    # allows for run of test for each [agg_type, expected] pair
    @pytest.mark.parametrize(
        "agg_type,expected",
        [
            ("avg", "prob_vector_avg"),
            ("max", "prob_vector_max"),
            ("block", "prob_vector_block"),
            ("binary", "open_text_block_0_shot_binary"),
            ("proba-limited", "open_text_block_0_shot"),
        ],
    )
    def test_known_agg_types(self, agg_type, expected):
        r = make_regressor(agg_type=agg_type)
        assert r._get_prob_col() == expected

    def test_unknown_agg_type_raises(self):
        r = make_regressor(agg_type="unknown")
        with pytest.raises(ValueError, match="Unrecognized agg_type"):
            r._get_prob_col()


class TestMakeModel:
    def setup_method(self):
        self.r = make_regressor()

    @pytest.mark.parametrize(
        "model_type,cls",
        [
            ("elastic", ElasticNet),
            ("ridge", Ridge),
            ("regular", LinearRegression),
        ],
    )
    def test_returns_correct_estimator(self, model_type, cls):
        m = self.r._make_model(alpha=0.1, l1_ratio=0.5, model_type=model_type)
        assert isinstance(m, cls)

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unrecognized model_type"):
            self.r._make_model(alpha=0.1, l1_ratio=0.5, model_type="lasso")


class TestFitPredict:
    @pytest.mark.parametrize(
        "agg_type, num_layers, impactme_outcome, linear_type",
        [
            ("avg", 2, None, "elastic"),
            ("avg", 3, None, "elastic"),
            ("avg", 2, None, "ridge"),
            ("avg", 2, "a1", "elastic"),
            ("avg", 2, "c2", "ridge"),
            ("block", 2, None, "elastic"),
            ("max", 2, None, "elastic"),
            ("binary", 2, "a1", "elastic"),
            ("proba-limited", 2, "a1", "ridge"),
            ("proba-limited", 2, "a1", "elastic"),
        ],
    )
    def test_fit_runs_without_error(
        self, base_df, y, agg_type, num_layers, impactme_outcome, linear_type
    ):
        # NOTE: base_df & y call functions in conftest.py
        r = make_regressor(
            agg_type=agg_type,
            num_layers=num_layers,
            impactme_outcome=impactme_outcome,
            linear_type=linear_type,
        )
        r.fit(base_df, y)

    @pytest.mark.parametrize(
        "impactme_outcome, linear_type, num_layers, model1, model2, model3",
        [
            (None, "elastic", 2, LinearRegression, LinearRegression, ElasticNet),
            (None, "ridge", 2, LinearRegression, LinearRegression, Ridge),
            (None, "elastic", 3, LinearRegression, ElasticNet, ElasticNet),
            (None, "ridge", 3, LinearRegression, Ridge, Ridge),
            ("a2", "elastic", 3, LinearRegression, ElasticNet, LinearRegression),
            ("a2", "ridge", 2, LinearRegression, LinearRegression, LinearRegression),
        ],
    )
    def test_fit_creates_correct_models_base(
        self,
        base_df,
        y,
        impactme_outcome,
        linear_type,
        num_layers,
        model1,
        model2,
        model3,
    ):
        r = make_regressor(
            impactme_outcome=impactme_outcome,
            linear_type=linear_type,
            num_layers=num_layers,
        )
        r.fit(base_df, y)
        assert isinstance(
            getattr(r, "stage1_"), model1
        ), f"Model1: {getattr(r, "stage1_")}"
        assert isinstance(
            getattr(r, "stage2_"), model2
        ), f"Model2: {getattr(r, "stage2_")}"
        assert isinstance(
            getattr(r, "stage3_"), model3
        ), f"Model3: {getattr(r, "stage3_")}"

    @pytest.mark.parametrize("num_layers", [2, 3])
    def test_fit_creates_stage_attributes(self, base_df, y, num_layers):
        r = make_regressor(num_layers=num_layers)
        r.fit(base_df, y)
        for attr in ["stage1_", "stage2_", "stage3_"]:
            assert hasattr(r, attr), f"Missing attribute: {attr}"

    def test_two_layer_stage2_is_stage1(self, base_df, y):
        """With num_layers=2 the granular questionnaire layer is skipped."""
        r = make_regressor(num_layers=2)
        r.fit(base_df, y)
        assert r.stage2_ is r.stage1_

    def test_two_layer_stage2_increment_is_zero(self, base_df, y):
        r = make_regressor(num_layers=2)
        r.fit(base_df, y)
        out = r.predict_all(base_df)
        np.testing.assert_array_equal(out["pred_stage2_increment"].values, 0)

    def test_predict_returns_correct_length(self, base_df, y):
        r = make_regressor()
        r.fit(base_df, y)
        preds = r.predict(base_df)
        assert len(preds) == len(y)

    def test_predict_returns_numpy_array(self, base_df, y):
        r = make_regressor()
        r.fit(base_df, y)
        assert isinstance(r.predict(base_df), np.ndarray)

    def test_predict_all_returns_expected_columns(self, base_df, y):
        expected_cols = {
            "pred_stage1",
            "pred_stage2_increment",
            "pred_stage2_total",
            "pred_stage3_increment",
            "pred_stage3_total",
        }
        r = make_regressor()
        r.fit(base_df, y)
        out = r.predict_all(base_df)
        assert set(out.columns) == expected_cols

    def test_build_x_blocks_requires_dataframe(self, y):
        r = make_regressor()
        with pytest.raises(TypeError, match="pandas DataFrame"):
            r._build_X_blocks(np.zeros((len(y), 3)))

    @pytest.mark.parametrize(
        "impactme_outcome, agg_type, size3",
        [
            (None, "max", 28),
            (None, "binary", 31),
            ("a1", "max", 1),
            ("a1", "binary", 1),
        ],
    )
    def test_X_block_sizes(self, base_df, impactme_outcome, agg_type, size3):
        r = make_regressor(impactme_outcome=impactme_outcome, agg_type=agg_type)
        X1, X2, X3 = r._build_X_blocks(base_df)
        # Ensure single sum score
        assert len(X1[0]) == 1
        # Ensure individual prob estimates
        assert len(X2[0]) == len(QUEST_COLS)
        # Ensure matches length of base_df prob col
        assert len(X3[0]) == size3


class TestGetLrOutput:
    def test_creates_parquet_at_export_path(self, base_df, tmp_path):
        export_path = tmp_path / "out.parquet"
        get_lr_output(
            df=base_df,
            agg_type="avg",
            num_layers=3,
            selected_alphas=[0.1, 0.1],
            selected_l1s=[0.5, 0.5],
            split_id=0,
            impact_type="self",
            impactme_outcome=None,
            export_path=export_path,
            max_iter=100,
            seed=SEED,
            linear_type="elastic",
        )
        assert export_path.exists()

    def test_output_has_expected_columns(self, base_df, tmp_path):
        export_path = tmp_path / "out.parquet"
        get_lr_output(
            df=base_df,
            agg_type="avg",
            num_layers=3,
            selected_alphas=[0.1, 0.1],
            selected_l1s=[0.5, 0.5],
            split_id=0,
            impact_type="self",
            impactme_outcome=None,
            export_path=export_path,
            max_iter=100,
            seed=0,
            linear_type="elastic",
        )
        df_out = pd.read_parquet(export_path)
        expected = {
            "DAWBA_ID",
            "true",
            "pred_stage1",
            "pred_stage2_increment",
            "pred_stage2_total",
            "pred_stage3_increment",
            "pred_stage3_total",
        }
        assert set(df_out.columns) == expected

    def test_output_row_count_matches_test_split(self, base_df, tmp_path):
        split_id = 0
        expected_rows = (base_df["split_id"] == split_id).sum()
        export_path = tmp_path / "out.parquet"
        get_lr_output(
            df=base_df,
            agg_type="avg",
            num_layers=3,
            selected_alphas=[0.1, 0.1],
            selected_l1s=[0.5, 0.5],
            split_id=split_id,
            impact_type="self",
            impactme_outcome=None,
            export_path=export_path,
            max_iter=100,
            seed=0,
            linear_type="elastic",
        )
        df_out = pd.read_parquet(export_path)
        assert len(df_out) == expected_rows
