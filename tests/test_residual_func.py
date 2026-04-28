# test_residual_func.py
# This script was partially written by an LLM
from src.config import SEED
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression

from src.residual_func import ResidualRegressor, get_lr_output
from .conftest import QUEST_COLS, N_OUTCOMES_LR


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
            perm_dir=None,
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
            perm_dir=None,
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
            perm_dir=None,
        )
        df_out = pd.read_parquet(export_path)
        assert len(df_out) == expected_rows


class TestShuffleVals:
    """Test shuffle_vals in ResidualRegressor."""

    def _write_perm_seed(self, perm_dir, seed):
        perm_dir.mkdir(parents=True, exist_ok=True)
        (perm_dir / "seed.txt").write_text(str(seed))

    def _make_prob_array(self, df):
        """2D array aligned to df rows."""
        return np.random.rand(len(df), N_OUTCOMES_LR)

    def test_raises_if_no_seed_file(self, base_df, tmp_path):
        r = make_regressor()
        X3 = self._make_prob_array(base_df)
        with pytest.raises(ValueError, match="No seed file"):
            r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)

    def test_output_shape_preserved(self, base_df, tmp_path):
        self._write_perm_seed(tmp_path, SEED)
        r = make_regressor()
        X3 = self._make_prob_array(base_df)
        result = r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)
        assert result.shape == X3.shape

    def test_rows_are_permuted(self, base_df, tmp_path):
        """Output rows should be a reordering of input rows, not identical."""
        self._write_perm_seed(tmp_path, SEED)
        r = make_regressor()
        X3 = self._make_prob_array(base_df)
        result = r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)
        assert not np.array_equal(result, X3)

    def test_values_conserved(self, base_df, tmp_path):
        """Every row in the output should exist somewhere in the input."""
        self._write_perm_seed(tmp_path, SEED)
        r = make_regressor()
        X3 = self._make_prob_array(base_df)
        result = r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)
        # Sort rows for comparison — order doesn't matter, content does
        orig_sorted = np.sort(X3, axis=0)
        result_sorted = np.sort(result, axis=0)
        np.testing.assert_array_almost_equal(orig_sorted, result_sorted)

    def test_reproducible_with_same_seed(self, base_df, tmp_path):
        self._write_perm_seed(tmp_path, SEED)
        r = make_regressor()
        X3 = self._make_prob_array(base_df)
        r1 = r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)
        r2 = r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_give_different_results(self, base_df, tmp_path):
        r = make_regressor()
        X3 = self._make_prob_array(base_df)
        results = []
        for i, seed in enumerate([SEED, SEED + 1, SEED + 2]):
            perm_dir = tmp_path / f"perm_{i}"
            self._write_perm_seed(perm_dir, seed)
            results.append(
                r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=perm_dir)
            )
        # No two results from different seeds should be identical
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert not np.array_equal(results[i], results[j])

    def test_id_based_not_positional(self, base_df, tmp_path):
        """Shuffling a reordered DataFrame should give the same assignment."""
        self._write_perm_seed(tmp_path, SEED)
        r = make_regressor()
        X3 = self._make_prob_array(base_df)

        # Shuffle the DataFrame rows and realign X3 accordingly
        df_reordered = base_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        id_order = df_reordered["DAWBA_ID"].values
        orig_order = base_df["DAWBA_ID"].values
        reorder_idx = [np.where(orig_order == i)[0][0] for i in id_order]
        X3_reordered = X3[reorder_idx]

        result_orig = r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)
        result_reordered = r.shuffle_vals(
            X=df_reordered, col_to_shuffle_np=X3_reordered, perm_dir=tmp_path
        )

        np.testing.assert_array_equal(result_orig, result_reordered)

    def test_shuffle_within_groups(self, base_df, tmp_path):
        """Subjects should only receive prob vectors from within their own eligible_group."""
        self._write_perm_seed(tmp_path, SEED)
        r = make_regressor()
        X3 = self._make_prob_array(base_df)

        result = r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)

        # Build a lookup from row content to eligible_group
        # For each row in result, find which subject it came from and check groups match
        for i, row in enumerate(result):
            orig_subject_group = base_df.iloc[i]["eligible_group"]
            # Find which original row this result row came from
            matches = [
                j for j, orig_row in enumerate(X3) if np.array_equal(orig_row, row)
            ]
            assert len(matches) == 1, "Row should match exactly one original row"
            source_group = base_df.iloc[matches[0]]["eligible_group"]
            assert (
                orig_subject_group == source_group
            ), f"Subject in group {orig_subject_group} received a vector from group {source_group}"

    def test_input_not_mutated(self, base_df, tmp_path):
        """shuffle_vals should not alter the input DataFrame or array."""
        self._write_perm_seed(tmp_path, SEED)
        r = make_regressor()
        X3 = self._make_prob_array(base_df)

        df_copy = base_df.copy()
        X3_copy = X3.copy()

        r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)

        pd.testing.assert_frame_equal(base_df, df_copy)
        np.testing.assert_array_equal(X3, X3_copy)

    def test_all_subjects_assigned(self, base_df, tmp_path):
        """Every subject should receive exactly one prob vector — no subject is skipped or duplicated."""
        self._write_perm_seed(tmp_path, SEED)
        r = make_regressor()
        X3 = self._make_prob_array(base_df)

        result = r.shuffle_vals(X=base_df, col_to_shuffle_np=X3, perm_dir=tmp_path)

        assert result.shape[0] == len(base_df), "Output has wrong number of rows"
        # Each row in result should be unique (no subject received the same vector twice)
        unique_rows = {tuple(row) for row in result}
        assert len(unique_rows) == len(
            base_df
        ), "Duplicate rows found — some subjects got the same vector"
