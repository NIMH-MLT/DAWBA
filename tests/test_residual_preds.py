# test_residual_preds.py
# This script was partially written by an LLM

import pytest
import numpy as np
import pandas as pd
from src.config import SEED
from src.residual_preds import (
    add_split_col,
    check_ratios,
    get_best_params_score,
    get_r2_by_outcome,
)


## Helper
def _write_fake_fold(path, n=30, seed=0):
    """Write a minimal parquet file mimicking get_lr_output() output."""
    rng = np.random.default_rng(seed)
    true = rng.integers(0, 16, size=n)
    path.parent.mkdir(parents=True, exist_ok=True)

    pred_stage1 = true + rng.standard_normal(n) * 0.5
    pred_stage2_increment = rng.standard_normal(n) * 0.1
    pred_stage2_total = pred_stage1 + pred_stage2_increment
    pred_stage3_increment = rng.standard_normal(n) * 0.1
    pred_stage3_total = pred_stage2_total + pred_stage3_increment

    pd.DataFrame(
        {
            "DAWBA_ID": np.arange(n),
            "true": true,
            "pred_stage1": pred_stage1,
            "pred_stage2_increment": pred_stage2_increment,
            "pred_stage2_total": pred_stage2_total,
            "pred_stage3_increment": pred_stage3_increment,
            "pred_stage3_total": pred_stage3_total,
        }
    ).to_parquet(path)


class TestAddSplitCol:
    def test_adds_split_id_column(self, base_df_no_splits):
        out = add_split_col(base_df_no_splits, pred_type="self", n_splits=4)
        assert "split_id" in out.columns

    def test_split_values_are_valid(self, base_df_no_splits):
        n_splits = 4
        out = add_split_col(base_df_no_splits, pred_type="self", n_splits=n_splits)
        assert set(out["split_id"].unique()) == set(range(n_splits))

    def test_every_row_assigned(self, base_df_no_splits):
        out = add_split_col(base_df_no_splits, pred_type="self", n_splits=4)
        assert (out["split_id"] == -1).sum() == 0

    def test_original_df_not_mutated(self, base_df_no_splits):
        add_split_col(base_df_no_splits, pred_type="self", n_splits=4)
        assert "split_id" not in base_df_no_splits.columns

    def test_reproducible_with_same_seed(self, base_df_no_splits):
        out1 = add_split_col(
            base_df_no_splits, pred_type="self", n_splits=4, random_state=SEED
        )
        out2 = add_split_col(
            base_df_no_splits, pred_type="self", n_splits=4, random_state=SEED
        )
        pd.testing.assert_series_equal(out1["split_id"], out2["split_id"])


class TestCheckRatios:
    def test_passes_for_balanced_splits(self, base_df_no_splits):
        df = add_split_col(base_df_no_splits, pred_type="self", n_splits=4)
        # Should not raise
        check_ratios(df_w_splits=df, pred_type="self", n_splits=4, tol=0.2)

    def test_raises_for_heavily_disjoint_splits(self, base_df_no_splits):
        """Manually craft a DataFrame where one group is absent in a split."""
        base_df_no_splits["eligible_group"] = [0] * 20 + [1] * 20
        base_df_no_splits["split_id"] = [0] * 20 + [1] * 20  # group 1 never in split 0

        with pytest.raises(ValueError):
            check_ratios(
                df_w_splits=base_df_no_splits,
                pred_type="self",
                n_splits=2,
                tol=0.05,
                verbose=1,
            )

    def test_raises_for_heavily_imbalanced_splits(self, base_df_no_splits):
        """Manually craft a DataFrame where one group is absent in a split."""
        base_df_no_splits["eligible_group"] = [0] * 20 + [1] * 20
        # only one group0 in split1; only one group1 in split0
        base_df_no_splits["split_id"] = [0] * 19 + [1] * 20 + [0]

        with pytest.raises(AssertionError):
            check_ratios(
                df_w_splits=base_df_no_splits,
                pred_type="self",
                n_splits=2,
                tol=0.05,
                verbose=1,
            )


class TestGetBestParamsScore:
    # TODO: test gets actual best metrics
    def _build_ridge_2layer_dir(self, tmp_path, n_repeats=2, n_splits=3):
        """
        Mimic the on-disk structure for ridge / 2-layer:
          <base>/<alpha_dir>/repeat_<r>/split_<s>.parquet
        """
        base = tmp_path / "preds"
        for alpha in [0.1, 1.0]:
            for repeat in range(n_repeats):
                for split in range(n_splits):
                    p = (
                        base
                        / f"alpha_{alpha}"
                        / f"repeat_{repeat}"
                        / f"split_{split}.parquet"
                    )
                    _write_fake_fold(p, seed=split + repeat * n_splits)
        return base

    @pytest.mark.parametrize("metric_agg", ["per_fold", "combined"])
    def test_returns_mse_and_r2_keys(self, tmp_path, metric_agg):
        base = self._build_ridge_2layer_dir(tmp_path)
        result = get_best_params_score(
            import_dir=base, num_layers=2, linear_type="ridge", metric_agg=metric_agg
        )
        assert "mse" in result and "r2" in result

    @pytest.mark.parametrize("metric_agg", ["per_fold", "combined"])
    def test_best_mse_is_lower_than_initial_sentinel(self, tmp_path, metric_agg):
        base = self._build_ridge_2layer_dir(tmp_path)
        result = get_best_params_score(
            import_dir=base, num_layers=2, linear_type="ridge", metric_agg=metric_agg
        )
        assert result["mse"]["avg_mse"] < 9999999

    @pytest.mark.parametrize("metric_agg", ["per_fold", "combined"])
    def test_best_r2_is_higher_than_initial_sentinel(self, tmp_path, metric_agg):
        base = self._build_ridge_2layer_dir(tmp_path)
        result = get_best_params_score(
            import_dir=base, num_layers=2, linear_type="ridge", metric_agg=metric_agg
        )
        assert result["r2"]["avg_r2"] > 0


class TestGetR2ByOutcome:

    def _build_outcome_dir(self, tmp_path, outcomes, n_repeats=2, n_splits=3):
        """
        Mimic the on-disk structure for 2-layer outcome analysis:
          <base>/<outcome>/repeat_<r>/split_<s>.parquet
        """
        base = tmp_path / "outcome_preds"
        for outcome in outcomes:
            for repeat in range(n_repeats):
                for split in range(n_splits):
                    p = base / outcome / f"repeat_{repeat}" / f"split_{split}.parquet"
                    _write_fake_fold(p, seed=split + repeat * n_splits)
        return base

    @pytest.mark.parametrize("metric_agg", ["per_folds", "Combined", "wrong"])
    def test_raises_unknown_metric_agg(self, tmp_path, metric_agg):
        outcomes = ["outcome_a", "outcome_b"]
        base = self._build_outcome_dir(tmp_path, outcomes)
        with pytest.raises(ValueError, match="Unrecognized metric_agg"):
            get_r2_by_outcome(
                pred_dir=base,
                outcomes=outcomes,
                nlayers=2,
                metric_agg=metric_agg,
            )

    @pytest.mark.parametrize("metric_agg", ["per_fold", "combined"])
    def test_returns_dataframe_with_expected_columns(self, tmp_path, metric_agg):
        outcomes = ["outcome_a", "outcome_b"]
        base = self._build_outcome_dir(tmp_path, outcomes)
        result = get_r2_by_outcome(
            pred_dir=base,
            outcomes=outcomes,
            nlayers=2,
            metric_agg=metric_agg,
        )
        assert isinstance(result, pd.DataFrame)
        for col in ["outcome", "avg_r2_L1", "avg_r2_outcome"]:
            assert col in result.columns

    @pytest.mark.parametrize("metric_agg", ["per_fold", "combined"])
    def test_returns_one_row_per_outcome(self, tmp_path, metric_agg):
        outcomes = ["outcome_a", "outcome_b", "outcome_c"]
        base = self._build_outcome_dir(tmp_path, outcomes)
        result = get_r2_by_outcome(
            pred_dir=base, outcomes=outcomes, nlayers=2, metric_agg=metric_agg
        )
        assert len(result) == len(outcomes)

    @pytest.mark.parametrize("metric_agg", ["per_fold", "combined"])
    def test_sorted_descending_by_r2(self, tmp_path, metric_agg):
        outcomes = ["outcome_a", "outcome_b"]
        base = self._build_outcome_dir(tmp_path, outcomes)
        result = get_r2_by_outcome(
            pred_dir=base, outcomes=outcomes, nlayers=2, metric_agg=metric_agg
        )
        r2_vals = result["avg_r2_outcome"].tolist()
        assert r2_vals == sorted(r2_vals, reverse=True)
