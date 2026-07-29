#test_cv_splitter.py

import numpy as np
import pytest

from src.residual_preds import add_split_col, check_ratios


# --------------------------------------------------------------------------- #
# add_split_col
# --------------------------------------------------------------------------- #
class TestAddSplitCol:
    def test_every_row_assigned_a_valid_split(self, make_stratified_df):
        df = make_stratified_df(n_per_group=15)  # 30 rows, 2 groups
        out = add_split_col(df, impact_type="self", n_splits=3)
        sid = out["split_id"]
        assert (sid >= 0).all()
        assert set(sid.unique()) == {0, 1, 2}

    def test_splits_partition_all_rows_near_evenly(self, make_stratified_df):
        df = make_stratified_df(n_per_group=20)  # 40 rows
        out = add_split_col(df, impact_type="self", n_splits=4)
        counts = out["split_id"].value_counts()
        assert counts.sum() == len(out)
        assert set(counts.index) == {0, 1, 2, 3}
        # StratifiedKFold balances fold sizes to within one row
        assert counts.max() - counts.min() <= 1

    def test_deterministic_for_fixed_random_state(self, make_stratified_df):
        df = make_stratified_df(n_per_group=15)
        a = add_split_col(df, impact_type="self", n_splits=3, random_state=7)
        b = add_split_col(df, impact_type="self", n_splits=3, random_state=7)
        np.testing.assert_array_equal(
            a["split_id"].to_numpy(), b["split_id"].to_numpy()
        )

    def test_different_random_state_changes_partition(self, make_stratified_df):
        # seed + repeat is how the pipeline gets a fresh partition each repeat
        df = make_stratified_df(n_per_group=15)
        a = add_split_col(df, impact_type="self", n_splits=3, random_state=1)
        b = add_split_col(df, impact_type="self", n_splits=3, random_state=2)
        assert not np.array_equal(
            a["split_id"].to_numpy(), b["split_id"].to_numpy()
        )

    def test_stratification_keeps_every_group_in_every_split(self, make_stratified_df):
        df = make_stratified_df(n_per_group=15)
        out = add_split_col(df, impact_type="self", n_splits=3)
        for s in range(3):
            present = set(out[out["split_id"] == s]["eligible_group"])
            assert present == {0, 1}

    def test_input_frame_not_mutated(self, make_stratified_df):
        df = make_stratified_df(n_per_group=15)
        cols_before = list(df.columns)
        _ = add_split_col(df, impact_type="self", n_splits=3)
        assert "split_id" not in df.columns
        assert list(df.columns) == cols_before

    def test_returned_frame_has_reset_index(self, make_stratified_df):
        df = make_stratified_df(n_per_group=15)
        df.index = df.index + 1000  # non-default index going in
        out = add_split_col(df, impact_type="self", n_splits=3)
        assert list(out.index) == list(range(len(out)))

    def test_too_few_members_per_class_raises(self, make_stratified_df):
        # StratifiedKFold cannot build 5 folds from 2 members of a class
        df = make_stratified_df(n_per_group=2)
        with pytest.raises(ValueError):
            add_split_col(df, impact_type="self", n_splits=5)

    def test_verbose_emits_pred_type(self, make_stratified_df, capsys):
        df = make_stratified_df(n_per_group=15)
        add_split_col(df, impact_type="self", n_splits=3, verbose=1)
        assert "Pred type: self" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# check_ratios
# --------------------------------------------------------------------------- #
class TestCheckRatios:
    def test_balanced_splits_pass(self, make_split_frame):
        df = make_split_frame({0: {0: 5, 1: 5}, 1: {0: 5, 1: 5}})
        check_ratios(df, impact_type="self", n_splits=2)  # must not raise

    def test_missing_group_in_a_split_raises(self, make_split_frame):
        # split 1 contains no group-1 members
        df = make_split_frame({0: {0: 5, 1: 5}, 1: {0: 10}})
        with pytest.raises(ValueError, match="No instance of"):
            check_ratios(df, impact_type="self", n_splits=2)

    def test_ratio_imbalance_beyond_tol_raises(self, make_split_frame):
        # group-0 ratio: split0 = 3/6 = 0.50, split1 = 3/4 = 0.75 -> diff 0.25
        df = make_split_frame({0: {0: 3, 1: 3}, 1: {0: 3, 1: 1}})
        with pytest.raises(AssertionError, match="Potential error in splitting"):
            check_ratios(df, impact_type="self", n_splits=2, tol=0.1)

    def test_imbalance_within_tol_passes(self, make_split_frame):
        # group ratios differ by 0.1; tol 0.15 admits it
        df = make_split_frame({0: {0: 5, 1: 5}, 1: {0: 6, 1: 4}})
        check_ratios(df, impact_type="self", n_splits=2, tol=0.15)

    def test_only_first_n_splits_are_examined(self, make_split_frame):
        # an extra split_id=2 (single group) exists but n_splits=2 ignores it
        df = make_split_frame({0: {0: 5, 1: 5}, 1: {0: 5, 1: 5}, 2: {0: 1}})
        check_ratios(df, impact_type="self", n_splits=2)  # must not raise

    def test_verbose_prints_scores_and_ok(self, make_split_frame, capsys):
        df = make_split_frame({0: {0: 5, 1: 5}, 1: {0: 5, 1: 5}})
        check_ratios(df, impact_type="self", n_splits=2, verbose=1)
        out = capsys.readouterr().out
        assert "avg score" in out
        assert "Splits are all good" in out