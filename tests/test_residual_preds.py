# test_residual_preds.py
# Run w/ uv run pytest tests/test_residual_preds.py -v
# Written in collaboration w/ Claude Opus 5

import numpy as np
import pandas as pd
import pytest

import src.residual_preds as pred_mod
from src.residual_preds import add_split_col, check_ratios, loop_through_resid_params

ALPHAS = [1.0, 5.0]


######## Fixtures ########
@pytest.fixture
def make_df():
    """eligible_group is the stratification key; impact_score_<cohort> is only
    read for the verbose print."""

    def _make(n=30, n_groups=2, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "DAWBA_ID": np.arange(1, n + 1),
                "eligible_group": np.tile(np.arange(n_groups), n // n_groups),
                "impact_score_self": rng.random(n),
            }
        )

    return _make


@pytest.fixture
def make_split_df(make_df):
    """A frame that already carries a balanced split_id, for check_ratios."""

    def _make(n=30, n_splits=3, n_groups=2):
        df = make_df(n=n, n_groups=n_groups)
        df["split_id"] = np.tile(np.arange(n_splits), n // n_splits)
        return df

    return _make


######## add_split_col ########
class TestAddSplitCol:
    def test_every_row_assigned_to_exactly_one_split(self, make_df):
        out = add_split_col(make_df(n=30), "self", n_splits=3)
        assert set(out["split_id"]) == {0, 1, 2}
        assert (out["split_id"] == -1).sum() == 0
        assert len(out) == 30

    def test_input_not_mutated(self, make_df):
        df = make_df()
        add_split_col(df, "self", n_splits=3)
        assert "split_id" not in df.columns

    def test_index_is_reset(self, make_df):
        """Positional fold indices are written with .iloc, so the returned frame
        must carry a clean RangeIndex for downstream positional alignment."""
        df = make_df(n=30)
        df.index = np.arange(100, 130)
        out = add_split_col(df, "self", n_splits=3)
        assert list(out.index) == list(range(30))

    def test_folds_are_near_equal_size(self, make_df):
        counts = add_split_col(make_df(n=30), "self", n_splits=3)[
            "split_id"
        ].value_counts()
        assert counts.max() - counts.min() <= 1

    def test_stratifies_on_eligible_group(self, make_df):
        out = add_split_col(make_df(n=30, n_groups=2), "self", n_splits=3)
        for split in range(3):
            assert set(out.loc[out["split_id"] == split, "eligible_group"]) == {0, 1}

    def test_same_random_state_is_deterministic(self, make_df):
        df = make_df()
        a = add_split_col(df, "self", n_splits=3, random_state=7)
        b = add_split_col(df, "self", n_splits=3, random_state=7)
        pd.testing.assert_series_equal(a["split_id"], b["split_id"])

    def test_different_random_state_gives_different_splits(self, make_df):
        df = make_df(n=30)
        a = add_split_col(df, "self", n_splits=3, random_state=1)
        b = add_split_col(df, "self", n_splits=3, random_state=2)
        assert not a["split_id"].equals(b["split_id"])

    def test_imbalance_beyond_tol_raises(self, make_df):
        df = make_df(n=30)
        df.loc[0, "eligible_group"] = 9  # singleton group
        with pytest.warns(UserWarning, match="least populated class"):
            with pytest.raises(ValueError, match="No instance of"):
                add_split_col(df, "self", n_splits=3)

    def test_missing_eligible_group_raises(self, make_df):
        with pytest.raises(KeyError):
            add_split_col(
                make_df().drop(columns=["eligible_group"]), "self", n_splits=3
            )


######## check_ratios ########
class TestCheckRatios:
    def test_balanced_splits_pass(self, make_split_df):
        check_ratios(make_split_df(), "self", n_splits=3)  # no raise

    def test_group_absent_from_a_split_raises(self, make_split_df):
        df = make_split_df(n=30, n_splits=3)
        df.loc[df["split_id"] == 1, "eligible_group"] = 0  # group 1 gone from split 1
        with pytest.raises(ValueError, match="No instance of 1 in split: 1"):
            check_ratios(df, "self", n_splits=3)

    def test_ratio_drift_beyond_tol_raises(self, make_split_df):
        """Split 0 is skewed toward group 0 while the others stay balanced, so the
        max-min spread exceeds tol."""
        df = make_split_df(n=60, n_splits=3)
        split0 = df.index[df["split_id"] == 0]
        df.loc[split0[: len(split0) - 2], "eligible_group"] = 0
        with pytest.raises(AssertionError, match="Potential error in splitting"):
            check_ratios(df, "self", n_splits=3, tol=0.1)

    def test_same_drift_passes_under_a_loose_tol(self, make_split_df):
        df = make_split_df(n=60, n_splits=3)
        split0 = df.index[df["split_id"] == 0]
        df.loc[split0[: len(split0) - 2], "eligible_group"] = 0
        check_ratios(df, "self", n_splits=3, tol=1.0)  # no raise


######## loop_through_resid_params — parameter coverage ########
class TestLoopCoverage:
    @pytest.fixture
    def run(self, make_df, tmp_path, monkeypatch):
        """Stub get_lr_output and export_dfs, and record one entry per
        (repeat, alphas, base) configuration plus the split ids inside it."""
        configs = []
        splits_seen = []

        def fake_get_lr_output(**kw):
            split_id = kw["split_id"]
            splits_seen.append((kw["selected_alphas"], kw["two_layer_base"], split_id))
            ids = np.arange(split_id * 10, split_id * 10 + 5)
            return (
                pd.DataFrame({f"perm_id_split_{split_id}": np.arange(5)}),
                pd.DataFrame({"DAWBA_ID": ids, "pred_stage3_total": np.zeros(5)}),
                pd.DataFrame({"stage": ["stage1"] * 2, "coef": [0.0, 0.0]}),
            )

        def fake_export_dfs(save_dir, **kw):
            configs.append(save_dir)

        monkeypatch.setattr(pred_mod, "get_lr_output", fake_get_lr_output)
        monkeypatch.setattr(pred_mod, "export_dfs", fake_export_dfs)

        def _run(num_layers, n_repeats=2, n_splits=3, alpha_list=None, **kw):
            path = tmp_path / "in.parquet"
            make_df(n=30).to_parquet(path)
            loop_through_resid_params(
                impact_type="self",
                agg_type="scale-new",
                desc_seg_type="original",
                df_path=path,
                base_save_dir=tmp_path / "out",
                n_splits=n_splits,
                num_layers=num_layers,
                perm_dir=None,
                alpha_list=ALPHAS if alpha_list is None else alpha_list,
                n_repeats=n_repeats,
                **kw,
            )
            return configs, splits_seen

        return _run

    def test_one_layer_covers_repeats_by_alpha3(self, run, tmp_path):
        configs, _ = run(num_layers=1)
        expected = {
            tmp_path / "out" / f"alpha_{a3}" / f"repeat_{r}" / "2_layer_base_None"
            for r in range(2)
            for a3 in ALPHAS
        }
        assert set(configs) == expected
        assert len(configs) == len(expected)  # no duplicates

    def test_three_layer_covers_the_full_alpha_grid(self, run, tmp_path):
        """Both penalties vary independently, so the grid is |alphas|²."""
        configs, _ = run(num_layers=3)
        expected = {
            tmp_path / "out" / f"alpha_{a2}_{a3}" / f"repeat_{r}" / "2_layer_base_None"
            for r in range(2)
            for a2 in ALPHAS
            for a3 in ALPHAS
        }
        assert set(configs) == expected
        assert len(configs) == len(expected)

    def test_two_layer_covers_S_once_and_Q_across_alpha2(self, run, tmp_path):
        """S reuses the sum stage, so alpha2 is inert and only alpha3 varies. Q
        fits an independent questionnaire stage, so both alphas vary."""
        configs, _ = run(num_layers=2)
        expected = set()
        for r in range(2):
            for a3 in ALPHAS:
                expected.add(
                    tmp_path / "out" / f"alpha_{a3}" / f"repeat_{r}" / "2_layer_base_S"
                )
                for a2 in ALPHAS:
                    expected.add(
                        tmp_path
                        / "out"
                        / f"alpha_{a2}_{a3}"
                        / f"repeat_{r}"
                        / "2_layer_base_Q"
                    )
        assert set(configs) == expected
        assert len(configs) == len(expected)

    @pytest.mark.parametrize(
        "num_layers,n_alphas,expected_configs",
        [(1, 2, 2), (1, 3, 3), (2, 2, 6), (2, 3, 12), (3, 2, 4), (3, 3, 9)],
    )
    def test_config_count_per_repeat(self, run, num_layers, n_alphas, expected_configs):
        """Counts per repeat: 1-layer |a|, 2-layer |a|(1+|a|), 3-layer |a|²."""
        alpha_list = [float(i) for i in range(1, n_alphas + 1)]
        configs, _ = run(num_layers=num_layers, n_repeats=1, alpha_list=alpha_list)
        assert len(configs) == expected_configs

    def test_every_config_visits_every_split(self, run):
        configs, splits_seen = run(num_layers=3, n_repeats=1, n_splits=3)
        assert len(splits_seen) == len(configs) * 3
        per_config = {}
        for alphas, base, split_id in splits_seen:
            per_config.setdefault((tuple(alphas), base), []).append(split_id)
        for visited in per_config.values():
            assert sorted(visited) == [0, 1, 2]

    def test_alphas_reach_get_lr_output_in_order(self, run):
        """selected_alphas is [alpha2, alpha3]; the S branch passes None for
        alpha2 since that stage is a copy of the sum stage."""
        _, splits_seen = run(num_layers=2, n_repeats=1)
        pairs = {(tuple(a), b) for a, b, _ in splits_seen}
        assert ((None, 1.0), "S") in pairs
        assert ((1.0, 5.0), "Q") in pairs
        assert all(a[0] is None for a, b in pairs if b == "S")

    def test_repeats_reshuffle_the_splits(self, run, monkeypatch):
        """Each repeat calls add_split_col with seed + repeat, so the folds differ
        across repeats rather than repeating the same partition."""
        seen_states = []
        real = pred_mod.add_split_col

        def spy(**kw):
            seen_states.append(kw["random_state"])
            return real(**kw)

        monkeypatch.setattr(pred_mod, "add_split_col", spy)
        run(num_layers=1, n_repeats=3)
        assert len(set(seen_states)) == 3

    def test_non_ridge_linear_type_rejected(self, run):
        with pytest.raises(ValueError, match="linear_type"):
            run(num_layers=3, linear_type="regular")
