# test_lmg_imp.py
# Run w/ uv run pytest tests/test_lmg_imp.py -v
# Written in collab w/ Claude Opus 5

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

import src.lmg_imp as lmg_mod
import src.Regressor as reg_mod
from src.lmg_imp import agg_over_perms, get_perm_feat_imp, get_starting_r2

QUEST_COLS = ["s_dep_q1", "s_dep_q2", "s_dep_q3"]
PROB_OUTCOMES = ["t1", "t2", "t3", "t4"]


######## Fixtures ########
@pytest.fixture(autouse=True)
def stub_registry(monkeypatch):
    """get_prob_col_outcomes is imported into both modules; pin both bindings."""

    def fake(agg_type, desc_seg_type):
        return ("prob_col", list(PROB_OUTCOMES))

    monkeypatch.setattr(lmg_mod, "get_prob_col_outcomes", fake)
    monkeypatch.setattr(reg_mod, "get_prob_col_outcomes", fake)


@pytest.fixture
def config():
    """ResidualRegressor kwargs. outcome_sub_list and two_layer_base are passed
    separately by lmg_imp, so they must NOT appear here."""
    return dict(
        num_layers=3,
        impact_type="self",
        desc_seg_type="shot_few",
        quest_cols=list(QUEST_COLS),
        agg_type="scale-new",
        alpha2=1.0,
        alpha3=1.0,
        seed=0,
        linear_type="ridge",
        max_iter=5000,
    )


@pytest.fixture
def xy():
    def _make(n=30, seed=0):
        rng = np.random.default_rng(seed)
        data = {"DAWBA_ID": np.arange(1, n + 1)}
        for c in QUEST_COLS:
            data[c] = rng.integers(0, 3, n) * 1.0
        for c in PROB_OUTCOMES:
            data[c] = rng.random(n)
        X = pd.DataFrame(data)
        y = (
            X[QUEST_COLS].sum(axis=1)
            + 2.0 * X["t1"]
            - 1.0 * X["t3"]
            + rng.normal(0, 0.2, n)
        ).to_numpy()
        return X, y

    return _make


######## get_starting_r2 ########
class TestGetStartingR2:
    def test_baseline_is_independent_of_the_text_subset(self, config, xy):
        # NOTE: get_starting_r2 only uses a single text outcome
        # so ensure that single vs full outcome lists give the same baseline
        X, y = xy()
        full = get_starting_r2(config, X, y, X, y)
        one_regressor = reg_mod.ResidualRegressor(
            **config, outcome_sub_list=["t1"], two_layer_base=None
        )
        multiple_regressor = reg_mod.ResidualRegressor(
            **config, outcome_sub_list=None, two_layer_base=None
        )
        one_regressor.fit(X, y)
        one_r2 = r2_score(y, one_regressor.predict_all(X)["pred_stage2_total"])

        multiple_regressor.fit(X, y)
        multiple_r2 = r2_score(
            y, multiple_regressor.predict_all(X)["pred_stage2_total"]
        )
        assert one_r2 == pytest.approx(multiple_r2)
        assert one_r2 == pytest.approx(full)

    def test_two_layer_config_rejected(self, config, xy):
        """two_layer_base is hardcoded to None, so only 3-layer configs can be
        decomposed -- a 2-layer config fails the estimator's own check."""
        X, y = xy()
        with pytest.raises(ValueError, match="two_layer_base"):
            get_starting_r2({**config, "num_layers": 2}, X, y, X, y)

    @pytest.mark.parametrize("bad_key", ["two_layer_base", "outcome_sub_list"])
    def test_hardcoded_keys_rejected(self, config, xy, bad_key):
        """Both are set by lmg_imp itself; a config carrying them (as a saved
        BEST_PARAM_DICT entry would) is rejected rather than colliding with the
        explicit arguments."""
        X, y = xy()
        with pytest.raises(ValueError, match="must not be in model_config_dict"):
            get_starting_r2({**config, bad_key: None}, X, y, X, y)

    def test_baseline_uses_one_outcome_but_matches_the_full_fit(self, config, xy):
        """pred_stage2_total never touches stage 3, so fitting a single text
        outcome gives the same SQ baseline as fitting all of them."""
        X, y = xy()
        full = reg_mod.ResidualRegressor(
            **config, outcome_sub_list=list(PROB_OUTCOMES), two_layer_base=None
        )
        full.fit(X, y)
        expected = r2_score(y, full.predict_all(X)["pred_stage2_total"])
        assert get_starting_r2(config, X, y, X, y) == pytest.approx(expected)


######## get_perm_feat_imp ########
class TestGetPermFeatImp:
    @pytest.mark.parametrize("bad_key", ["two_layer_base", "outcome_sub_list"])
    def test_guard_fires_before_any_fitting(self, config, xy, bad_key):
        """The check is at the top of get_perm_feat_imp, so a bad config fails
        immediately rather than after the first outcome is fit."""
        X, y = xy()
        with pytest.raises(ValueError, match="must not be in model_config_dict"):
            get_perm_feat_imp(
                X_train=X,
                y_train=y,
                X_test=X,
                y_test=y,
                model_config_dict={**config, bad_key: None},
                random_state=1,
            )

    def test_shape_and_labels(self, config, xy):
        X, y = xy()
        res = get_perm_feat_imp(
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            model_config_dict=config,
            random_state=1,
        )
        assert list(res.index) == sorted(PROB_OUTCOMES)
        assert res.index.name == "outcome"
        assert list(res.columns) == ["original", "reversed"]

    def test_columns_suffixed_when_fold_given(self, config, xy):
        """lmg_imp_wrapper joins the per-fold frames on the outcome index, so the
        columns must be unique per (repeat, split_id) or the join collides."""
        X, y = xy()
        res = get_perm_feat_imp(
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            model_config_dict=config,
            random_state=1,
            repeat=2,
            split_id=1,
        )
        assert list(res.columns) == ["original_2_1", "reversed_2_1"]

    def test_contributions_sum_to_the_text_stage_increment(self, config, xy):
        X, y = xy()
        res = get_perm_feat_imp(
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            model_config_dict=config,
            random_state=1,
        )
        r2_sq = get_starting_r2(config, X, y, X, y)
        full = reg_mod.ResidualRegressor(
            **config, outcome_sub_list=list(PROB_OUTCOMES), two_layer_base=None
        )
        full.fit(X, y)
        r2_full = r2_score(y, full.predict(X))

        assert res["original"].sum() == pytest.approx(r2_full - r2_sq, abs=1e-6)
        assert res["reversed"].sum() == pytest.approx(r2_full - r2_sq, abs=1e-6)

    def test_both_orderings_allocate_the_same_total(self, config, xy):
        """Ridge is invariant to feature order, so the two orderings differ only
        in HOW they split the increment -- averaging them is the whole point."""
        X, y = xy()
        res = get_perm_feat_imp(
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            model_config_dict=config,
            random_state=1,
        )
        assert res["original"].sum() == pytest.approx(res["reversed"].sum(), abs=1e-6)
        assert not np.allclose(res["original"], res["reversed"])

    def test_deterministic_given_random_state(self, config, xy):
        X, y = xy()
        kw = dict(
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            model_config_dict=config,
        )
        pd.testing.assert_frame_equal(
            get_perm_feat_imp(random_state=7, **kw),
            get_perm_feat_imp(random_state=7, **kw),
        )

    def test_random_state_changes_the_ordering(self, config, xy):
        """Each permutation iteration draws a different feature order; identical
        results across seeds would mean the ordering never varied."""
        X, y = xy()
        kw = dict(
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            model_config_dict=config,
        )
        a = get_perm_feat_imp(random_state=1, **kw)
        assert any(
            not np.allclose(
                a["original"], get_perm_feat_imp(random_state=s, **kw)["original"]
            )
            for s in (2, 3, 4)
        )

    def test_contributions_may_be_negative(self, config, xy):
        """Evaluated out-of-sample, adding a feature can lower r2 -- so a negative
        contribution is valid rather than a bug, and anything clipping at zero
        downstream would break the sum invariant above."""
        X, y = xy(n=40, seed=3)
        train = slice(0, 20)
        test = slice(20, None)
        res = get_perm_feat_imp(
            X_train=X.iloc[train],
            y_train=y[train],
            X_test=X.iloc[test],
            y_test=y[test],
            model_config_dict={**config, "alpha3": 0.001},
            random_state=1,
        )
        assert (res.to_numpy() < 0).any()


######## agg_over_perms ########
class TestAggOverPerms:
    """Collapses the per-ordering LMG results into a single allocation of
    explained variance across outcomes.

    Usage: each ordering is a separate SLURM job writing
    lmg_results/perm_XXX/{impact_type}.parquet via lmg_imp_wrapper. The
    aggregation step reads those files and averages over orderings. This is the
    LMG decomposition itself, not a significance test -- avg_mean is each
    outcome's share of the text stage's incremental R^2, and std_mean is Monte
    Carlo error telling you whether enough orderings were sampled.
    """

    N_REPEATS = 10
    N_SPLITS = 3

    def _wrapper_output(self, perm_num, outcome_means, seed=0):
        """A frame shaped exactly as lmg_imp_wrapper returns it: two per-fold
        columns for every (repeat, split) plus the four appended aggregates."""
        rng = np.random.default_rng(seed + perm_num)
        idx = pd.Index(sorted(PROB_OUTCOMES), name="outcome")
        cols = {}
        for repeat in range(self.N_REPEATS):
            for split_id in range(self.N_SPLITS):
                for prefix in ("original", "reversed"):
                    cols[f"{prefix}_{repeat}_{split_id}"] = [
                        m + rng.normal(0, 0.01) for m in outcome_means
                    ]
        df = pd.DataFrame(cols, index=idx)
        og = [c for c in df.columns if c.startswith("original")]
        rev = [c for c in df.columns if c.startswith("reversed")]
        return df.assign(
            original_mean=df[og].mean(axis=1).round(4),
            original_std=df[og].std(axis=1).round(4),
            reversed_mean=df[rev].mean(axis=1).round(4),
            reversed_std=df[rev].std(axis=1).round(4),
        )

    def _prepared(self, perm_num, outcome_means, seed=0):
        """The read-side step the aggregation needs: keep only the aggregates and
        tag them with the ordering's directory name."""
        return self._wrapper_output(perm_num, outcome_means, seed)[
            ["original_mean", "original_std", "reversed_mean", "reversed_std"]
        ].add_suffix(f"_perm_{perm_num:03d}")

    ## Contract with lmg_imp_wrapper ##
    def test_raw_wrapper_output_is_rejected(self):
        with pytest.raises(AssertionError):
            agg_over_perms([self._wrapper_output(0, [0.1, 0.2, 0.3, 0.4])])

    def test_prepared_frames_aggregate(self):
        """With the subset-and-suffix step applied, the ordering files collapse to
        one row per outcome."""
        dfs = [self._prepared(n, [0.1, 0.2, 0.3, 0.4]) for n in range(5)]
        out = agg_over_perms(dfs)
        assert list(out.columns) == ["avg_mean", "std_mean"]
        assert list(out.index) == sorted(PROB_OUTCOMES)
        assert out.index.name == "outcome"

    ## The allocation ##
    def test_allocation_sums_to_the_text_stage_increment(self):
        """LMG's defining property: every ordering allocates exactly the full
        model's increment over the SQ baseline across the outcomes, so the average
        over orderings does too. Without this the per-outcome numbers are not
        shares of anything and cannot be read as percentages."""
        increment = 0.40
        shares = [0.05, 0.10, 0.15, 0.10]  # sums to increment
        dfs = [self._prepared(n, shares, seed=n) for n in range(20)]
        out = agg_over_perms(dfs)
        assert out["avg_mean"].sum() == pytest.approx(increment, abs=0.01)

    def test_avg_mean_pools_orderings_and_their_reverses(self):
        """Each file contributes original_mean and reversed_mean; both are
        orderings and both are averaged, so avg_mean is the mean over
        2 x n_orderings estimates."""
        dfs = [
            self._prepared(0, [0.10, 0.20, 0.30, 0.40], seed=1),
            self._prepared(1, [0.30, 0.20, 0.10, 0.40], seed=1),
        ]
        out = agg_over_perms(dfs)
        assert out.loc["t1", "avg_mean"] == pytest.approx(0.20, abs=0.01)
        assert out.loc["t2", "avg_mean"] == pytest.approx(0.20, abs=0.01)

    def test_relative_shares_are_preserved(self):
        """The interpretable claim is each outcome's share of the increment, so
        the ordering and ratios between outcomes must survive aggregation."""
        dfs = [self._prepared(n, [0.05, 0.10, 0.20, 0.05], seed=n) for n in range(10)]
        out = agg_over_perms(dfs)
        assert out["avg_mean"].idxmax() == "t3"
        total = out["avg_mean"].sum()
        assert out.loc["t3", "avg_mean"] / total == pytest.approx(0.5, abs=0.02)

    def test_negative_contributions_survive_aggregation(self):
        """Out-of-sample deltas can be negative; nothing here may clip them, or
        the shares stop summing to the increment."""
        dfs = [self._prepared(n, [-0.05, 0.10, -0.02, 0.20], seed=4) for n in range(4)]
        out = agg_over_perms(dfs)
        assert out.loc["t1", "avg_mean"] < 0
        assert out.loc["t3", "avg_mean"] < 0
        assert out["avg_mean"].sum() == pytest.approx(0.23, abs=0.01)

    ## Monte Carlo error ##
    def test_std_mean_shrinks_when_orderings_agree(self):
        """std_mean is convergence diagnostic, not inference: small relative to
        avg_mean means enough orderings were sampled for the allocation to have
        settled."""
        agreeing = [self._prepared(n, [0.2, 0.2, 0.2, 0.2], seed=5) for n in range(4)]
        disagreeing = [
            self._prepared(0, [0.05] * 4, seed=5),
            self._prepared(1, [0.35] * 4, seed=5),
            self._prepared(2, [0.05] * 4, seed=5),
            self._prepared(3, [0.35] * 4, seed=5),
        ]
        assert agg_over_perms(agreeing)["std_mean"].max() < 0.02
        assert agg_over_perms(disagreeing)["std_mean"].min() > 0.10

    def test_per_ordering_std_columns_excluded(self):
        """The *_std columns in each file are fold-level noise within one
        ordering, a different quantity from the across-ordering spread -- folding
        them in would conflate the two."""
        dfs = [self._prepared(n, [0.1] * 4, seed=2) for n in range(3)]
        for df in dfs:
            for c in [c for c in df.columns if "std" in c]:
                df[c] = 99.0
        out = agg_over_perms(dfs)
        assert out["avg_mean"].max() < 1.0
        assert out["std_mean"].max() < 1.0

    def test_single_mean_column_gives_nan_spread(self):
        df = pd.DataFrame(
            {"original_mean_perm_000": [0.1, 0.2, 0.3, 0.4]},
            index=pd.Index(sorted(PROB_OUTCOMES), name="outcome"),
        )
        out = agg_over_perms([df])
        assert out["std_mean"].isna().all()

    ## Edge cases from the swarm layout ##
    def test_outcome_present_in_only_some_orderings(self):
        """The join is outer, so an ordering whose registry differed averages over
        fewer estimates for the missing outcomes without any warning -- and the
        shares then no longer sum to the increment. A real risk across ~1000 jobs
        run against a shared config."""
        full = self._prepared(0, [0.10, 0.10, 0.10, 0.10], seed=7)
        partial = self._prepared(1, [0.30, 0.30, 0.30, 0.30], seed=7).iloc[:2]
        with pytest.raises(ValueError):
            agg_over_perms([full, partial])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            agg_over_perms([])
