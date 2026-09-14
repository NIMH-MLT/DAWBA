# test_sig.py
# Run w/ uv run pytest tests/test_sig.py -v
# Written in collaboration w/ Claude Opus 5

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import r2_score

import src.sig as sig_mod
from src.sig import (
    format_p_val,
    get_baseline_reduced,
    get_hyp1_null,
    get_hyp2_null,
    get_p,
    order_perm_dir,
    prep_train_test_model,
)
from src.Regressor import IndependentRegressor, ResidualRegressor

QUEST = {"self": ["q1", "q2", "q3"], "parent": ["p1", "p2"]}


######## Fixtures ########
@pytest.fixture
def perm_tree(tmp_path):
    """A permutation parent dir as write_dirs leaves it: perm_xxx folders each
    holding seed.txt, plus the seedless baseline folder."""

    def _make(nums=range(0, 12), start_seed=100, extras=("baseline",), files=()):
        root = tmp_path / "perms"
        root.mkdir()
        for n in nums:
            d = root / f"perm_{n:03d}"
            d.mkdir()
            (d / "seed.txt").write_text(str(start_seed + n))
        for name in extras:
            (root / name).mkdir()
        for name in files:
            (root / name).write_text("not a dir")
        return root

    return _make


@pytest.fixture
def null_frame():
    """Westfall-Young input: row 0 is the observed statistic, rows 1..n are the
    permutation null, one column per sub-hypothesis."""

    def _make(observed, null_rows, cols=("SQT_self", "ST_self", "QT_self")):
        return pd.DataFrame(
            [list(observed), *[list(r) for r in null_rows]], columns=list(cols)
        )

    return _make


@pytest.fixture
def stub_model():
    """Records how fit() was called and returns per-perm predictions, so the null
    loop can be checked without standing up a real regressor."""

    class _Stub:
        def __init__(self, preds_by_perm, default):
            self.preds_by_perm = preds_by_perm
            self.default = default
            self.fit_calls = []
            self._current = None

        def fit(self, X, y, shuffle=None, perm_dir=None):
            self.fit_calls.append(
                (shuffle, None if perm_dir is None else perm_dir.name)
            )
            self._current = None if perm_dir is None else perm_dir.name
            return self

        def predict(self, X):
            return self.preds_by_perm.get(self._current, self.default)

    return _Stub


@pytest.fixture
def prep_env(tmp_path, monkeypatch):
    """Stubs the four collaborators prep_train_test_model reaches through --
    path builders, the loader, and the column registry -- and writes real
    parquet files so the read path is exercised."""

    def _setup(train_rows=6, test_rows=4):
        train_df = pd.DataFrame({"DAWBA_ID": range(train_rows), "split": "train"})
        test_df = pd.DataFrame({"DAWBA_ID": range(test_rows), "split": "test"})
        train_path = tmp_path / "train.parquet"
        test_path = tmp_path / "test.parquet"
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)

        seen = {}
        monkeypatch.setattr(
            sig_mod, "get_df_path", lambda **kw: seen.update(train=kw) or train_path
        )
        monkeypatch.setattr(
            sig_mod, "get_test_df_path", lambda **kw: seen.update(test=kw) or test_path
        )
        monkeypatch.setattr(
            sig_mod,
            "get_X_y",
            lambda df, config_dict, scale: (
                df.copy(),
                pd.Series(np.arange(len(df), dtype=float)),
            ),
        )
        monkeypatch.setattr(
            sig_mod,
            "col_config_sensitivity",
            dict(self_quest=QUEST["self"], parent_quest=QUEST["parent"]),
        )
        return seen

    return _setup


@pytest.fixture
def selected_params():
    def _make(num_layers=3, **overrides):
        params = dict(
            num_layers=num_layers,
            desc_seg_type="original",
            agg_type="avg",
            alpha2=1.0,
            alpha3=1.0,
            llm_model="llama_80b",
            temp=0.0,
        )
        if num_layers == 2:
            params["two_layer_base"] = "S"
        params.update(overrides)
        return params

    return _make


@pytest.fixture
def shared_dict():
    return dict(seed=0, linear_type="ridge", max_iter=5000)


######## order_perm_dir ########
class TestOrderPermDir:
    def test_range_is_half_open(self, perm_tree):
        root = perm_tree(nums=range(0, 10))
        got = [n for n, _ in order_perm_dir(root, start_perm=2, end_perm=5)]
        assert got == [2, 3, 4]

    def test_sorted_numerically_not_lexically(self, perm_tree):
        """iterdir() order is arbitrary and perm_002 < perm_010 only holds by
        luck of zero-padding; the null distribution's row order has to be
        reproducible across machines."""
        root = perm_tree(nums=[10, 2, 7, 1])
        got = [n for n, _ in order_perm_dir(root, start_perm=0, end_perm=100)]
        assert got == [1, 2, 7, 10]

    def test_baseline_and_loose_files_skipped(self, perm_tree):
        root = perm_tree(nums=range(0, 3), files=("perm_009.txt", "notes.md"))
        names = [d.name for _, d in order_perm_dir(root, 0, 100)]
        assert names == ["perm_000", "perm_001", "perm_002"]

    @pytest.mark.parametrize("bad", ["perm_abc", "noperm", "perm_"])
    def test_unparseable_dir_names_skipped_not_raised(self, perm_tree, bad):
        """A stray directory shouldn't abort a multi-hour null run."""
        root = perm_tree(nums=range(0, 2), extras=("baseline", bad))
        assert [n for n, _ in order_perm_dir(root, 0, 100)] == [0, 1]

    def test_empty_range_returns_empty(self, perm_tree):
        root = perm_tree(nums=range(0, 5))
        assert order_perm_dir(root, start_perm=10, end_perm=20) == []

    def test_returns_paths_that_still_carry_the_seed(self, perm_tree):
        """Downstream reads seed.txt out of these, so they must be the perm dirs
        themselves, not their names."""
        root = perm_tree(nums=[3], start_seed=100)
        ((_, perm_dir),) = order_perm_dir(root, 0, 100)
        assert (perm_dir / "seed.txt").read_text() == "103"


######## format_p_val ########
class TestFormatPVal:
    @pytest.mark.parametrize(
        "p,expected",
        [
            (0.0, "<0.001"),
            (0.0009, "<0.001"),
            (0.001, "0.001"),
            (0.0123, "0.012"),
            (0.0497, "0.050"),
            (0.052, "0.05"),
            (0.05, "0.05"),
            (0.5, "0.50"),
            (1.0, "1.00"),
        ],
    )
    def test_threshold_boundaries(self, p, expected):
        assert format_p_val(p) == expected

    def test_always_returns_a_string(self):
        """Uniform type keeps the formatted column object-dtype by intent rather
        than by accident, and stops anything downstream comparing a display
        value numerically."""
        assert all(
            isinstance(format_p_val(p), str)
            for p in (0.0, 0.0005, 0.001, 0.03, 0.2, 1.0)
        )

    def test_decimal_count_disambiguates_the_alpha_boundary(self):
        """Both round to 0.05, so the number of decimals is what separates them:
        three decimals means the raw value was below alpha, two means it was not."""
        assert format_p_val(0.0499) == "0.050"
        assert format_p_val(0.052) == "0.05"
        assert format_p_val(0.0499) != format_p_val(0.052)

    def test_trailing_zero_is_preserved(self):
        assert format_p_val(0.0499).count("0") == 3
        assert len(format_p_val(0.0499).split(".")[1]) == 3
        assert len(format_p_val(0.052).split(".")[1]) == 2

    def test_three_digits_below_alpha_two_above(self):
        assert format_p_val(0.04444) == "0.044"
        assert format_p_val(0.06666) == "0.07"

    def test_exact_alpha_is_not_significant(self):
        """p == 0.05 takes the else branch, so it formats with two decimals like
        any other non-significant value."""
        assert format_p_val(0.05) == "0.05"


######## get_baseline_reduced ########
class TestGetBaselineReduced:
    @pytest.fixture
    def pred_df(self):
        return pd.DataFrame(
            {
                "pred_stage1": [1.0, 2.0, 3.0, 4.0],
                "pred_stage2_total": [1.5, 2.5, 2.5, 4.5],
                "pred_stage3_total": [9.0, 9.0, 9.0, 9.0],
            }
        )

    @pytest.mark.parametrize(
        "setup,col",
        [
            ("SQT", "pred_stage2_total"),
            ("QT", "pred_stage2_total"),
            ("ST", "pred_stage1"),
        ],
    )
    def test_reduced_column_per_setup(self, pred_df, setup, col):
        """The reduced model is everything below the text layer: S+Q for SQT, Q
        alone for QT (stage1 is zeroed there), S alone for ST."""
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert get_baseline_reduced(setup, y, pred_df) == r2_score(y, pred_df[col])

    def test_never_reads_the_full_model_column(self, pred_df):
        """pred_stage3_total is the full model; if it leaked in here the observed
        delta would be 0 by construction."""
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        full = r2_score(y, pred_df["pred_stage3_total"])
        assert all(
            get_baseline_reduced(s, y, pred_df) != full for s in ("SQT", "ST", "QT")
        )

    @pytest.mark.parametrize("bad", ["SQ", "sqt", "", "T", None])
    def test_unknown_setup_raises(self, pred_df, bad):
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="Unknown residual_setup"):
            get_baseline_reduced(bad, y, pred_df)

    def test_missing_column_raises(self):
        y = pd.Series([1.0, 2.0])
        with pytest.raises(KeyError):
            get_baseline_reduced(
                "ST", y, pd.DataFrame({"pred_stage2_total": [1.0, 2.0]})
            )


######## get_p ########
class TestGetP:
    def test_counts_rows_whose_max_reaches_the_observed(self, null_frame):
        """max-T: the null is the row-wise maximum across sub-hypotheses, which
        is what buys the family-wise error control."""
        df = null_frame(
            observed=(0.5, 0.2, 0.1),
            null_rows=[
                (0.6, 0.0, 0.0),
                (0.5, 0.1, 0.1),
                (0.1, 0.2, 0.3),
                (0.0, 0.0, 0.0),
            ],
        )
        p, col = get_p(df)
        assert col == "SQT_self"
        assert p == pytest.approx(3 / 5)  # observed row + two null rows at/above 0.5

    def test_a_non_winning_column_can_still_beat_the_observed(self, null_frame):
        """The whole point of max-T: a permutation counts if ANY sub-hypothesis
        exceeds the observed max, not just the column that won."""
        df = null_frame(observed=(0.5, 0.2, 0.1), null_rows=[(0.0, 0.0, 0.9)])
        p, _ = get_p(df)
        assert p == pytest.approx(2 / 2)

    def test_observed_row_is_inside_the_denominator(self, null_frame):
        """04_sig_test prepends the observed delta to the null, so the +1 in the
        p-value numerator and denominator is already there. p can never be 0."""
        df = null_frame(observed=(0.9, 0.1, 0.1), null_rows=[(0.0, 0.0, 0.0)] * 9)
        p, _ = get_p(df)
        assert p == pytest.approx(1 / 10)

    def test_add_1_flag_double_counts_when_observed_is_prepended(self, null_frame):
        """add_1_flag exists for a null list that does NOT already contain the
        observed row. Passing True in the notebook's setup adds the observed
        twice -- pinned so the two call styles stay distinguishable."""
        df = null_frame(observed=(0.9, 0.1, 0.1), null_rows=[(0.0, 0.0, 0.0)] * 9)
        assert get_p(df, add_1_flag=True)[0] == pytest.approx(2 / 11)

    def test_ties_count_as_more_extreme(self, null_frame):
        """>= not >, so an exact tie is conservative (raises p) rather than
        discarded."""
        df = null_frame(observed=(0.5, 0.0, 0.0), null_rows=[(0.5, 0.0, 0.0)])
        assert get_p(df)[0] == pytest.approx(2 / 2)

    def test_single_column_frame(self, null_frame):
        """Westfall-Young iteration 3 drops down to one remaining column."""
        df = null_frame(
            observed=(0.4,), null_rows=[(0.5,), (0.1,), (0.1,)], cols=("QT_self",)
        )
        p, col = get_p(df)
        assert col == "QT_self"
        assert p == pytest.approx(2 / 4)

    def test_negative_statistics_supported(self, null_frame):
        """R^2 deltas can be negative on a shuffled fit; idxmax/max must not be
        assuming positives."""
        df = null_frame(observed=(-0.1, -0.4, -0.9), null_rows=[(-0.5, -0.5, -0.5)] * 3)
        p, col = get_p(df)
        assert col == "SQT_self"
        assert p == pytest.approx(1 / 4)

    def test_step_down_sequence_is_monotone(self, null_frame):
        """The notebook takes a running max over iterations, so the reported
        p-values are non-decreasing as columns drop out."""
        df = null_frame(
            observed=(0.5, 0.3, 0.1),
            null_rows=[(0.6, 0.2, 0.0), (0.1, 0.4, 0.0), (0.0, 0.0, 0.0)],
        )
        p1, c1 = get_p(df)
        p2, c2 = get_p(df.drop(c1, axis=1))
        p3, _ = get_p(df.drop([c1, c2], axis=1))
        assert max(p1, p2) >= p1 and max(p1, p2, p3) >= max(p1, p2)


######## get_hyp1_null ########
class TestGetHyp1Null:
    def test_one_entry_per_perm_in_range(self, perm_tree, stub_model):
        """Guards against the loop returning early -- the null needs every
        permutation in [start, end), not the first one."""
        root = perm_tree(nums=range(0, 10))
        y_test = pd.Series([1.0, 2.0, 3.0, 4.0])
        m = stub_model({}, default=np.array([1.0, 2.0, 3.0, 4.0]))
        out = get_hyp1_null(None, None, None, y_test, m, root, 0.4, 2, 7)
        assert len(out) == 5
        assert [c[1] for c in m.fit_calls] == [f"perm_{n:03d}" for n in range(2, 7)]

    def test_refits_with_shuffle_text_and_the_current_perm_dir(
        self, perm_tree, stub_model
    ):
        """The model is refit per permutation; a missed shuffle= or perm_dir=
        would silently reuse the previous draw."""
        root = perm_tree(nums=range(0, 3))
        y_test = pd.Series([1.0, 2.0, 3.0])
        m = stub_model({}, default=np.array([1.0, 2.0, 3.0]))
        get_hyp1_null(None, None, None, y_test, m, root, 0.0, 0, 3)
        assert all(shuffle == "text" for shuffle, _ in m.fit_calls)
        assert all(name is not None for _, name in m.fit_calls)

    def test_empty_range_returns_empty_and_never_fits(self, perm_tree, stub_model):
        root = perm_tree(nums=range(0, 5))
        m = stub_model({}, default=np.array([1.0]))
        assert (
            get_hyp1_null(None, None, None, pd.Series([1.0]), m, root, 0.1, 99, 100)
            == []
        )
        assert m.fit_calls == []


######## get_hyp2_null ########
class TestGetHyp2Null:
    @pytest.fixture
    def preds(self):
        rng = np.random.default_rng(0)
        y_true = pd.Series(rng.normal(size=25))
        return (
            y_true,
            pd.Series(y_true + rng.normal(0, 0.5, 25)),
            pd.Series(y_true + rng.normal(0, 0.3, 25)),
        )

    def test_one_entry_per_perm(self, perm_tree, preds):
        y, sq, st = preds
        root = perm_tree(nums=range(0, 8))
        assert len(get_hyp2_null(sq, st, y, root, 1, 6)) == 5

    def test_matches_an_independent_swap_implementation(self, perm_tree, preds):
        """Pins the exact draw: RandomState(seed).rand < 0.5 over the paired
        predictions. A switch to default_rng, or flipping the threshold, changes
        every null already on disk."""
        y, sq, st = preds
        root = perm_tree(nums=[0, 1], start_seed=100)
        expected = []
        for seed in (100, 101):
            rng = np.random.RandomState(seed)
            mask = rng.rand(len(y)) < 0.5
            a = np.where(mask, st, sq)
            b = np.where(mask, sq, st)
            expected.append(r2_score(y, a) - r2_score(y, b))
        np.testing.assert_allclose(get_hyp2_null(sq, st, y, root, 0, 2), expected)

    def test_swap_is_complementary(self, perm_tree, preds):
        """Each participant contributes to exactly one side per draw -- so a
        null entry is the observed diff with a random subset sign-flipped, and
        flipping the argument order negates the whole list."""
        y, sq, st = preds
        root = perm_tree(nums=range(0, 5))
        forward = get_hyp2_null(sq, st, y, root, 0, 5)
        reverse = get_hyp2_null(st, sq, y, root, 0, 5)
        np.testing.assert_allclose(forward, [-d for d in reverse])

    def test_identical_predictions_give_a_degenerate_null(self, perm_tree, preds):
        """If both models predict the same thing, no swap can produce a
        difference -- the null collapses to zeros rather than to noise."""
        y, sq, _ = preds
        root = perm_tree(nums=range(0, 4))
        np.testing.assert_allclose(get_hyp2_null(sq, sq, y, root, 0, 4), 0.0)

    def test_same_seed_gives_the_same_draw(self, perm_tree, preds, tmp_path):
        """Two perm dirs sharing a seed must produce identical entries -- the
        only randomness is seed.txt, not iteration position."""
        y, sq, st = preds
        root = tmp_path / "dup"
        root.mkdir()
        for name in ("perm_000", "perm_001"):
            d = root / name
            d.mkdir()
            (d / "seed.txt").write_text("42")
        out = get_hyp2_null(sq, st, y, root, 0, 2)
        assert out[0] == pytest.approx(out[1])

    def test_alignment_is_positional_not_by_index(self, perm_tree, preds):
        """r2_score and np.where both drop to numpy, so a non-matching index is
        ignored rather than aligned. Callers must pass y_true and the two
        prediction Series in the same row order."""
        y, sq, st = preds
        root = perm_tree(nums=range(0, 3))
        baseline = get_hyp2_null(sq, st, y, root, 0, 3)

        reindexed = get_hyp2_null(
            sq.set_axis(np.arange(100, 100 + len(sq))), st, y, root, 0, 3
        )
        np.testing.assert_allclose(baseline, reindexed)

        reordered = get_hyp2_null(sq.iloc[::-1], st, y, root, 0, 3)
        assert not np.allclose(baseline, reordered)

    def test_length_mismatch_raises(self, perm_tree, preds):
        y, sq, st = preds
        root = perm_tree(nums=range(0, 2))
        with pytest.raises(ValueError):
            get_hyp2_null(sq.iloc[:-1], st, y, root, 0, 2)


######## prep_train_test_model ########
class TestPrepTrainTestModel:
    @pytest.mark.parametrize(
        "num_layers,expected",
        [(1, IndependentRegressor), (2, ResidualRegressor), (3, ResidualRegressor)],
    )
    def test_model_class_dispatch(
        self, prep_env, selected_params, shared_dict, num_layers, expected
    ):
        prep_env()
        *_, model = prep_train_test_model(
            selected_params=selected_params(num_layers=num_layers),
            impact_type="self",
            train_data_dir="train_dir",
            test_data_dir="test_dir",
            shared_dict=shared_dict,
        )
        assert isinstance(model, expected)

    def test_llm_model_and_temp_stripped_from_the_estimator(
        self, prep_env, selected_params, shared_dict
    ):
        """They select the data file but aren't estimator hyper-params; leaving
        them in raises TypeError at construction."""
        prep_env()
        *_, model = prep_train_test_model(
            selected_params=selected_params(),
            impact_type="self",
            train_data_dir="train_dir",
            test_data_dir="test_dir",
            shared_dict=shared_dict,
        )
        params = model.get_params()
        assert "llm_model" not in params and "temp" not in params

    @pytest.mark.parametrize("impact_type", ["self", "parent"])
    def test_quest_cols_follow_impact_type(
        self, prep_env, selected_params, shared_dict, impact_type
    ):
        prep_env()
        *_, model = prep_train_test_model(
            selected_params=selected_params(),
            impact_type=impact_type,
            train_data_dir="train_dir",
            test_data_dir="test_dir",
            shared_dict=shared_dict,
        )
        assert model.quest_cols == QUEST[impact_type]
        assert model.impact_type == impact_type

    def test_train_and_test_come_from_different_path_builders(
        self, prep_env, selected_params, shared_dict
    ):
        """get_df_path and get_test_df_path are separate functions; swapping them
        would score the training fold twice and inflate every test R^2."""
        seen = prep_env(train_rows=6, test_rows=4)
        X_train, _, X_test, _, _ = prep_train_test_model(
            selected_params=selected_params(),
            impact_type="self",
            train_data_dir="train_dir",
            test_data_dir="test_dir",
            shared_dict=shared_dict,
        )
        assert seen["train"]["df_dir"] == "train_dir"
        assert seen["test"]["df_dir"] == "test_dir"
        assert (X_train["split"] == "train").all()
        assert (X_test["split"] == "test").all()
        assert len(X_train) == 6 and len(X_test) == 4

    def test_path_builders_receive_the_file_selecting_params(
        self, prep_env, selected_params, shared_dict
    ):
        seen = prep_env()
        params = selected_params(llm_model="llama_7b", temp=0.7, agg_type="scale-new")
        prep_train_test_model(
            selected_params=params,
            impact_type="parent",
            train_data_dir="train_dir",
            test_data_dir="test_dir",
            shared_dict=shared_dict,
        )
        for kw in (seen["train"], seen["test"]):
            assert kw["llm_model"] == "llama_7b"
            assert kw["temp"] == 0.7
            assert kw["agg_type"] == "scale-new"
            assert kw["impact_type"] == "parent"

    def test_unknown_param_surfaces_as_typeerror(
        self, prep_env, selected_params, shared_dict
    ):
        """A typo'd key in BEST_PARAM_DICT reaches the constructor -- it should
        fail loudly here rather than being silently dropped."""
        prep_env()
        with pytest.raises(TypeError):
            # Raises in ResidualRegressor construction
            prep_train_test_model(
                selected_params=selected_params(alpha_2=1.0),
                impact_type="self",
                train_data_dir="train_dir",
                test_data_dir="test_dir",
                shared_dict=shared_dict,
            )
