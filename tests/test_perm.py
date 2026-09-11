# test_perm.py
# Run w/ uv run pytest tests/test_perm.py -v
# Written in collab w/ Claude Opus 5

import numpy as np
import pandas as pd
import pytest

import src.Regressor as reg_mod
from src.perm import create_perms
from src.Regressor import BaseRegressor

PERM_SEED = 7


######## Fixtures ########
@pytest.fixture
def make_perm_df():
    """DAWBA_ID deliberately unsorted and groups deliberately interleaved: any
    code path that assumes row order == ID order or == group order shows up as a
    failure rather than as a coincidentally-correct permutation."""

    def _make(groups=("a", "b", "a", "b", "a", "b"), ids=(5, 3, 1, 4, 2, 6)):
        return pd.DataFrame({"DAWBA_ID": list(ids), "eligible_group": list(groups)})

    return _make


@pytest.fixture
def group_of():
    def _map(df):
        return dict(zip(df["DAWBA_ID"], df["eligible_group"]))

    return _map


@pytest.fixture
def regressor():
    """shuffle_vals only touches create_perms and get_perm_seed, so the estimator
    hyper-params are irrelevant -- btu they're required by __init__."""
    return BaseRegressor(
        num_layers=3,
        impact_type="self",
        desc_seg_type="original",
        quest_cols=["s_dep_q1"],
        agg_type="scale-new",
        alpha2=15.0,
        alpha3=75.0,
        seed=0,
        linear_type="ridge",
    )


@pytest.fixture(autouse=True)
def stub_seed(monkeypatch):
    """get_perm_seed reads seed.txt out of a perm_xxx dir; the permutation logic
    is what's under test, not the file read."""
    monkeypatch.setattr(reg_mod, "get_perm_seed", lambda perm_dir: PERM_SEED)


@pytest.fixture
def feats():
    """One column whose value encodes the donor's DAWBA_ID (id * 10), so every
    output row can be traced back to the participant it came from."""

    def _make(df):
        return (df["DAWBA_ID"].to_numpy(dtype=float) * 10.0).reshape(-1, 1)

    return _make


######## create_perms ########
class TestCreatePerms:
    @pytest.mark.parametrize(
        "groups",
        [
            ("a", "b", "a", "b", "a", "b"),  # balanced, interleaved
            ("a", "a", "a", "b", "b", "b"),  # balanced, blocked
            ("a", "b", "b", "b", "b", "b"),  # unbalanced, one singleton
        ],
    )
    def test_is_a_within_group_permutation(self, make_perm_df, group_of, groups):
        """The whole hypothesis test rests on this: every ID is reused exactly
        once (so the marginal distribution of text features is untouched) and
        never crosses a group boundary (exchangeability)."""
        df = make_perm_df(groups=groups)
        perm = create_perms(perm_seed=PERM_SEED, df_to_permute=df)
        g = group_of(df)

        assert sorted(perm) == sorted(df["DAWBA_ID"])
        assert all(g[donor] == g[row] for row, donor in zip(df["DAWBA_ID"], perm))

    def test_labels_are_positional_not_group_ordered(self, make_perm_df, group_of):
        """create_perms writes into a Series by .loc after a groupby(sort=True),
        so groups are visited in sorted order. The returned list must still be in
        df row order -- callers zip it against un-grouped rows."""
        df = make_perm_df().set_index(np.array([10, 3, 7, 1, 9, 5]))
        perm = create_perms(perm_seed=PERM_SEED, df_to_permute=df)
        g = group_of(df)

        assert all(g[donor] == g[row] for row, donor in zip(df["DAWBA_ID"], perm))

    def test_seed_fully_determines_the_permutation(self, make_perm_df):
        """Each perm_xxx dir is a seed, and the null distribution is only
        reproducible if the seed is the sole source of randomness."""
        df = make_perm_df()
        assert create_perms(PERM_SEED, df) == create_perms(PERM_SEED, df)
        assert any(
            create_perms(PERM_SEED, df) != create_perms(PERM_SEED + k, df)
            for k in range(1, 6)
        )

    def test_singleton_group_is_an_identity(self, make_perm_df):
        """A group of size 1 can only map to itself, so it contributes no
        randomness to the null. Asserted so that a stratification scheme that
        drifts toward many tiny groups fails loudly here."""
        df = make_perm_df(groups=("solo", "b", "b", "b", "b", "b"))
        perm = create_perms(perm_seed=PERM_SEED, df_to_permute=df)
        assert perm[0] == df["DAWBA_ID"].iloc[0]

    def test_missing_group_label_raises(self, make_perm_df):
        """groupby drops NaN keys, so a participant with no eligible_group is
        silently left unassigned -- raise value error"""
        df = make_perm_df()
        df.loc[2, "eligible_group"] = np.nan
        with pytest.raises(ValueError):
            _ = create_perms(perm_seed=PERM_SEED, df_to_permute=df)

    def test_labels_keep_the_id_dtype(self, make_perm_df):
        """pd.Series(index=..., dtype='int64') with no data comes back float64,
        so integer DAWBA_IDs are returned as floats and any downstream merge or
        equality check on IDs is comparing across dtypes."""
        df = make_perm_df()
        perm = create_perms(perm_seed=PERM_SEED, df_to_permute=df)
        assert pd.Series(perm).dtype == df["DAWBA_ID"].dtype


######## BaseRegressor.shuffle_vals ########
class TestShuffleVals:
    def test_features_follow_perm_ids(self, regressor, make_perm_df, feats):
        """Self-consistency of the reindex step: row i must hold the features of
        perm_ids[i], with no NaN rows introduced by an ID that isn't in the
        frame's index."""
        df = make_perm_df()
        out = regressor.shuffle_vals(df, feats(df), perm_dir=None)

        assert out.shape == (len(df), 1)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(
            out.ravel(), np.asarray(regressor.perm_ids, dtype=float) * 10.0
        )

    @pytest.mark.parametrize("ids", [(1, 2, 3, 4, 5, 6), (5, 3, 1, 4, 2, 6)])
    def test_donor_shares_group_with_the_row_it_lands_on(
        self, regressor, make_perm_df, group_of, feats, ids
    ):
        """shuffle_vals sorts X by DAWBA_ID before permuting but returns rows in
        that sorted order, while fit() leaves y and the sum/questionnaire blocks
        in the original row order. If X arrives unsorted, row i of the shuffled
        text block belongs to a different participant than y[i] -- which breaks
        the within-group constraint the permutation just enforced."""
        df = make_perm_df(ids=ids)
        out = regressor.shuffle_vals(df, feats(df), perm_dir=None).ravel()
        g = group_of(df)

        donors = np.round(out / 10.0).astype(int)
        assert all(g[d] == g[row] for d, row in zip(donors, df["DAWBA_ID"]))

    def test_duplicate_ids_are_rejected(self, regressor, make_perm_df, feats):
        """DAWBA_ID is the join key for the whole permutation; a duplicate makes
        the reindex ambiguous. Confirming it raises rather than fanning rows out
        into a longer array."""
        df = make_perm_df(ids=(5, 3, 1, 4, 2, 5))
        with pytest.raises(ValueError):
            regressor.shuffle_vals(df, feats(df), perm_dir=None)
