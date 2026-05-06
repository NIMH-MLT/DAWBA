# test_perm.py
import pytest
import pandas as pd
import numpy as np
from src.config import SEED
from src.perm import create_perms, format_folder, write_dirs


class TestFormatFolder:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, "perm_000"),
            (9, "perm_009"),
            (10, "perm_010"),
            (99, "perm_099"),
            (100, "perm_100"),
        ],
    )
    def test_format(self, n, expected):
        assert format_folder(n) == expected


class TestWriteDirs:
    def test_creates_expected_dirs(self, base_df_no_splits, tmp_path):
        write_dirs(
            start_perm=0,
            end_perm=3,
            parent_df=base_df_no_splits,
            self_df=base_df_no_splits,
            save_dir=tmp_path,
            start_seed=SEED,
            add_baseline=False,
        )
        for n in range(3):
            assert (tmp_path / format_folder(n)).exists()

    def test_creates_seed_files(self, base_df_no_splits, tmp_path):
        write_dirs(
            start_perm=0,
            end_perm=3,
            parent_df=base_df_no_splits,
            self_df=base_df_no_splits,
            save_dir=tmp_path,
            start_seed=SEED,
            add_baseline=False,
        )
        for n in range(3):
            seed = int((tmp_path / format_folder(n) / "seed.txt").read_text())
            assert seed == SEED + n

    def test_creates_baseline_when_requested(self, base_df_no_splits, tmp_path):
        write_dirs(
            start_perm=0,
            end_perm=1,
            parent_df=base_df_no_splits,
            self_df=base_df_no_splits,
            save_dir=tmp_path,
            start_seed=SEED,
            add_baseline=True,
        )
        assert (tmp_path / "baseline").exists()

    def test_no_baseline_when_not_requested(self, base_df_no_splits, tmp_path):
        write_dirs(
            start_perm=0,
            end_perm=1,
            parent_df=base_df_no_splits,
            self_df=base_df_no_splits,
            save_dir=tmp_path,
            start_seed=SEED,
            add_baseline=False,
        )
        assert not (tmp_path / "baseline").exists()

    def test_raises_if_dir_exists(self, base_df_no_splits, tmp_path):
        write_dirs(
            start_perm=0,
            end_perm=1,
            parent_df=base_df_no_splits,
            self_df=base_df_no_splits,
            save_dir=tmp_path,
            start_seed=SEED,
            add_baseline=False,
        )
        with pytest.raises(FileExistsError):
            write_dirs(
                start_perm=0,
                end_perm=1,
                parent_df=base_df_no_splits,
                self_df=base_df_no_splits,
                save_dir=tmp_path,
                start_seed=SEED,
                add_baseline=False,
            )

    def test_partial_range(self, base_df_no_splits, tmp_path):
        """start_perm/end_perm should only create dirs in that range."""
        write_dirs(
            start_perm=2,
            end_perm=5,
            parent_df=base_df_no_splits,
            self_df=base_df_no_splits,
            save_dir=tmp_path,
            start_seed=SEED,
            add_baseline=False,
        )
        for n in range(2, 5):
            assert (tmp_path / format_folder(n)).exists()
        for n in [0, 1, 5]:
            assert not (tmp_path / format_folder(n)).exists()


class TestCreatePerms:
    def test_returns_correct_length(self, base_df_no_splits):
        perm = create_perms(SEED, df_to_permute=base_df_no_splits)
        assert len(perm) == len(base_df_no_splits)

    def test_returns_valid_dawba_ids(self, base_df_no_splits):
        perm = create_perms(SEED, df_to_permute=base_df_no_splits)
        assert set(perm) == set(base_df_no_splits["DAWBA_ID"].tolist())

    def test_within_groups(self, base_df_no_splits):
        """Each subject should only be mapped to a DAWBA_ID from within their group."""
        perm = create_perms(SEED, df_to_permute=base_df_no_splits)
        id_to_group = base_df_no_splits.set_index("DAWBA_ID")[
            "eligible_group"
        ].to_dict()
        orig_ids = base_df_no_splits["DAWBA_ID"].tolist()
        for orig_id, perm_id in zip(orig_ids, perm):
            assert (
                id_to_group[orig_id] == id_to_group[perm_id]
            ), f"ID {orig_id} (group {id_to_group[orig_id]}) mapped to {perm_id} (group {id_to_group[perm_id]})"

    def test_reproducible_with_same_seed(self, base_df_no_splits):
        p1 = create_perms(SEED, df_to_permute=base_df_no_splits)
        p2 = create_perms(SEED, df_to_permute=base_df_no_splits)
        assert p1 == p2

    def test_different_seeds_give_different_results(self, base_df_no_splits):
        p1 = create_perms(SEED, df_to_permute=base_df_no_splits)
        p2 = create_perms(SEED + 1, df_to_permute=base_df_no_splits)
        assert p1 != p2

    def test_is_actually_shuffled(self, base_df_no_splits):
        """Permuted IDs should differ from original ordering."""
        perm = create_perms(SEED, df_to_permute=base_df_no_splits)
        assert perm != base_df_no_splits["DAWBA_ID"].tolist()
