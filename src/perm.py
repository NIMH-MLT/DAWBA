# perm.py
from src.config import BASE_PATH
import pandas as pd
import numpy as np


def ensure_dfs_match(loop_dir_imp, loop_dir_shot, impact_type):
    """
    - Ensures all dfs for a given impact type share the same
        index & DAWBA_ID ordering
    - This assumption is implicitly made later in perm process, as
        each impact type undergoes a global (grouped) imputation
        that is shared across all hyper-params. Including:
        model type (Llama 7B,80B), segmentation (turns, pt_noshort),
        description (original, augment, negative)
    Params
    ------
    loop_dir_imp: pathlib Path
        Directory holding dfs used for avg, max, & block aggregations
        All use ImpactMe LR models
    loop_dir_shot: pathlib Path
        Directory holding dfs used for binary, proba-limited aggregations
        A
    impact_type: str
        One of ['self', 'parent']

    Returns
    -------
    True if all dfs consistent within the given impact type
    False otherwise
    """
    all_dfs = []
    for model_dir in loop_dir_imp.iterdir():
        for seg_dir in model_dir.iterdir():
            file_path = seg_dir / f"{impact_type}.parquet"
            file = pd.read_parquet(file_path)
            assert (file.index == range(len(file))).all()
            all_dfs.append(file)

    for shot_dir in loop_dir_shot.iterdir():
        for model_dir in shot_dir.iterdir():
            for desc_dir in model_dir.iterdir():
                file_path = desc_dir / f"{impact_type}.parquet"
                file = pd.read_parquet(file_path)
                assert (file.index == range(len(file))).all()
                all_dfs.append(file)
    ## Ensure all DAWBA_IDs match
    ref = all_dfs[0]["DAWBA_ID"]
    return all(df["DAWBA_ID"].equals(ref) for df in all_dfs[1:])


def create_perms(perm_seed, df_to_permute, group_col="eligible_group"):
    """
    Permutes DAWBA_ID within `group_col` for df_to_permute

    Params
    -----
    perm_seed: int
        Seed used for permutation
    df_to_permute: pandas DF
    group_col: str
        Name of column used for grouping during permutation
        Essential for exchangability

    Returns
    -------
    perm_labels: list[int] of shuffled DAWBA_IDS
    """
    ## Permute
    rng = np.random.default_rng(seed=perm_seed)
    out = pd.Series(index=df_to_permute.index, dtype=df_to_permute["DAWBA_ID"].dtype)
    for _, idx in df_to_permute.groupby(group_col, sort=True).groups.items():
        out.loc[idx] = rng.permutation(df_to_permute.loc[idx, "DAWBA_ID"].to_numpy())
    perm_labels = out.to_list()
    return perm_labels


def format_folder(n):
    """
    Return permutation iteration in 3-digit format (`perm_xxx`)
    """
    if n < 10:
        return f"perm_00{n}"
    elif n < 100:
        return f"perm_0{n}"
    else:
        return f"perm_{n}"


def write_dirs(start_perm, end_perm, save_dir, start_seed, add_baseline):
    """
    For each of num_perm permutations:
        - ensure permutation dir does not exist
        - write unique random seed to seed.txt
        - generate permutations for self & parent and write
            shuffled DAWBA IDs to perm_self and perm_parent.txt
    Also create dir for base predictions (without permutation)

    Params
    ------
    num_perm: int
        Number of permutations
        num_perm + 1 folders will be created
    save_dir: pathlib Path
        Directory where permutation folders are written to
    start_seed: int
        Starting seed for permutation seeds
        A permutation iteration's random seed is:
        start_seed + iteration

    Returns nothing
    """
    save_dir.mkdir(exist_ok=True, parents=True)
    for n in range(start_perm, end_perm):
        perm_dir_name = format_folder(n)
        perm_dir = save_dir / perm_dir_name
        perm_dir.mkdir(exist_ok=False, parents=True)
        ## write seed
        perm_seed = start_seed + n
        with open(perm_dir / "seed.txt", "w") as f:
            f.write(str(perm_seed))
    if add_baseline:
        ## make dir for baseline w/o seed or perm txt files
        (save_dir / "baseline").mkdir(exist_ok=False)
