from sklearn.model_selection import StratifiedKFold
from .config import SEED
import numpy as np


def add_split_col(df, pred_type, n_splits=4, tol=0.1, random_state=SEED, verbose=0):
    if verbose != 0:
        print(f"Pred type: {pred_type}")
    y = df["eligible_group"]
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    folds_positional = [tst for trn, tst in skf.split(np.zeros(len(y)), y)]
    ## Make col with split ID
    split_col_name = "split_id"
    df[split_col_name] = -1
    for split_id, split_indices in enumerate(folds_positional):
        df.loc[split_indices, split_col_name] = split_id
    ## Ensure splits look good + explore outcome var
    check_ratios(df, pred_type, tol, verbose)
    return df


def check_ratios(df_w_splits, pred_type, tol=0.1, verbose=0):
    ratio_list_dict = {k: [] for k in df_w_splits["eligible_group"].unique()}
    ## Populate dict of lists w/ normalized ratio of eligible group
    for split in range(4):
        sub_df = df_w_splits[df_w_splits["split_id"] == split]
        if verbose != 0:
            print(
                f"\tSplit {split+1} avg score: {np.mean(sub_df[f'Impact_Score_{pred_type}'])}"
            )
        val_counts = sub_df["eligible_group"].value_counts(normalize=True)
        for group in ratio_list_dict.keys():
            ratio_list_dict[group].append(val_counts[group])
    ## Check ratios are ~=
    for group, group_list in ratio_list_dict.items():
        diff = max(group_list) - min(group_list)
        assert (
            diff <= tol
        ), f"Potential error in splitting, group: {group} contains a difference of {diff}"
    if verbose != 0:
        print(f"\tSplits are all good!")
