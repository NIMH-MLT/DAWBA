# elastic_precs.py
from .config import SEED
from .utils import L1_LIST
from .elastic_func import get_lr_output

import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
import argparse
from sklearn.model_selection import StratifiedKFold
import numpy as np


def add_split_col(df, pred_type, n_splits, tol=0.1, random_state=SEED, verbose=0):
    df = df.copy()
    if verbose != 0:
        print(f"Pred type: {pred_type}")
    y = df["eligible_group"]
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    folds_positional = [tst for trn, tst in skf.split(np.zeros(len(y)), y)]
    ## Make col with split ID
    split_col_name = "split_id"
    df[split_col_name] = -1
    df = df.reset_index(drop=True)
    for split_id, split_indices in enumerate(folds_positional):
        df.iloc[split_indices, df.columns.get_loc(split_col_name)] = split_id
    ## Ensure splits look good + explore outcome var
    check_ratios(
        df_w_splits=df, pred_type=pred_type, tol=tol, verbose=verbose, n_splits=n_splits
    )
    return df


def check_ratios(df_w_splits, pred_type, n_splits, tol=0.1, verbose=0):
    ratio_list_dict = {k: [] for k in df_w_splits["eligible_group"].unique()}
    ## Populate dict of lists w/ normalized ratio of eligible group
    for split in range(n_splits):
        sub_df = df_w_splits[df_w_splits["split_id"] == split]
        if verbose != 0:
            print(
                f"\tSplit {split+1} avg score: {np.mean(sub_df[f'impact_score_{pred_type}'])}"
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


def loop_through_l1(
    *_,
    impact_type,
    agg_type,
    alpha_val,
    df_path,
    base_save_dir,
    n_splits,
    max_iter=50000,
    seed=SEED,
    n_repeats=5,
    split_tol=0.1,
    n_jobs=1,
    verbose=0,
):
    joblist = []
    df = pd.read_parquet(df_path)
    for repeat in range(n_repeats):
        random_state_for_split = seed + repeat
        df_w_splits = add_split_col(
            df=df,
            pred_type=impact_type,
            tol=split_tol,
            random_state=random_state_for_split,
            n_splits=n_splits,
            verbose=verbose,
        )
        for l1_ratio in L1_LIST:
            for split_id in range(n_splits):
                export_path = (
                    base_save_dir
                    / f"l1_{l1_ratio}"
                    / f"repeat_{repeat}"
                    / f"split_{split_id}.parquet"
                )
                joblist.append(
                    delayed(get_lr_output)(
                        df=df_w_splits,
                        alpha_val=alpha_val,
                        l1_val=l1_ratio,
                        split_id=split_id,
                        impact_type=impact_type,
                        export_path=export_path,
                        max_iter=max_iter,
                        seed=seed,
                        agg_type=agg_type,
                    )
                )
    ## Run in LR call for each split/c-val combo in parallel
    _ = Parallel(n_jobs=n_jobs, verbose=verbose)(joblist)
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        prog="elastic_preds",
        description="For each alpha and l1 value, get elasticnet regression results for each fold",
    )
    parser.add_argument(
        "--impact_type",
        required=True,
        help="One of ['self', 'parent']",
        choices=["self", "parent"],
    )
    parser.add_argument(
        "--agg_type",
        required=True,
        choices=["reduced", "block", "avg", "max", "binary", "proba-limited"],
    )
    parser.add_argument("--alpha_val", required=True, type=float)
    parser.add_argument("--df_path", required=True)
    parser.add_argument("--base_save_dir", required=True)
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        help=f"Optional integer determining random state, defaults to {SEED}",
        default=SEED,
    )
    parser.add_argument(
        "--n_repeats",
        required=False,
        type=int,
        help="Optional integer determining number of times to repeat CV, defaults to 5. Must be >0",
        default=5,
    )
    parser.add_argument(
        "--split_tol",
        required=False,
        type=float,
        help="Optional float determining how close eligible_group ratios must be between splits, defaults to 0.1",
        default=0.1,
    )
    parser.add_argument(
        "--n_splits",
        required=False,
        type=int,
        help="Optional integer determining how many folds to use for CV, defaults to 4",
        default=3,
    )
    parser.add_argument(
        "--n_jobs",
        required=False,
        type=int,
        help="Optional integer determining number of regression jobs to run in parallel, defaults to 1",
        default=1,
    )
    parser.add_argument(
        "--max_iter",
        required=False,
        type=int,
        help=f"Optional integer determining maximum iterations for Logistic Regression, defaults to 5000",
        default=5000,
    )
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)
    loop_through_l1(
        impact_type=args.impact_type,
        agg_type=args.agg_type,
        alpha_val=args.alpha_val,
        df_path=Path(args.df_path),
        base_save_dir=Path(args.base_save_dir),
        max_iter=args.max_iter,
        seed=args.seed,
        n_repeats=args.n_repeats,
        split_tol=args.split_tol,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
