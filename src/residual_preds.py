# residual_preds.py
from .config import SEED
from .utils import ALPHA_LIST
from .residual_func import get_lr_output
import pandas as pd
from pathlib import Path
import argparse
from sklearn.model_selection import StratifiedKFold
import numpy as np
from itertools import product


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
            if group not in val_counts.keys():
                raise ValueError(f"No instance of {group} in split: {split}")
            ratio_list_dict[group].append(val_counts[group])
    ## Check ratios are ~=
    for group, group_list in ratio_list_dict.items():
        diff = max(group_list) - min(group_list)
        assert (
            diff <= tol
        ), f"Potential error in splitting, group: {group} contains a difference of {diff}"
    if verbose != 0:
        print(f"\tSplits are all good!")


def export_dfs(combined_output_df, combined_perm_ids, combined_coef_df, save_dir):
    """
    Export dfs at specified dir. If file already exists, skip without error.
    """
    save_dir.mkdir(exist_ok=True, parents=True)
    pred_path = save_dir / "output_preds.parquet"
    id_path = save_dir / "perm_ids.parquet"
    coef_path = save_dir / "coefs.parquet"

    if pred_path.exists():
        print(f"skipping export OUTPUT PRED: already exists at {pred_path}")
    else:
        combined_output_df.to_parquet(pred_path)

    if id_path.exists():
        print(f"skipping export PERM ID: already exists at {id_path}")
    else:
        combined_perm_ids.to_parquet(id_path)

    if coef_path.exists():
        print(f"skipping export COEFS: already exists at {coef_path}")
    else:
        combined_coef_df.to_parquet(coef_path)


def loop_through_resid_params(
    *_,
    impact_type,
    agg_type,
    desc_seg_type,
    df_path,
    base_save_dir,
    n_splits,
    num_layers,
    perm_dir,
    linear_type="ridge",
    shuffle=None,
    alpha_list=ALPHA_LIST,
    max_iter=50000,
    seed=SEED,
    n_repeats=5,
    split_tol=0.1,
    verbose=0,
):
    if linear_type not in {"ridge"}:
        raise ValueError("linear_type must be 'ridge'")
    df = pd.read_parquet(df_path)

    def loop_through_splits(
        selected_alphas,
        df_w_splits,
        two_layer_base_val,
    ):
        perm_id_list = []
        output_df_list = []
        coef_df_list = []
        for split_id in range(n_splits):
            perm_id_df, output_df, coef_df = get_lr_output(  # unpack 3 values
                df=df_w_splits,
                agg_type=agg_type,
                desc_seg_type=desc_seg_type,
                num_layers=num_layers,
                selected_alphas=selected_alphas,
                split_id=split_id,
                impact_type=impact_type,
                max_iter=max_iter,
                seed=seed,
                perm_dir=perm_dir,
                two_layer_base=two_layer_base_val,
                linear_type=linear_type,
                hypothesis_shuffle=shuffle,
            )
            perm_id_list.append(perm_id_df)
            output_df_list.append(output_df)
            coef_df_list.append(coef_df)
        ## Aggregate
        combined_perm_ids = pd.concat(perm_id_list, axis=1)
        combined_output_df = pd.concat(output_df_list, axis=0)
        combined_coef_df = pd.concat(coef_df_list, axis=0, ignore_index=True)
        assert combined_perm_ids.shape[1] == 3
        assert combined_output_df["DAWBA_ID"].nunique() == len(combined_output_df)
        return combined_perm_ids, combined_output_df, combined_coef_df

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
        if num_layers == 1:
            for alpha_3 in alpha_list:
                selected_alphas = [None, alpha_3]
                export_dir = (
                    base_save_dir
                    / f"alpha_{alpha_3}"
                    / f"repeat_{repeat}"
                    / f"2_layer_base_None"
                )
                combined_perm_ids, combined_output_df, combined_coef_df = (
                    loop_through_splits(
                        selected_alphas=selected_alphas,
                        df_w_splits=df_w_splits,
                        two_layer_base_val=None,
                    )
                )
                export_dfs(
                    combined_perm_ids=combined_perm_ids,
                    combined_output_df=combined_output_df,
                    combined_coef_df=combined_coef_df,
                    save_dir=export_dir,
                )
        elif num_layers == 2:
            for two_layer_base_val in ["S", "Q"]:
                for alpha_3 in alpha_list:
                    selected_alphas = [None, alpha_3]
                    export_dir = (
                        base_save_dir
                        / f"alpha_{alpha_3}"
                        / f"repeat_{repeat}"
                        / f"2_layer_base_{two_layer_base_val}"
                    )
                    combined_perm_ids, combined_output_df, combined_coef_df = (
                        loop_through_splits(
                            selected_alphas=selected_alphas,
                            df_w_splits=df_w_splits,
                            two_layer_base_val=two_layer_base_val,
                        )
                    )
                    export_dfs(
                        combined_perm_ids=combined_perm_ids,
                        combined_output_df=combined_output_df,
                        combined_coef_df=combined_coef_df,
                        save_dir=export_dir,
                    )
        else:  # num_layers == 3
            for alpha_2, alpha_3 in product(alpha_list, alpha_list):
                selected_alphas = [alpha_2, alpha_3]
                export_dir = (
                    base_save_dir
                    / f"alpha_{alpha_2}_{alpha_3}"
                    / f"repeat_{repeat}"
                    / f"2_layer_base_None"
                )
                combined_perm_ids, combined_output_df, combined_coef_df = (
                    loop_through_splits(
                        selected_alphas=selected_alphas,
                        df_w_splits=df_w_splits,
                        two_layer_base_val=None,
                    )
                )
                export_dfs(
                    combined_perm_ids=combined_perm_ids,
                    combined_output_df=combined_output_df,
                    combined_coef_df=combined_coef_df,
                    save_dir=export_dir,
                )
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        prog="residual_preds",
        description="For each alpha, get ridge regression results for each fold",
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
        choices=["block", "avg", "max", "binary", "proba-limited"],
    )
    parser.add_argument(
        "--desc_seg_type",
        required=True,
        choices=["pt_noshort", "turns", "expanded", "original", "combined"],
    )
    parser.add_argument("--df_path", required=True, type=Path)
    parser.add_argument("--base_save_dir", required=True, type=Path)
    parser.add_argument(
        "--n_splits",
        required=False,
        type=int,
        help="Optional integer determining how many folds to use for CV, defaults to 3",
        default=3,
    )
    parser.add_argument(
        "--num_layers", required=False, type=int, default=3, choices=[1, 2, 3]
    )
    parser.add_argument("--perm_dir", required=True, type=Path)
    parser.add_argument("--linear_type", required=True, choices=["elastic", "ridge"])
    parser.add_argument("--shuffle", required=True, choices=["text", "none"])
    parser.add_argument("--alpha_list", required=False, nargs="+", default=ALPHA_LIST)
    parser.add_argument(
        "--max_iter",
        required=False,
        type=int,
        help=f"Optional integer determining maximum iterations for Logistic Regression, defaults to 5000",
        default=5000,
    )
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
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)
    alpha_list = (
        None if args.alpha_list == ["None"] else [float(a) for a in args.alpha_list]
    )
    ## Perm shuffle str
    if args.shuffle == "none":
        shuffle = None
    else:
        shuffle = args.shuffle
    loop_through_resid_params(
        impact_type=args.impact_type,
        agg_type=args.agg_type,
        desc_seg_type=args.desc_seg_type,
        df_path=args.df_path,
        base_save_dir=args.base_save_dir,
        n_splits=args.n_splits,
        num_layers=args.num_layers,
        perm_dir=args.perm_dir,
        linear_type=args.linear_type,
        shuffle=shuffle,
        alpha_list=alpha_list,
        max_iter=args.max_iter,
        seed=args.seed,
        n_repeats=args.n_repeats,
        split_tol=args.split_tol,
        verbose=0,
    )


if __name__ == "__main__":
    main()
