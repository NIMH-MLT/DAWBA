# residual_preds.py
from .config import SEED, BASE_PATH
from .utils import L1_LIST, ALPHA_LIST
from .residual_func import get_lr_output
import json
import ast
import pandas as pd
from pathlib import Path
import argparse
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
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
    linear_type,
    perm_dir,
    pca_k_list,
    pca_scale,
    two_layer_base_val=None,
    shuffle=None,
    impactme_outcome=None,
    alpha_list=ALPHA_LIST,
    l1_list=L1_LIST,
    max_iter=50000,
    seed=SEED,
    n_repeats=5,
    split_tol=0.1,
    n_jobs=1,
    verbose=0,
):
    if linear_type not in {"elastic", "ridge"}:
        raise ValueError("linear_type must be either 'elastic' or 'ridge'")

    df = pd.read_parquet(df_path)

    # Ridge never sweeps l1_ratio.
    l1_values = l1_list if linear_type == "elastic" else [None]

    def loop_through_splits(
        selected_alphas, selected_l1s, df_w_splits, pca_k, two_layer_base_val=None
    ):
        perm_id_list = []
        output_df_list = []
        coef_df_list = []
        for split_id in range(n_splits):
            perm_id_df, output_df, coef_df = get_lr_output(  # unpack 3 values
                df=df_w_splits,
                num_layers=num_layers,
                desc_seg_type=desc_seg_type,
                selected_alphas=selected_alphas,
                selected_l1s=selected_l1s,
                split_id=split_id,
                impact_type=impact_type,
                impactme_outcome=impactme_outcome,
                two_layer_base=two_layer_base_val,
                max_iter=max_iter,
                seed=seed,
                agg_type=agg_type,
                linear_type=linear_type,
                perm_dir=perm_dir,
                pca_k=pca_k,
                pca_scale=pca_scale,
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
        if impactme_outcome is not None:
            # ImpactMe outcome iteration
            pass
            # TODO: remove?
        else:  # Regular iteration
            for pca_k in pca_k_list:
                if num_layers == 1:
                    two_layer_base_val = None
                    for l1_3 in l1_values:
                        for alpha_3 in alpha_list:
                            selected_alphas = [None, alpha_3]
                            selected_l1s = [None, l1_3]
                            export_dir = (
                                base_save_dir
                                / f"alpha_{alpha_3}"
                                / f"l1_{l1_3}"
                                / f"repeat_{repeat}"
                                / f"2_layer_base_{two_layer_base_val}"
                                / f"select_{pca_k}_{pca_scale}"
                            )
                            combined_perm_ids, combined_output_df, combined_coef_df = (
                                loop_through_splits(
                                    selected_alphas,
                                    selected_l1s,
                                    df_w_splits,
                                    pca_k,
                                    two_layer_base_val,
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
                        for l1_3 in l1_values:
                            for alpha_3 in alpha_list:
                                selected_alphas = [None, alpha_3]
                                selected_l1s = [None, l1_3]
                                export_dir = (
                                    base_save_dir
                                    / f"alpha_{alpha_3}"
                                    / f"l1_{l1_3}"
                                    / f"repeat_{repeat}"
                                    / f"2_layer_base_{two_layer_base_val}"
                                    / f"select_{pca_k}_{pca_scale}"
                                )
                                (
                                    combined_perm_ids,
                                    combined_output_df,
                                    combined_coef_df,
                                ) = loop_through_splits(
                                    selected_alphas,
                                    selected_l1s,
                                    df_w_splits,
                                    pca_k,
                                    two_layer_base_val,
                                )
                                export_dfs(
                                    combined_perm_ids=combined_perm_ids,
                                    combined_output_df=combined_output_df,
                                    combined_coef_df=combined_coef_df,
                                    save_dir=export_dir,
                                )
                else:  # num_layers == 3
                    two_layer_base_val = None
                    for l1_2, l1_3 in product(l1_values, l1_values):
                        for alpha_2, alpha_3 in product(alpha_list, alpha_list):
                            selected_alphas = [alpha_2, alpha_3]
                            selected_l1s = [l1_2, l1_3]
                            export_dir = (
                                base_save_dir
                                / f"alpha_{alpha_2}_{alpha_3}"
                                / f"l1_{l1_2}_{l1_3}"
                                / f"repeat_{repeat}"
                                / f"2_layer_base_{two_layer_base_val}"
                                / f"select_{pca_k}_{pca_scale}"
                            )
                            combined_perm_ids, combined_output_df, combined_coef_df = (
                                loop_through_splits(
                                    selected_alphas,
                                    selected_l1s,
                                    df_w_splits,
                                    pca_k,
                                    two_layer_base_val,
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
    parser.add_argument(
        "--desc_seg_type",
        required=True,
        choices=["pt_noshort", "turns", "new", "expanded", "original", "combined"],
    )

    parser.add_argument("--df_path", required=True, type=Path)
    parser.add_argument("--base_save_dir", required=True, type=Path)
    parser.add_argument("--alpha_list", required=False, nargs="+", default=ALPHA_LIST)
    parser.add_argument("--l1_list", required=False, nargs="+", default=L1_LIST)
    parser.add_argument("--perm_dir", required=True, type=Path)
    parser.add_argument("--linear_type", required=True, choices=["elastic", "ridge"])
    parser.add_argument("--shuffle", required=True, choices=["text", "none"])
    parser.add_argument(
        "--metric_agg", required=True, choices=["per_fold", "combined_folds"]
    )
    parser.add_argument(
        "--impactme_outcome_list", required=False, default=None, nargs="+"
    )
    parser.add_argument(
        "--two_layer_base", required=False, choices=["S", "Q"], default=None
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
    parser.add_argument(
        "--n_splits",
        required=False,
        type=int,
        help="Optional integer determining how many folds to use for CV, defaults to 4",
        default=3,
    )
    parser.add_argument(
        "--num_layers", required=False, type=int, default=3, choices=[1, 2, 3]
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
    parser.add_argument(
        "--tune",
        required=False,
        choices=["True", "False"],
        help="Whether or not to get best params",
        default="False",
    )
    parser.add_argument("--pca_k_list", required=True, type=ast.literal_eval)
    parser.add_argument(
        "--pca_scale", required=True, type=str, choices=["True", "False"]
    )
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)
    ## PCA args
    pca_scale = True if args.pca_scale == "True" else False
    ## Lists
    alpha_list = (
        None if args.alpha_list == ["None"] else [float(a) for a in args.alpha_list]
    )
    l1_list = None if args.l1_list == ["None"] else [float(l) for l in args.l1_list]
    impactme_outcome_list = (
        None if args.impactme_outcome_list == ["None"] else args.impactme_outcome_list
    )
    ## Perm shuffle str
    if args.shuffle == "none":
        shuffle = None
    else:
        shuffle = args.shuffle
    shared_args = dict(
        impact_type=args.impact_type,
        agg_type=args.agg_type,
        desc_seg_type=args.desc_seg_type,
        df_path=args.df_path,
        base_save_dir=args.base_save_dir,
        num_layers=args.num_layers,
        alpha_list=alpha_list,
        l1_list=l1_list,
        perm_dir=args.perm_dir,
        linear_type=args.linear_type,
        two_layer_base=args.two_layer_base,
        max_iter=args.max_iter,
        seed=args.seed,
        n_repeats=args.n_repeats,
        split_tol=args.split_tol,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        shuffle=shuffle,
        pca_k_list=args.pca_k_list,
        pca_scale=pca_scale,
    )
    if impactme_outcome_list is None:
        # This runs to get raw preds on each param combo
        loop_through_resid_params(
            **shared_args,
            impactme_outcome=None,
        )
    else:
        pass
    # TODO: Do we even need this?
    # print(
    #     f"\t Running per outcome for agg: {args.agg_type}, impact: {args.impact_type}..."
    # )
    # # This runs to get raw preds w/ each impactme outcome as sole predictor
    # for outcome_name in impactme_outcome_list:
    #     loop_through_resid_params(
    #         **shared_args,
    #         pca_k=pca_k,
    #         pca_scale=pca_scale,
    #         impactme_outcome=outcome_name,
    #     )
    ## TODO: Remove?
    if args.tune == "True":
        pass
        if impactme_outcome_list is None:
            pass
        else:
            pass


if __name__ == "__main__":
    main()
