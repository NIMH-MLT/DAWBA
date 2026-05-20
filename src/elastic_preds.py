# elastic_precs.py
from .config import SEED, BASE_PATH
from .utils import L1_LIST, ALPHA_LIST
from .elastic_func import get_lr_output
import json
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
import argparse
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm


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
    df_path,
    base_save_dir,
    n_splits,
    num_layers,
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
    """
    NOTE: For reg/impactme outcome iterations & 2 vs 3 layer iterations, elasticNet is needed on:

      | Reg    | Outcome
    --|--------|-------
    2L|  L3    |  NONE
    3L|  L2,L3 |  L2
    """
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
        for split_id in range(n_splits):
            if (
                impactme_outcome is not None
            ):  # ImpactMe Outcome iteration --> Only 3 layer has elastic
                if num_layers == 2:  # 2 layers
                    selected_alphas = [None, None]
                    selected_l1s = [None, None]
                    export_path = (
                        base_save_dir
                        / impactme_outcome
                        / f"repeat_{repeat}"
                        / f"split_{split_id}.parquet"
                    )
                    joblist.append(
                        delayed(get_lr_output)(
                            df=df_w_splits,
                            num_layers=num_layers,
                            selected_alphas=selected_alphas,
                            selected_l1s=selected_l1s,
                            split_id=split_id,
                            impact_type=impact_type,
                            impactme_outcome=impactme_outcome,
                            export_path=export_path,
                            max_iter=max_iter,
                            seed=seed,
                            agg_type=agg_type,
                        )
                    )
                else:  # 3layers
                    for alpha_2 in alpha_list:
                        for l1_ratio_2 in l1_list:
                            selected_alphas = [alpha_2, None]
                            selected_l1s = [l1_ratio_2, None]
                            export_path = (
                                base_save_dir
                                / impactme_outcome
                                / f"alpha_{alpha_2}"
                                / f"l1_{l1_ratio_2}"
                                / f"repeat_{repeat}"
                                / f"split_{split_id}.parquet"
                            )
                            joblist.append(
                                delayed(get_lr_output)(
                                    df=df_w_splits,
                                    num_layers=num_layers,
                                    selected_alphas=selected_alphas,
                                    selected_l1s=selected_l1s,
                                    split_id=split_id,
                                    impact_type=impact_type,
                                    impactme_outcome=impactme_outcome,
                                    export_path=export_path,
                                    max_iter=max_iter,
                                    seed=seed,
                                    agg_type=agg_type,
                                )
                            )
            else:  # Regular iteration --> both 2/3 layers have elastic on L3
                for alpha_3 in alpha_list:
                    for l1_ratio_3 in l1_list:
                        if num_layers == 2:  # 2 layers
                            selected_alphas = [None, alpha_3]
                            selected_l1s = [None, l1_ratio_3]
                            export_path = (
                                base_save_dir
                                / f"alpha_{alpha_3}"
                                / f"l1_{l1_ratio_3}"
                                / f"repeat_{repeat}"
                                / f"split_{split_id}.parquet"
                            )
                            joblist.append(
                                delayed(get_lr_output)(
                                    df=df_w_splits,
                                    num_layers=num_layers,
                                    selected_alphas=selected_alphas,
                                    selected_l1s=selected_l1s,
                                    split_id=split_id,
                                    impact_type=impact_type,
                                    impactme_outcome=impactme_outcome,
                                    export_path=export_path,
                                    max_iter=max_iter,
                                    seed=seed,
                                    agg_type=agg_type,
                                )
                            )
                        else:  # 3 layers
                            for alpha_2 in alpha_list:
                                for l1_ratio_2 in l1_list:
                                    selected_alphas = [alpha_2, alpha_3]
                                    selected_l1s = [l1_ratio_2, l1_ratio_3]
                                    export_path = (
                                        base_save_dir
                                        / f"alpha_{alpha_2}_{alpha_3}"
                                        / f"l1_{l1_ratio_2}_{l1_ratio_3}"
                                        / f"repeat_{repeat}"
                                        / f"split_{split_id}.parquet"
                                    )
                                    joblist.append(
                                        delayed(get_lr_output)(
                                            df=df_w_splits,
                                            num_layers=num_layers,
                                            selected_alphas=selected_alphas,
                                            selected_l1s=selected_l1s,
                                            split_id=split_id,
                                            impact_type=impact_type,
                                            impactme_outcome=impactme_outcome,
                                            export_path=export_path,
                                            max_iter=max_iter,
                                            seed=seed,
                                            agg_type=agg_type,
                                        )
                                    )

    ## Run in LR call for each split/c-val combo in parallel
    _ = Parallel(n_jobs=n_jobs, verbose=verbose)(joblist)
    return None


def get_best_params_score(import_dir, num_layers, show_progress=False):
    best_params_score_dict_mse = {
        "avg_mse": 9999999,
        "stdev_mse": 9999999,
        "avg_r2": 0,
        "stdev_r2": 9999999,
        "avg_diff_2_1": 0,
        "stdev_diff_2_1": 0,
        "avg_diff_3_2": 0,
        "stdev_diff_3_2": 0,
        "l1_2": "None",
        "alpha_2": "None",
        "l1_3": "None",
        "alpha_3": "None",
    }
    best_params_score_dict_r2 = {
        "avg_mse": 9999999,
        "stdev_mse": 9999999,
        "avg_r2": 0,
        "stdev_r2": 9999999,
        "avg_diff_2_1": 0,
        "stdev_diff_2_1": 0,
        "avg_diff_3_2": 0,
        "stdev_diff_3_2": 0,
        "l1_2": "None",
        "alpha_2": "None",
        "l1_3": "None",
        "alpha_3": "None",
    }
    for alpha_dir in import_dir.iterdir():
        alpha_list = alpha_dir.name.split("_")
        for l1_dir in alpha_dir.iterdir():
            l1_list = l1_dir.name.split("_")
            mse_list = []
            final_r2_list = []  # r^2 of layer3 final preds (sum of all layers)
            diff_2_1_list = []  # layer2 r^2 - layer1 r^2
            diff_3_2_list = []  # layer3 r^2 - layer2 r^2
            if num_layers == 3:
                alpha_2 = alpha_list[1]
                alpha_3 = alpha_list[2]
                l1_2 = l1_list[1]
                l1_3 = l1_list[2]
            else:  # 2 layers
                alpha_2 = None
                alpha_3 = alpha_list[1]
                l1_2 = None
                l1_3 = l1_list[1]
            for repeat_dir in l1_dir.iterdir():
                for split_file in repeat_dir.iterdir():
                    df_import = pd.read_parquet(split_file)
                    y_true = df_import["true"]
                    # Preds at each layer
                    y_pred1 = df_import["pred_stage1"]
                    y_pred2 = df_import["pred_stage2_total"]
                    y_pred3 = df_import["pred_stage3_total"]
                    # SE avg
                    mse = mean_squared_error(y_true=y_true, y_pred=y_pred3)
                    mse_list.append(mse)
                    ## r^2
                    r2_3 = r2_score(y_true=y_true, y_pred=y_pred3)
                    r2_2 = r2_score(y_true=y_true, y_pred=y_pred2)
                    r2_1 = r2_score(y_true=y_true, y_pred=y_pred1)
                    final_r2_list.append(r2_3)
                    ## Layer 2-1 avg
                    diff_2_1 = r2_2 - r2_1
                    diff_2_1_list.append(diff_2_1)
                    ## Layer 3-2 avg
                    diff_3_2 = r2_3 - r2_2
                    diff_3_2_list.append(diff_3_2)
            # get avg mse across folds/repeats for l1-alpha combo
            avg_mse = np.mean(mse_list)
            avg_r2 = np.mean(final_r2_list)
            mse_condition = avg_mse < best_params_score_dict_mse["avg_mse"]
            r2_condition = avg_r2 > best_params_score_dict_r2["avg_r2"]
            if mse_condition or r2_condition:
                ## MSE/r^2 stdev
                stdev_mse = np.std(mse_list)
                stdev_r2 = np.std(final_r2_list)
                # Layer2-Layer1 AVG + STDEV
                avg_diff_2_1 = np.mean(diff_2_1_list)
                stdev_diff_2_1 = np.std(diff_2_1_list)
                # Layer3-Layer2 AVG + STDEV
                avg_diff_3_2 = np.mean(diff_3_2_list)
                stdev_diff_3_2 = np.std(diff_3_2_list)
                if mse_condition:
                    best_params_score_dict_mse = {
                        "avg_mse": round(avg_mse, 3),
                        "stdev_mse": round(stdev_mse, 3),
                        "avg_r2": round(avg_r2, 4),
                        "stdev_r2": round(stdev_r2, 4),
                        "avg_diff_2_1": round(avg_diff_2_1, 4),
                        "stdev_diff_2_1": round(stdev_diff_2_1, 4),
                        "avg_diff_3_2": round(avg_diff_3_2, 4),
                        "stdev_diff_3_2": round(stdev_diff_3_2, 4),
                        "l1_2": l1_2,
                        "alpha_2": alpha_2,
                        "l1_3": l1_3,
                        "alpha_3": alpha_3,
                    }
                if r2_condition:
                    best_params_score_dict_r2 = {
                        "avg_mse": round(avg_mse, 3),
                        "stdev_mse": round(stdev_mse, 3),
                        "avg_r2": round(avg_r2, 4),
                        "stdev_r2": round(stdev_r2, 4),
                        "avg_diff_2_1": round(avg_diff_2_1, 4),
                        "stdev_diff_2_1": round(stdev_diff_2_1, 4),
                        "avg_diff_3_2": round(avg_diff_3_2, 4),
                        "stdev_diff_3_2": round(stdev_diff_3_2, 4),
                        "l1_2": l1_2,
                        "alpha_2": alpha_2,
                        "l1_3": l1_3,
                        "alpha_3": alpha_3,
                    }
    return {"mse": best_params_score_dict_mse, "r2": best_params_score_dict_r2}


def get_r2_by_outcome(pred_dir, outcomes, nlayers):
    result_list = []
    for outcome_name in outcomes:
        if nlayers == 2:
            glob_pattern = "repeat_*"
        else:  # 3 layers
            glob_pattern = "*/*/repeat_*"
        mse_list = []
        r2_3_list = []
        r2_1_list = []
        pred_dir_outcome = pred_dir / outcome_name
        for repeat_dir in pred_dir_outcome.glob(glob_pattern):
            for split_file in repeat_dir.iterdir():
                df_import = pd.read_parquet(split_file)
                y_true = df_import["true"]
                # Preds at each layer
                y_pred1 = df_import["pred_stage1"]
                y_pred2 = df_import["pred_stage2_total"]
                y_pred3 = df_import["pred_stage3_total"]
                # SE avg
                mse = mean_squared_error(y_true=y_true, y_pred=y_pred3)
                ## r^2
                r2_3 = r2_score(y_true=y_true, y_pred=y_pred3)
                r2_2 = r2_score(y_true=y_true, y_pred=y_pred2)
                r2_1 = r2_score(y_true=y_true, y_pred=y_pred1)
                r2_1_list.append(r2_1)
                r2_3_list.append(r2_3)
                mse_list.append(mse)

        avg_mse = np.mean(mse_list)

        avg_r2_L1 = np.mean(r2_1_list)
        stdev_r2_L1 = np.std(r2_1_list)

        avg_r2_outcome = np.mean(r2_3_list)
        stdev_r2_outcome = np.std(r2_3_list)
        result_list.append(
            {
                "outcome": outcome_name,
                "avg_r2_L1": f"{avg_r2_L1:.3f} ({stdev_r2_L1:.3f})",
                "avg_r2_outcome": f"{avg_r2_outcome:.3f} ({stdev_r2_outcome:.3f})",
            }
        )
    return (
        pd.DataFrame(result_list)
        .sort_values(by="avg_r2_outcome", ascending=False)
        .reset_index(drop=True)
    )


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
    parser.add_argument("--df_path", required=True)
    parser.add_argument("--base_save_dir", required=True)
    parser.add_argument("--alpha_list", required=False, nargs="+", default=ALPHA_LIST)
    parser.add_argument("--l1_list", required=False, nargs="+", default=L1_LIST)
    parser.add_argument(
        "--impactme_outcome_list", required=False, default=None, nargs="+"
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
        "--num_layers", required=False, type=int, default=3, choices=[2, 3]
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
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)
    alpha_list = (
        None if args.alpha_list == ["None"] else [float(a) for a in args.alpha_list]
    )
    l1_list = None if args.l1_list == ["None"] else [float(l) for l in args.l1_list]
    impactme_outcome_list = (
        None if args.impactme_outcome_list == ["None"] else args.impactme_outcome_list
    )
    ## Brute force preds
    save_dir = Path(args.base_save_dir)

    if impactme_outcome_list is None:
        # This runs to get raw preds on each param combo
        loop_through_l1(
            impact_type=args.impact_type,
            agg_type=args.agg_type,
            df_path=Path(args.df_path),
            base_save_dir=save_dir,
            num_layers=args.num_layers,
            alpha_list=alpha_list,
            l1_list=l1_list,
            impactme_outcome=None,
            max_iter=args.max_iter,
            seed=args.seed,
            n_repeats=args.n_repeats,
            split_tol=args.split_tol,
            n_splits=args.n_splits,
            n_jobs=args.n_jobs,
        )
    else:
        print(
            f"\t Running per outcome for agg: {args.agg_type}, impact: {args.impact_type}..."
        )
        # This runs to get raw preds w/ each impactme outcome as sole predictor
        for outcome_name in impactme_outcome_list:
            loop_through_l1(
                impact_type=args.impact_type,
                agg_type=args.agg_type,
                df_path=Path(args.df_path),
                base_save_dir=save_dir,
                num_layers=args.num_layers,
                alpha_list=alpha_list,
                l1_list=l1_list,
                impactme_outcome=outcome_name,
                max_iter=args.max_iter,
                seed=args.seed,
                n_repeats=args.n_repeats,
                split_tol=args.split_tol,
                n_splits=args.n_splits,
                n_jobs=args.n_jobs,
            )
    if args.tune == "True":
        relative_save_dir = save_dir.relative_to(BASE_PATH)
        if impactme_outcome_list is None:
            # This runs to get best alpha/l1 for each impact/agg/model/desc_seg combo
            agg_type = relative_save_dir.parts[1]
            impact_type = relative_save_dir.parts[2]
            model_name = relative_save_dir.parts[3]
            desc_seg_type = relative_save_dir.parts[
                4
            ]  # desc for 0-shot, seg for others
            print("\t RUNNING REGULAR TUNER...")
            best_params_dict = get_best_params_score(
                import_dir=save_dir, num_layers=args.num_layers
            )
            param_save_path = (
                BASE_PATH
                / f"initial_results_{args.num_layers}"
                / "best_params"
                / agg_type
                / impact_type
                / model_name
                / f"{desc_seg_type}.json"
            )
            param_save_path.parent.mkdir(exist_ok=True, parents=True)
            with open(param_save_path, "w") as f:
                json.dump(best_params_dict, f, indent=4)
        else:
            # This runs to aggregate r2 in all outcomes for each impact/agg/n_layers combo
            print("\t Getting r2 by outcome...")
            r2_by_outcome_df = get_r2_by_outcome(
                pred_dir=save_dir,
                outcomes=impactme_outcome_list,
                nlayers=args.num_layers,
            )
            agg_type = relative_save_dir.parts[2]
            impact_type = relative_save_dir.parts[3]
            save_path = (
                BASE_PATH
                / "outcome_analysis"
                / "outcome_tables"
                / agg_type
                / impact_type
                / f"{args.num_layers}_layers.tsv"
            )
            if save_path.exists():
                save_path.unlink()
            save_path.parent.mkdir(exist_ok=True, parents=True)
            r2_by_outcome_df.to_csv(save_path, sep="\t")
        print("\t DONE!")


if __name__ == "__main__":
    main()
