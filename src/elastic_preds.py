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
    max_iter=50000,
    seed=SEED,
    n_repeats=5,
    split_tol=0.1,
    n_jobs=1,
    verbose=0,
):
    joblist = []
    df = pd.read_parquet(df_path)
    for alpha_2 in ALPHA_LIST:  # ALPHA_LIST
        for alpha_3 in ALPHA_LIST:
            selected_alphas = [alpha_2, alpha_3]
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
                for l1_ratio_2 in L1_LIST:
                    for l1_ratio_3 in L1_LIST:
                        for split_id in range(n_splits):
                            export_path = (
                                base_save_dir
                                / f"alpha_{alpha_2}_{alpha_3}"
                                / f"l1_{l1_ratio_2}_{l1_ratio_3}"
                                / f"repeat_{repeat}"
                                / f"split_{split_id}.parquet"
                            )
                            selected_l1s = [l1_ratio_2, l1_ratio_3]
                            joblist.append(
                                delayed(get_lr_output)(
                                    df=df_w_splits,
                                    selected_alphas=selected_alphas,
                                    selected_l1s=selected_l1s,
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


def get_best_params_score(import_dir, show_progress=False):
    best_params_score_dict_mse = {
        "avg_mse": 9999999,
        "stdev_mse": 9999999,
        "avg_r2": 0,
        "stdev_r2": 9999999,
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
        "l1_2": "None",
        "alpha_2": "None",
        "l1_3": "None",
        "alpha_3": "None",
    }
    total_files = sum(
        1
        for alpha_dir in import_dir.iterdir()
        for l1_dir in alpha_dir.iterdir()
        for repeat_dir in l1_dir.iterdir()
        for _ in repeat_dir.iterdir()
    )
    with tqdm(
        total=total_files,
        desc="Scoring params",
        leave=False,  # disappears when done
        dynamic_ncols=True,
        disable=not show_progress,
    ) as pbar:
        for alpha_dir in import_dir.iterdir():
            alpha_list = alpha_dir.name.split("_")
            alpha_2 = alpha_list[1]
            alpha_3 = alpha_list[2]
            for l1_dir in alpha_dir.iterdir():
                l1_list = l1_dir.name.split("_")
                l1_2 = l1_list[1]
                l1_3 = l1_list[2]
                mse_list = []
                r2_list = []
                for repeat_dir in l1_dir.iterdir():
                    for split_file in repeat_dir.iterdir():
                        df_import = pd.read_parquet(split_file)
                        y_true = df_import["true"]
                        y_pred = df_import["pred_stage3_total"]
                        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
                        r2 = r2_score(y_true=y_true, y_pred=y_pred)
                        mse_list.append(mse)
                        r2_list.append(r2)
                        pbar.update(1)
                # get avg mse across folds/repeats for l1-alpha combo
                avg_mse = np.mean(mse_list)
                avg_r2 = np.mean(r2_list)
                mse_condition = avg_mse < best_params_score_dict_mse["avg_mse"]
                r2_condition = avg_r2 > best_params_score_dict_r2["avg_r2"]
                if mse_condition or r2_condition:
                    stdev_mse = np.std(mse_list)
                    stdev_r2 = np.std(r2_list)
                    if mse_condition:
                        best_params_score_dict_mse = {
                            "avg_mse": round(avg_mse, 3),
                            "stdev_mse": round(stdev_mse, 3),
                            "avg_r2": round(avg_r2, 4),
                            "stdev_r2": round(stdev_r2, 4),
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
                            "l1_2": l1_2,
                            "alpha_2": alpha_2,
                            "l1_3": l1_3,
                            "alpha_3": alpha_3,
                        }
    return {"mse": best_params_score_dict_mse, "r2": best_params_score_dict_r2}


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
    ## Brute force preds
    save_dir = Path(args.base_save_dir)
    loop_through_l1(
        impact_type=args.impact_type,
        agg_type=args.agg_type,
        df_path=Path(args.df_path),
        base_save_dir=save_dir,
        max_iter=args.max_iter,
        seed=args.seed,
        n_repeats=args.n_repeats,
        split_tol=args.split_tol,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
    )

    if args.tune == "True":
        print("RUNNING TUNER...")
        best_params_dict = get_best_params_score(save_dir)

        relative_save_dir = save_dir.relative_to(BASE_PATH)
        agg_type = relative_save_dir.parts[1]
        impact_type = relative_save_dir.parts[2]
        model_name = relative_save_dir.parts[3]
        param_save_path = (
            BASE_PATH
            / "initial_results"
            / "best_params"
            / agg_type
            / impact_type
            / model_name
        )
        param_save_path.mkdir(exist_ok=True, parents=True)
        ## 0-shot run
        if "0_shot" in agg_type:
            desc_type = relative_save_dir.parts[4]
            with open(param_save_path / f"{desc_type}.json", "w") as f:
                json.dump(best_params_dict, f, indent=4)
        # ImpactMe LR models
        else:
            seg_type = relative_save_dir.parts[4]
            with open(param_save_path / f"{seg_type}.json", "w") as f:
                json.dump(best_params_dict, f, indent=4)


if __name__ == "__main__":
    main()
