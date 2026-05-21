# perm_res.py

import argparse
import json
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from src.config import BASE_PATH
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_ranking(baseline, null_dist):
    extreme = len([1 for null_diff in null_dist if null_diff >= baseline])
    p_val = extreme / len(null_dist)
    return p_val


def get_collapsed_null(
    impact_type, best_alpha, perm_dir_base=BASE_PATH / "perm_res" / "r2_tables"
):
    """
    Creates null distribution by first collapsing hypothesis axis (min) then collapsing model axis (max)
    for each permutation. Results in a single diff value per permutation to create a shared null distribution
    for a given impact_type.

    Params
    -----
    impact_type: str
        One of ['self', 'parent']
    """
    null_dist = []
    baseline_df = None
    for perm_dir in perm_dir_base.iterdir():
        ## 3-layer r2 prep
        r2_df_3 = pd.read_parquet(perm_dir / f"{impact_type}_3.parquet")
        # Ensure only using best 3Q alpha
        r2_df_3 = r2_df_3[r2_df_3["alpha"].str.split("_").str[0] == best_alpha]
        r2_df_3["baseline_diff"] = r2_df_3["r2_T"] - r2_df_3["r2_Q"]

        ## 2-layer r2 prep
        r2_df_2 = pd.read_parquet(perm_dir / f"{impact_type}_2.parquet")
        r2_df_2_dup = r2_df_2.copy()
        r2_df_2_dup["alpha"] = f"{best_alpha}_" + r2_df_2["alpha"]
        r2_df_2_dup["baseline_diff"] = r2_df_2_dup["r2_T"] - r2_df_2_dup["r2_S"]
        # Make indicator col for neg r2
        # if perm_dir.name == "baseline":
        #    r2_df_3["neg_r2"] = np.where(r2_df_3["r2_T"] < 0, 1, 0)
        #    r2_df_2_dup["neg_r2"] = np.where(r2_df_2_dup["r2_T"] < 0, 1, 0)
        # Drop r2 cols
        # r2_df_3.drop(["r2_S", "r2_Q", "r2_T"], axis=1, inplace=True)
        # r2_df_2_dup.drop(["r2_S", "r2_Q", "r2_T"], axis=1, inplace=True)
        ## Collapse hypothesis axis
        combined_df = collapse_hypothesis(r2_df_2_dup, r2_df_3, remove_p_col=False)
        # Save baseline df for models
        if perm_dir.name == "baseline":
            baseline_df = combined_df
            # combined_df["neg_r2"] = (
            #     (combined_df["neg_r2_2"] + combined_df["neg_r2_3"]) != 0
            # ).astype(int)
            # combined_df.drop(["neg_r2_2", "neg_r2_3"], axis=1, inplace=True)
        ## Collapse model-config axis
        max_diff = combined_df["min_diff"].max()
        null_dist.append(max_diff)
    return null_dist, baseline_df


def expand_alpha_rows(df, best_alpha):
    """
    Expands df by duplicating singular alpha vals from 2-layer dfs.
    Resulting df will have len(alphas) * len(alphas) rows (increase from len(alphas) rows before)
    of the format <original_alpha>_<alpha_n> for alpha_n in alphas.

    Used to create consistency across 2 and 3 layer dfs so that they may be merged.
    """
    alphas = list(df["alpha"].unique())
    new_rows = []
    for _, row in df.iterrows():
        for alpha in alphas:
            base_row = row.to_dict()
            #
            base_row["alpha"] = f"{row.alpha}_{alpha}"
            new_rows.append(base_row)
    return pd.DataFrame(new_rows)


def collapse_hypothesis(df1, df2, remove_p_col=True, remove_baseline_diffs=True):
    """
    Collapses hypothesis across 2 and 3 layers by taking the min of the baseline diffs.
    Returns a df with columns: agg, llama_model, desc_seg, alpha, min_diff
    """
    combined = pd.merge(
        df1, df2, on=["agg", "llama_model", "desc_seg", "alpha"], suffixes=("_2", "_3")
    )
    if remove_p_col:
        combined = combined.drop(columns=["p_val_2", "p_val_3"])
    combined["min_diff"] = combined.apply(
        lambda row: min(row.baseline_diff_2, row.baseline_diff_3), axis=1
    )
    if remove_baseline_diffs:
        combined = combined.drop(columns=["baseline_diff_2", "baseline_diff_3"])
    return combined.sort_values("min_diff", ascending=False)


def get_p_df(num_layers, r2_df_dict, best_alpha=None):
    print(f"{num_layers}...")

    # Determine diff column based on num_layers
    if num_layers == 2:
        diff_col = "r2_S"
    elif num_layers == 3:
        diff_col = "r2_Q"
    else:
        raise ValueError(f"Num layers must be one of: [2,3]. Got {num_layers}!")

    # Concatenate all perm dfs, tagging each with its perm_iter key
    all_perms_df = pd.concat(
        {k: v for k, v in r2_df_dict.items()}, names=["perm_iter"]
    ).reset_index(level="perm_iter")
    if num_layers == 3:
        ## Filter w/ best alpha
        all_perms_df = all_perms_df[
            all_perms_df["alpha"].str.split("_").str[0] == best_alpha
        ]
        assert np.allclose(
            all_perms_df[diff_col].values, all_perms_df[diff_col].values[0]
        ), all_perms_df[diff_col].unique()
    all_perms_df["diff"] = all_perms_df["r2_T"] - all_perms_df[diff_col]

    diff_list = all_perms_df.groupby("perm_iter")["diff"].max()
    baseline_df = all_perms_df[all_perms_df["perm_iter"] == "baseline"].drop(
        "perm_iter", axis=1
    )
    baseline_df["p_val"] = baseline_df["diff"].apply(lambda x: (diff_list >= x).mean())
    baseline_df["p_val"] = np.where(
        baseline_df["r2_T"] < 0, np.nan, baseline_df["p_val"]
    )
    baseline_df.sort_values(["p_val", "diff"], ascending=[True, False], inplace=True)

    return baseline_df, diff_list.to_list()


################################################################
################################################################
################################################################
def get_sub_df(df, condition_dict):
    """
    Given some condition, subset the df accordingly, usually resulting in a single row
    """
    mask = pd.Series(True, index=df.index)
    for col, val in condition_dict.items():
        mask &= df[col] == val
    return df[mask]


def compute_r2s(g):
    scores = {
        "r2_S": r2_score(g["true"], g["pred_stage1"]),
        "r2_Q": r2_score(g["true"], g["pred_stage2_total"]),
        "r2_T": r2_score(g["true"], g["pred_stage3_total"]),
    }
    return pd.Series(scores)


def get_avg_r2(df):
    r2_df = (
        df.groupby(
            ["agg_type", "llama_model", "desc_seg", "alpha", "num_layers", "repeat"]
        )
        .apply(compute_r2s)
        .reset_index()
        .groupby(["agg_type", "llama_model", "desc_seg", "alpha", "num_layers"])[
            ["r2_S", "r2_Q", "r2_T"]
        ]
        .mean()
        .reset_index()
    )
    return r2_df


def ensure_consistent(id_df, num_layers, num_repeats=10, num_splits=3):
    """
    Ensure consistency within a given permutation/repeat/split for some combination
    of impact/num_layers.
    """
    id_df = id_df[id_df["num_layers"] == num_layers]
    col_list = [f"perm_id_split_{i}" for i in range(num_splits)]
    for repeat in range(num_repeats):
        repeat_df = id_df[id_df["repeat"] == repeat]
        groups = [
            group[col_list].reset_index(drop=True)
            for _, group in repeat_df.groupby(
                ["agg_type", "llama_model", "desc_seg", "alpha"]
            )
        ]
        all_identical = all(groups[0].equals(g) for g in groups)
        if not all_identical:
            print(groups)
        assert all_identical


def build_parser():
    parser = argparse.ArgumentParser(
        prog="perm_results",
    )
    parser.add_argument("--num_layers", required=True, type=int)
    parser.add_argument("--import_dir", required=True)
    parser.add_argument("--search_dict", type=json.loads)
    parser.add_argument("--export_path")
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)
    import_dir = Path(args.import_dir)
    ## Create r2 df
    id_df = pd.read_parquet(import_dir / "id_df_combined_new.parquet")
    preds_df = pd.read_parquet(import_dir / "pred_df_combined_new.parquet")
    ## Ensure consistent
    log("Checking consistency...")
    ensure_consistent(
        id_df=id_df,
        num_layers=args.num_layers,
    )
    log("\tConsistent!")
    log("Getting r^2...")
    preds_df_sub = preds_df[preds_df["num_layers"] == args.num_layers]
    r2_df = get_avg_r2(preds_df_sub)

    ## Export r2 df
    log("Exporting...")
    export_path = Path(args.export_path)
    if export_path.exists():
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    r2_df.to_parquet(export_path)
    assert export_path.exists()
    log("DONE!")


if __name__ == "__main__":
    main()
