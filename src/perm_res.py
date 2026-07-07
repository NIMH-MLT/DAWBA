# perm_res.py --> permutation results + other sig test helpers
from src.config import BASE_PATH
from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt


###### HELPERS ######
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_layer1(r2_df):
    r2s_1 = r2_df[r2_df["num_layers"] == 1]
    if len(r2s_1) == 0:
        return None
    # r2s_1
    best_1Q_row = r2s_1.sort_values(by="r2_Q_mean", ascending=False).iloc[0]
    best_1Q_alpha = best_1Q_row["alpha"].split("_")[0]
    best_1Q_val = best_1Q_row["r2_Q_mean"]
    r2s_1["best_Q_alpha"] = f"{best_1Q_val}_{best_1Q_alpha}"
    r2s_1["diff"] = round(r2s_1["r2_T_mean"] - best_1Q_val, 3)
    r2s_1 = r2s_1.sort_values(by="r2_T_mean", ascending=False)
    return r2s_1


def get_layer2(r2_df):
    r2s_2 = r2_df[r2_df["num_layers"] == 2]

    r2s_2_S_base = r2s_2[r2s_2["two_layer_base"] == "S"]
    r2s_2_S_base["diff"] = round(
        r2s_2_S_base["r2_T_mean"] - r2s_2_S_base["r2_S_mean"], 3
    )
    r2s_2_S_base = r2s_2_S_base.sort_values(by="diff", ascending=False)

    r2s_2_Q_base = r2s_2[r2s_2["two_layer_base"] == "Q"]
    r2s_2_Q_base["best_Q"] = max(r2s_2_Q_base["r2_Q_mean"])
    r2s_2_Q_base["diff"] = round(r2s_2_Q_base["r2_T_mean"] - r2s_2_Q_base["best_Q"], 3)
    r2s_2_Q_base = r2s_2_Q_base.sort_values(by="diff", ascending=False)

    return r2s_2_S_base, r2s_2_Q_base


def get_layer3(r2_df):
    r2s_3 = r2_df[r2_df["num_layers"] == 3]

    best_3Q_row = r2s_3.sort_values(by="r2_Q_mean", ascending=False).iloc[0]
    best_3Q_alpha = best_3Q_row["alpha"].split("_")[0]
    best_3Q_val = best_3Q_row["r2_Q_mean"]
    r2s_3["best_Q_alpha"] = f"{best_3Q_val}_{best_3Q_alpha}"
    r2s_3["diff"] = round(r2s_3["r2_T_mean"] - best_3Q_val, 3)
    r2s_3 = r2s_3.sort_values(by="diff", ascending=False)
    return r2s_3


def compute_r2s(g):
    scores = {
        "r2_S": r2_score(g["true"], g["pred_stage1"]),
        "r2_Q": r2_score(g["true"], g["pred_stage2_total"]),
        "r2_T": r2_score(g["true"], g["pred_stage3_total"]),
    }
    return pd.Series(scores)


def get_null_diff_list(num_layers, df, layer_2_base=None):
    """
    - 2 layer: diff = r2_T - r2_S
    - 3 layer: diff = r2_T - r2_Q
    - NULL distribvution = diff of max(ST) - max(S,SQ) of each permutation
        - only within hypotheses
    """
    sub_df = df[df["num_layers"] == num_layers]
    if num_layers == 2:
        if layer_2_base is None:
            return ValueError("Layer2 base must not be None when num_layers=2")
        sub_df = sub_df[sub_df["two_layer_base"] == "S"]
        reduced_layer_str = "r2_S_mean"
    else:
        reduced_layer_str = "r2_Q_mean"
    # List of max diff from each perm
    diff_list = (
        sub_df.groupby("perm")
        .apply(lambda g: max(g["r2_T_mean"]) - max(g[reduced_layer_str]))
        .round(10)
    )
    return diff_list


def get_raw_p_df(num_layers, df, layer_2_base=None):
    diff_list = get_null_diff_list(num_layers, df, layer_2_base)
    baseline_df = df[df["perm"] == "baseline"].reset_index(drop=True)
    if num_layers == 2:
        r2s_2_S_base, r2s_2_Q_base = get_layer2(baseline_df)
        basline_df_w_diff = r2s_2_S_base if layer_2_base == "S" else r2s_2_Q_base
    else:
        basline_df_w_diff = get_layer3(baseline_df)
    basline_df_w_diff["p_val"] = basline_df_w_diff["diff"].apply(
        lambda x: (diff_list >= x).mean()
    )
    basline_df_w_diff["p_val"] = np.where(
        basline_df_w_diff["r2_T_mean"] < 0, np.nan, basline_df_w_diff["p_val"]
    )
    basline_df_w_diff.sort_values(
        ["p_val", "diff"], ascending=[True, False], inplace=True
    )
    return basline_df_w_diff, diff_list


def get_collapsed_null(
    impact_type,
    keep_cols,
    fill_na_cols,
    perm_dir_base=BASE_PATH / "perm_res" / "r2_tables",
):
    """
    Creates null distribution by first collapsing hypothesis axis (min) then collapsing model axis (max)
    for each permutation. Results in a single diff value per permutation to create a shared null distribution
    for a given impact_type.

    Params
    -----
    impact_type: str
        One of ['self', 'parent']

    Returns
    -------
    null_dist_hyp: list[float]
        List of max( <-- across all models
                    min( <-- for each model
                        (r2_SQ - r2_S),
                        (r2_SQT-max(r2_SQ))
                    )
                )
    baseline_df: pandas DF

    """
    null_dist_hyp = []
    baseline_df = None
    for perm_dir in perm_dir_base.iterdir():
        ## 3-layer r2 prep
        r2_df = pd.read_parquet(perm_dir / f"{impact_type}.parquet")
        r2_df[fill_na_cols] = r2_df[fill_na_cols].fillna("None")
        # 3 layer diff
        r2_df_3 = r2_df[r2_df["num_layers"] == 3]
        best_Q = float(r2_df_3["r2_Q_mean"].max())
        r2_df_3["diff_SQT_SQ"] = r2_df_3["r2_T_mean"] - best_Q
        r2_df_3.drop("two_layer_base", inplace=True, axis=1)

        # 2 layer diff
        r2_df_2 = r2_df[(r2_df["num_layers"] == 2) & (r2_df["two_layer_base"] == "S")]
        r2_df_2["diff_ST_S"] = r2_df_2["r2_T_mean"] - r2_df_2["r2_S_mean"]
        r2_df_2["alpha"] = r2_df_2["alpha"] + "_" + r2_df_2["alpha"]
        r2_df_2.drop("two_layer_base", inplace=True, axis=1)

        ## Collapse hypothesis axis
        combined_df = collapse_hypothesis(
            df1=r2_df_2,
            df2=r2_df_3,
            col1="diff_ST_S",
            col2="diff_SQT_SQ",
            keep_cols=keep_cols,
        )
        # Save baseline df for models
        if perm_dir.name == "baseline":
            baseline_df = combined_df
            # Mark min_diff NA when either r^2 T-layer <0
            baseline_df["min_diff"] = np.where(
                (baseline_df["r2_T_mean_2"] < 0) | (baseline_df["r2_T_mean_3"] < 0),
                np.nan,
                baseline_df["min_diff"],
            )
        null_dist_hyp.append(combined_df["min_diff"].max())
    return null_dist_hyp, baseline_df


def collapse_hypothesis(df1, df2, col1, col2, keep_cols):
    common_cols = [
        "agg_type",
        "llm_model",
        "desc_seg",
        "alpha",
    ]
    combined = pd.merge(
        df1,
        df2,
        on=common_cols,
        suffixes=("_2", "_3"),
    )
    combined["min_diff"] = combined.apply(
        lambda row: min(getattr(row, col1), getattr(row, col2)), axis=1
    )
    combined_sub = combined[keep_cols].sort_values("min_diff", ascending=False)
    return combined_sub


def get_corr_p_vals(
    baseline_df,
    null_dist,
    title,
    diff_col="min_diff",
    show_plot=True,
    xlim=(-0.05, 0.1),
    ylim=(0, 50),
):
    baseline_df["p_val"] = baseline_df[diff_col].apply(
        lambda baseline: (np.array(null_dist) >= baseline).mean()
        # get_ranking(baseline, null_dist)
    )
    baseline_df["p_val"] = np.where(
        baseline_df[diff_col].isna(), np.nan, baseline_df["p_val"]
    )
    combined_sig = baseline_df[(baseline_df["p_val"] < 0.05)].sort_values(
        diff_col, ascending=False
    )
    ## Includes NA min diffs (at least one of r2_T_2 and/or r2_T_3 <0)
    non_sig = baseline_df[~(baseline_df["p_val"] < 0.05)].sort_values(
        diff_col, ascending=False
    )
    assert len(combined_sig) + len(non_sig) == len(baseline_df)
    print(f"Significant model configs: {len(combined_sig)} / {len(baseline_df)}")
    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.xlim(xlim)
        plt.ylim(ylim)
        if len(combined_sig) > 0:
            plt.hist(
                combined_sig[diff_col],
                bins=25,
                alpha=0.6,
                color="green",
                label=f"SIG Observed diffs ({len(combined_sig)} / {len(baseline_df)})",
            )
        plt.hist(
            non_sig[diff_col],
            bins=50,
            alpha=0.6,
            color="red",
            label=f"NON-SIG Observed diffs ({len(non_sig)} / {len(baseline_df)})",
        )
        plt.hist(
            null_dist,
            bins=50,
            alpha=0.6,
            color="grey",
            label="Null distribution (1000 perms)",
        )
        plt.title(title)
        plt.xlabel("Difference in R²")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    return combined_sig, non_sig


######## MAIN FUNCS #######
def get_avg_r2(df):
    df = df.fillna("None")
    params = [
        "num_layers",
        "agg_type",
        "llm_model",
        "desc_seg",
        "alpha",
        "two_layer_base",
    ]
    r2_df = (
        df.groupby(params + ["repeat"])
        .apply(compute_r2s)  # compute r2 within repeat
        .reset_index()
        .groupby(params)[["r2_S", "r2_Q", "r2_T"]]
        .agg(["mean", "std"])
        .round(3)
        .reset_index()
    )
    r2_df = r2_df.fillna("None")
    r2_df.columns = ["_".join(c).strip("_") for c in r2_df.columns]
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
                [
                    "agg_type",
                    "llm_model",
                    "desc_seg",
                    "alpha",
                    "two_layer_base",
                ]
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
    parser.add_argument("--num_layer_list", required=True, type=int, nargs="+")
    parser.add_argument("--import_dir", required=True, type=Path)
    parser.add_argument("--search_dict", type=json.loads)
    parser.add_argument("--export_path", type=Path)
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)

    ## Create r2 df
    id_df = pd.read_parquet(args.import_dir / "id_df.parquet")
    preds_df = pd.read_parquet(args.import_dir / "pred_df.parquet")
    ## Ensure consistent
    log("Checking consistency...")
    for num_layers in args.num_layer_list:
        ensure_consistent(
            id_df=id_df,
            num_layers=num_layers,
        )
        log(f"\t{num_layers} layer(s) consistent!")
    log("Getting r^2...")
    r2_df = get_avg_r2(preds_df)

    ## Export r2 df
    log("Exporting...")
    if args.export_path.exists():
        args.export_path.unlink()
    args.export_path.parent.mkdir(exist_ok=True, parents=True)
    r2_df.to_parquet(args.export_path)
    assert args.export_path.exists()
    log("DONE!")


if __name__ == "__main__":
    main()
