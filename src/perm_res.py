# perm_res.py

import argparse
import json
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from src.config import BASE_PATH


def get_ranking(baseline, null_dist):
    extreme = len([1 for null_diff in null_dist if null_diff >= baseline])
    p_val = extreme / len(null_dist)
    return p_val


def get_collapsed_null(impact_type, perm_dir_base=BASE_PATH / "perm_res" / "r2_tables"):
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
        r2_df_3["baseline_diff"] = r2_df_3["r2_T"] - r2_df_3["r2_Q"]

        ## 2-layer r2 prep
        r2_df_2 = pd.read_parquet(perm_dir / f"{impact_type}_2.parquet")
        r2_df_2_dup = expand_alpha_rows(r2_df_2)
        r2_df_2_dup["baseline_diff"] = r2_df_2_dup["r2_T"] - r2_df_2_dup["r2_S"]
        # Make indicator col for neg r2
        if perm_dir.name == "baseline":
            r2_df_3["neg_r2"] = np.where(r2_df_3["r2_T"] < 0, 1, 0)
            r2_df_2_dup["neg_r2"] = np.where(r2_df_2_dup["r2_T"] < 0, 1, 0)
        # Drop r2 cols
        r2_df_3.drop(["r2_S", "r2_Q", "r2_T"], axis=1, inplace=True)
        r2_df_2_dup.drop(["r2_S", "r2_Q", "r2_T"], axis=1, inplace=True)
        ## Collapse hypothesis axis
        combined_df = collapse_hypothesis(r2_df_2_dup, r2_df_3, remove_p_col=False)
        # Save baseline df for models
        if perm_dir.name == "baseline":
            baseline_df = combined_df
            combined_df["neg_r2"] = (
                (combined_df["neg_r2_2"] + combined_df["neg_r2_3"]) != 0
            ).astype(int)
            combined_df.drop(["neg_r2_2", "neg_r2_3"], axis=1, inplace=True)
        ## Collapse model-config axis
        max_diff = combined_df["min_diff"].max()
        null_dist.append(max_diff)
    return null_dist, baseline_df


def expand_alpha_rows(df):
    """
    Expands df by duplicating singular alpha vals from 2-layer dfs.
    Resulting df will have len(alphas) * len(alphas) rows (increase from len(alphas) rows before)
    of the format <original_alpha>_<alpha_n> for alpha_n in alphas.

    Used to create consistency across 2 and 3 layer dfs so that they may be merged.
    """
    alphas = [0.5, 1.0, 2.5, 5.0, 10.0, 15.0]
    new_rows = []
    for _, row in df.iterrows():
        for alpha in alphas:
            base_row = row.to_dict()
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


def get_p_df(search_dict, num_layers, r2_df_dict):
    """
    Returns a df with columns: agg, llama_model, desc_seg, alpha, baseline_diff, p_val
     - baseline_diff is the diff in r2 between T and S for the baseline (unpermuted) data
     - p_val is the % of permuted diffs that are as or more extreme than the baseline diff (in the pos direction)
     - Each row corresponds to a unique model configuration (agg, llama_model, desc_seg, alpha)
    """
    print(f"{num_layers}...")
    p_val_list = []
    for agg in search_dict["agg"]:
        for llama_model in search_dict["llama_model"]:
            if agg in ["max", "avg", "block"]:
                desc_seg_list = search_dict["seg"]
            else:
                desc_seg_list = search_dict["desc"]
            for desc_seg in desc_seg_list:
                if num_layers == 2:
                    alpha_list = search_dict["alpha"]
                elif num_layers == 3:
                    alpha_list = [
                        f"{alpha2}_{alpha3}"
                        for alpha2, alpha3 in product(
                            search_dict["alpha"], search_dict["alpha"]
                        )
                    ]
                else:
                    raise ValueError(
                        f"Num layers must be one of: [2,3]. Got {num_layers}!"
                    )
                for alpha in alpha_list:
                    ### At this point, all unique vals
                    diff_list = []
                    baseline_diff = -99
                    for perm_iter, perm_df in r2_df_dict.items():
                        condition = (
                            (perm_df["agg"] == agg)
                            & (perm_df["llama_model"] == llama_model)
                            & (perm_df["desc_seg"] == desc_seg)
                            & (perm_df["alpha"] == alpha)
                        )
                        cur_row = perm_df[condition]
                        r2_S = cur_row["r2_S"].values[0]
                        r2_Q = cur_row["r2_Q"].values[0]
                        r2_T = cur_row["r2_T"].values[0]
                        if num_layers == 2:
                            diff = r2_T - r2_S
                        elif num_layers == 3:
                            diff = r2_T - r2_Q
                        else:
                            raise ValueError(
                                f"Num layers must be one of: [2,3]. Got {num_layers}!"
                            )

                        if perm_iter == "baseline":
                            # Save baseline diff seperately
                            baseline_diff = diff
                            pos_list = (perm_df["r2_T"] > 0).astype(int)
                        # Append perm diffs and baseline diff
                        diff_list.append(diff)
                    ## P-VALUE
                    # % of perms as or more extreme than baseline (in pos direction)
                    extreme = len(
                        [
                            1
                            for perm_result_diff in diff_list
                            if perm_result_diff >= baseline_diff
                        ]
                    )
                    p_val = extreme / len(diff_list)
                    p_val_list.append(
                        {
                            "agg": agg,
                            "llama_model": llama_model,
                            "desc_seg": desc_seg,
                            "alpha": alpha,
                            "baseline_diff": baseline_diff,
                            "p_val": p_val,
                            "pos": pos_list,
                        }
                    )
    return pd.DataFrame(p_val_list)


def get_sub_df(df, condition_dict):
    """
    Given some condition, subset the df accordingly, usually resulting in a single row
    """
    mask = pd.Series(True, index=df.index)
    for col, val in condition_dict.items():
        mask &= df[col] == val
    return df[mask]


def get_avg_r2s(df, num_repeats):
    """
    Avg r2 values across repeats for a given previously subset (to some model config) df
    Does this for each of Sum, Questionnaire, and Text layers
    """
    r2_S_list = []
    r2_Q_list = []
    r2_T_list = []
    for repeat in range(num_repeats):
        repeat_df = df[df["repeat"] == repeat]
        y_true = repeat_df["true"]
        y_pred_S = repeat_df["pred_stage1"]
        y_pred_Q = repeat_df["pred_stage2_total"]
        y_pred_T = repeat_df["pred_stage3_total"]
        r2_S_list.append(r2_score(y_true, y_pred_S))
        r2_Q_list.append(r2_score(y_true, y_pred_Q))
        r2_T_list.append(r2_score(y_true, y_pred_T))
    return {
        "r2_S": np.mean(r2_S_list),
        "r2_Q": np.mean(r2_Q_list),
        "r2_T": np.mean(r2_T_list),
    }


def get_r2s_single_iter(id_df, preds_df, search_dict, num_layers, num_repeats=10):
    """
    Aggregate raw predictions from each of S,Q, and T tiers for a given impact-num_layer configuration.

    Params
    ------
    id_df: pd.DataFrame
        For a given num_layers/agg/llama/desc_seg/alpha/repeat combination, contains
        ordered DAWBAIDs for each of split_0, split_1, split_2,
        Columns: perm_id_split_0, perm_id_split_1, perm_id_split_2,
                    repeat, num_layers, agg_type, llama_model,
                    desc_seg, alpha
    preds_df: pd.DataFrame
        For a given num_layers/agg/llama/desc_seg/alpha/repeat combination, contains
        true and predicted scores for each participant (concat across test splits)
        Columns: DAWBA_ID, true, pred_stage1, pred_stage2_increment, pred_stage2_total, pred_stage3_increment,
                    ored_stage3_total, repeat, num_layers, agg_type, llama_model, desc_seg, alpha
    search_dict: dict{}
        Contains possible values for each key
        Keys: agg, llama_model, desc, seg, alpha
    num_layers: int
        One of [2,3]
    """
    ## Ensure consistent permutations across model configs for a given
    # permutation/impact/num_layer combination
    print("Checking consistency...")
    ensure_consistent(
        id_df=id_df,
        search_dict=search_dict,
        num_layers=num_layers,
    )
    print("\tConsistent!")
    res_list = []
    for agg in search_dict["agg"]:
        for llama_model in search_dict["llama_model"]:
            if agg in ["max", "avg", "block"]:
                desc_seg_list = search_dict["seg"]
            else:
                desc_seg_list = search_dict["desc"]
            for desc_seg in desc_seg_list:
                if num_layers == 2:
                    alpha_list = search_dict["alpha"]
                else:
                    alpha_list = [
                        f"{alpha2}_{alpha3}"
                        for alpha2, alpha3 in product(
                            search_dict["alpha"], search_dict["alpha"]
                        )
                    ]
                for alpha in alpha_list:
                    #### AT THIS POINT: all unique values for a given model
                    condition_dict = {
                        "num_layers": num_layers,
                        "agg_type": agg,
                        "llama_model": llama_model,
                        "desc_seg": desc_seg,
                        "alpha": str(alpha),
                    }
                    specific_df = get_sub_df(preds_df, condition_dict)
                    r2_dict = get_avg_r2s(specific_df, num_repeats=num_repeats)
                    res_list.append(
                        {
                            "agg": agg,
                            "llama_model": llama_model,
                            "desc_seg": desc_seg,
                            "alpha": alpha,
                            **r2_dict,
                        }
                    )
    r2_df = pd.DataFrame(res_list)
    return r2_df


def ensure_consistent(id_df, search_dict, num_layers, num_repeats=10, num_splits=3):
    """
    Ensure consistency within a given permutation/repeat/split for some combination
    of impact/num_layers.
    """
    for repeat in range(num_repeats):
        series_list = []
        for agg in search_dict["agg"]:
            for llama_model in search_dict["llama_model"]:
                if agg in ["max", "avg", "block"]:
                    desc_seg_list = search_dict["seg"]
                else:
                    desc_seg_list = search_dict["desc"]
                for desc_seg in desc_seg_list:
                    if num_layers == 2:
                        alpha_list = search_dict["alpha"]
                    else:
                        alpha_list = [
                            f"{alpha2}_{alpha3}"
                            for alpha2, alpha3 in product(
                                search_dict["alpha"], search_dict["alpha"]
                            )
                        ]
                    for alpha in alpha_list:
                        ### At this point, all unique vals
                        condition_dict = {
                            "repeat": repeat,
                            "num_layers": num_layers,
                            "agg_type": agg,
                            "llama_model": llama_model,
                            "desc_seg": desc_seg,
                            "alpha": str(alpha),
                        }
                        specific_df = get_sub_df(id_df, condition_dict)
                        col_list = [f"perm_id_split_{i}" for i in range(num_splits)]
                        sub_df = specific_df[col_list].reset_index(drop=True)
                        series_list.append(sub_df)
        all_identical = all(series_list[0].equals(s) for s in series_list)
        if not all_identical:
            print(series_list)
        assert all_identical


def build_parser():
    parser = argparse.ArgumentParser(
        prog="perm_results",
    )
    parser.add_argument("--num_layers", required=True, type=int)
    parser.add_argument("--import_dir", required=True)
    parser.add_argument("--search_dict", type=json.loads)
    parser.add_argument("--export_path")
    parser.add_argument("--num_repeats", type=int)
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)
    import_dir = Path(args.import_dir)
    ## Create r2 df
    id_df = pd.read_parquet(import_dir / "id_df_combined.parquet")
    preds_df = pd.read_parquet(import_dir / "pred_df_combined.parquet")
    r2_df = get_r2s_single_iter(
        id_df=id_df,
        preds_df=preds_df,
        search_dict=args.search_dict,
        num_layers=args.num_layers,
        num_repeats=args.num_repeats,
    )
    ## Export r2 df
    # export_path = perm_dir /f"{args.num_layers}_layers/r2_table.parquet"
    export_path = Path(args.export_path)
    if export_path.exists():
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    r2_df.to_parquet(export_path)
    assert export_path.exists()


if __name__ == "__main__":
    main()
