# perm_res.py

import argparse
import json
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def get_avg_r2s(unique_dir, search_dict):
    r2_S_list = []
    r2_Q_list = []
    r2_T_list = []
    for repeat in search_dict["repeats"]:
        dfs = [
            pd.read_parquet(unique_dir / repeat / split)
            for split in search_dict["splits"]
        ]
        combined_dfs = pd.concat(dfs)
        y_true = combined_dfs["true"]
        y_pred_S = combined_dfs["pred_stage1"]
        y_pred_Q = combined_dfs["pred_stage2_total"]
        y_pred_T = combined_dfs["pred_stage3_total"]
        r2_S_list.append(r2_score(y_true, y_pred_S))
        r2_Q_list.append(r2_score(y_true, y_pred_Q))
        r2_T_list.append(r2_score(y_true, y_pred_T))
    return {
        "r2_S": np.mean(r2_S_list),
        "r2_Q": np.mean(r2_Q_list),
        "r2_T": np.mean(r2_T_list),
    }


def get_r2s_single_iter(perm_dir, search_dict, num_layers, impact_type):
    ## Ensure consistent permutations across model configs for a given permutation/impact/num_layer combination
    print("Checking consistency...")
    ensure_consistent(
        perm_dir=perm_dir,
        search_dict=search_dict,
        num_layers=num_layers,
        impact_type=impact_type,
    )
    print("\tConsistent!")
    perm_iter = perm_dir.name
    print(f"{perm_iter}...")
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
                    param_sub_path = param_sub_path = (
                        Path(f"{num_layers}_layers")
                        / "ridge"
                        / agg
                        / impact_type
                        / llama_model
                        / desc_seg
                        / f"alpha_{alpha}"
                    )
                    #### AT THIS POINT: all unique values for a given model
                    unique_dir = perm_dir / param_sub_path
                    r2_dict = get_avg_r2s(unique_dir, search_dict)
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


def ensure_consistent(perm_dir, search_dict, num_layers, impact_type):
    for repeat in search_dict["repeats"]:
        for split in search_dict["split_perms"]:
            #### Want to make sure we have consistent permutations for each repeat/split WITHIN permutation/num-layers/impact-type ###
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
                            imp_path = (
                                perm_dir
                                / f"{num_layers}_layers"
                                / "ridge"
                                / agg
                                / impact_type
                                / llama_model
                                / desc_seg
                                / f"alpha_{alpha}"
                                / repeat
                                / split
                            )
                            perm_df = pd.read_parquet(imp_path)
                            perm_ids = perm_df["perm_id_train"]
                            series_list.append(perm_ids)
            all_identical = all(series_list[0].equals(s) for s in series_list)
            if not all_identical:
                print(series_list)
            assert all_identical


def build_parser():
    parser = argparse.ArgumentParser(
        prog="perm_results",
    )
    parser.add_argument("--impact_type", required=True)
    parser.add_argument("--num_layers", required=True, type=int)
    parser.add_argument("--perm_dir", required=True)
    parser.add_argument("--search_dict", type=json.loads)
    parser.add_argument("--export_path")
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)
    perm_dir = Path(args.perm_dir)
    ## Create r2 df
    r2_df = get_r2s_single_iter(
        perm_dir=perm_dir,
        search_dict=args.search_dict,
        num_layers=args.num_layers,
        impact_type=args.impact_type,
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
