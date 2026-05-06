from src.residual_preds import loop_through_resid_params
from itertools import product
import argparse
import json
from pathlib import Path
import pandas as pd


def get_df_path(df_dir, llama_model, agg_type, desc_seg_type, impact_type):
    if agg_type in ["avg", "max", "block"]:
        intermediate_dir = "w_probs"
    elif agg_type in ["binary", "proba-limited"]:
        intermediate_dir = f"w_0_shot_probs/temp_0.6/{agg_type}"
    else:
        raise ValueError(f"Unreccognized agg type: {agg_type}")
    return (
        df_dir
        / intermediate_dir
        / llama_model
        / desc_seg_type
        / f"{impact_type}.parquet"
    )


def loop_through_params(
    param_dict, df_dir, scratch_dir, perm_dir, config_dict, impact_type
):
    """
    For each model configuration/10 repeats, fit and predict linear model over CV folds
    Temporarily export to scratch_dir

    Params
    ------
    param_dict: dict{}
        layers: list[int]
            [2,3]
        agg: str
            ['max', 'avg', etc]
        llama: str
            ['Llama-3.3-70B-Instruct', etc]
        desc: str; used for 0-shot aggs
            ['desc_original', etc]
        seg: str; used for impactme LR aggs
            ['pt_noshort', etc]
        alpha: list[float] --> possible alpha vals for ridge regression
            [0.5, 1.0, etc]
        impact_type: str
            One of ['parent', 'self']
    df_dir: pathlib.Path
        Path to dir of base df containing X (Questionnaire/Text probs) and y (impact score) data
    scratch_dir: pathlib.Path
        Path to dir that will hold temporary file writes
    perm_dir: pathlib.Path
        For the purpose of this fucntion, raw path to dir holding seed used for permutation
        perm_dir.name should be appended to scratch_dir to save temporary results
    config_dict: dict{}
        max_iter
        seed
        n_repeats
        split_tol
        n_splits
        n_jobs
        shuffle
    """
    tot = len(param_dict["layers"]) * len(param_dict["agg"]) * len(param_dict["llama"])
    for cur_iter, (num_layers, agg_type, llama_model) in enumerate(
        product(param_dict["layers"], param_dict["agg"], param_dict["llama"])
    ):
        print(f"\t{cur_iter+1} / {tot}")
        if agg_type in ["max", "avg", "block"]:
            desc_seg_list = param_dict["seg"]
        else:
            desc_seg_list = param_dict["desc"]
        for desc_seg in desc_seg_list:
            df_path = get_df_path(
                df_dir=df_dir,
                llama_model=llama_model,
                agg_type=agg_type,
                desc_seg_type=desc_seg,
                impact_type=impact_type,
            )
            save_dir = (
                scratch_dir
                / perm_dir.name
                / f"{num_layers}_layers"
                / "ridge"
                / agg_type
                / impact_type
                / llama_model
                / desc_seg
            )
            loop_through_resid_params(
                num_layers=num_layers,
                agg_type=agg_type,
                impact_type=impact_type,
                df_path=df_path,
                base_save_dir=save_dir,  # sctratch dir where results temporarily written to
                alpha_list=param_dict["alpha"],
                l1_list=[None],
                perm_dir=perm_dir,  # actual perm dir where results eventually saved & seed.txt file exists
                linear_type="ridge",
                impactme_outcome=None,
                max_iter=config_dict["max_iter"],
                seed=config_dict["seed"],
                n_repeats=config_dict["n_repeats"],
                split_tol=config_dict["split_tol"],
                n_splits=config_dict["n_splits"],
                n_jobs=config_dict["n_jobs"],
                shuffle=config_dict["shuffle"],
            )


def aggregate_across_params(param_dict, base_import_dir, n_repeats, impact_type):
    """
    Aggregate exported files for individual model configs/repeats and aggregate into single file.
    Label each sub-df with relevant config values (agg, impact, etc) + repeat iteration.
    """

    pred_list = []
    id_list = []
    for repeat in range(n_repeats):
        id_check_list = []
        for num_layers, agg_type, llama_model in product(
            param_dict["layers"],
            param_dict["agg"],
            param_dict["llama"],
        ):
            if agg_type in ["max", "avg", "block"]:
                desc_seg_list = param_dict["seg"]
            else:
                desc_seg_list = param_dict["desc"]
            for desc_seg in desc_seg_list:
                if num_layers == 2:
                    alpha_list = param_dict["alpha"]
                else:
                    alpha_list = [
                        f"{alpha2}_{alpha3}"
                        for alpha2, alpha3 in product(
                            param_dict["alpha"], param_dict["alpha"]
                        )
                    ]
                for alpha in alpha_list:
                    import_dir = (
                        base_import_dir
                        / f"{num_layers}_layers"
                        / "ridge"
                        / agg_type
                        / impact_type
                        / llama_model
                        / desc_seg
                        / f"alpha_{alpha}"
                        / f"repeat_{repeat}"
                    )
                    ## Preds ##
                    df_preds = pd.read_parquet(import_dir / "output_preds.parquet")
                    # Add meta data
                    df_preds_w_config = df_preds.assign(
                        repeat=repeat,
                        num_layers=num_layers,
                        agg_type=agg_type,
                        # impact_type=impact_type,
                        llama_model=llama_model,
                        desc_seg=desc_seg,
                        alpha=str(alpha),
                    )
                    # Append
                    pred_list.append(df_preds_w_config)
                    ## IDS ##
                    df_ids = pd.read_parquet(import_dir / "perm_ids.parquet")
                    # Append raw to check consistency
                    id_check_list.append(df_ids)
                    # Add meta
                    df_ids_w_config = df_ids.assign(
                        repeat=repeat,
                        num_layers=num_layers,
                        agg_type=agg_type,
                        # impact_type=impact_type,
                        llama_model=llama_model,
                        desc_seg=desc_seg,
                        alpha=str(alpha),
                    )
                    # Append for export
                    id_list.append(df_ids_w_config)
        # Ensure permuted IDs consistent across all model configuration trainings for this repeat
        assert all(id_check_list[0].equals(s) for s in id_check_list)
    # Concat lists of dfs
    pred_df_combined = pd.concat(pred_list, ignore_index=True)
    id_df_combined = pd.concat(id_list, ignore_index=True)
    return pred_df_combined, id_df_combined


def export_df(export_path, df):
    export_path.parent.mkdir(exist_ok=True, parents=True)
    if export_path.exists():
        print(f"REMOVING existing export path at: {export_path}")
        export_path.unlink()
    df.to_parquet(export_path)
    assert export_path.exists()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="residual_preds",
        description="For each alpha and l1 value, get elasticnet regression results for each fold",
    )
    parser.add_argument("--param_dict", type=json.loads)
    parser.add_argument("--config_dict", type=json.loads)
    parser.add_argument("--df_dir")
    parser.add_argument("--scratch_dir")
    parser.add_argument("--perm_dir")
    parser.add_argument("--impact_type")
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)

    scratch_dir = Path(args.scratch_dir)
    df_dir = Path(args.df_dir)
    perm_dir = Path(args.perm_dir)
    ## Perm shuffle str --> extract from config dict
    args.config_dict["shuffle"] = (
        None if args.config_dict["shuffle"] == "none" else args.config_dict["shuffle"]
    )
    # This runs to get raw preds on each param combo
    print("Looping through params...")
    loop_through_params(
        param_dict=args.param_dict,
        df_dir=df_dir,
        scratch_dir=scratch_dir,
        perm_dir=perm_dir,
        config_dict=args.config_dict,
        impact_type=args.impact_type,
    )
    ### Aggregate + export ###
    print("Aggregating and exporting...")
    pred_df_combined, id_df_combined = aggregate_across_params(
        param_dict=args.param_dict,
        base_import_dir=scratch_dir / perm_dir.name,
        n_repeats=args.config_dict["n_repeats"],
        impact_type=args.impact_type,
    )
    export_path_preds = perm_dir / args.impact_type / "pred_df_combined.parquet"
    export_path_ids = perm_dir / args.impact_type / "id_df_combined.parquet"
    export_df(export_path_preds, pred_df_combined)
    export_df(export_path_ids, id_df_combined)
    print("DONE!")


if __name__ == "__main__":
    main()
