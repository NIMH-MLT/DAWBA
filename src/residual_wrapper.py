from src.residual_preds import loop_through_resid_params
from itertools import product
import argparse
import json
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from src.utils import get_df_path


def is_good_config(desc_seg, model_name, prob_agg):
    if desc_seg in ["zero_shot", "few_shot"]:
        if prob_agg not in ["binary", "scale"]:
            return False
    # Qwen model & scale agg must use ^ desc_seg
    elif model_name == "qwen" or prob_agg == "scale":
        return False
    return True


def _run_one_config(
    num_layers,
    agg_type,
    llm_model,
    desc_seg,
    alpha_list,
    df_dir,
    scratch_dir,
    perm_dir,
    config_dict,
    impact_type,
):
    df_path = get_df_path(
        df_dir=df_dir,
        llm_model=llm_model,
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
        / llm_model
        / desc_seg
    )
    loop_through_resid_params(
        impact_type=impact_type,
        agg_type=agg_type,
        desc_seg_type=desc_seg,
        df_path=df_path,
        base_save_dir=save_dir,
        n_splits=config_dict["n_splits"],
        num_layers=num_layers,
        perm_dir=perm_dir,
        linear_type="ridge",
        shuffle=config_dict["shuffle"],
        alpha_list=alpha_list,
        max_iter=config_dict["max_iter"],
        seed=config_dict["seed"],
        n_repeats=config_dict["n_repeats"],
        split_tol=config_dict["split_tol"],
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
            [1,2,3]
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
        shuffle
    """
    jobs = []
    for num_layers, agg_type, llm_model in product(
        param_dict["layers"], param_dict["agg"], param_dict["model"]
    ):
        desc_seg_list = (
            param_dict["seg"]
            if agg_type in ["max", "avg", "block"]
            else param_dict["desc"]
        )
        for desc_seg in desc_seg_list:
            if is_good_config(desc_seg, llm_model, agg_type):
                jobs.append((num_layers, agg_type, llm_model, desc_seg))
    Parallel(n_jobs=len(jobs), verbose=10)(
        delayed(_run_one_config)(
            num_layers=num_layers,
            agg_type=agg_type,
            llm_model=llm_model,
            desc_seg=desc_seg,
            alpha_list=param_dict["alpha"],
            df_dir=df_dir,
            scratch_dir=scratch_dir,
            perm_dir=perm_dir,
            config_dict=config_dict,
            impact_type=impact_type,
        )
        for num_layers, agg_type, llm_model, desc_seg in jobs
    )


def aggregate_across_params(param_dict, base_import_dir, n_repeats, impact_type):
    """
    Aggregate exported files for individual model configs/repeats into single files.
    Labels each sub-df with relevant config values + repeat iteration.
    Also aggregates coefs.parquet alongside preds and perm IDs.
    """
    pred_list = []
    id_list = []
    coef_list = []

    for repeat in range(n_repeats):
        id_check_list = []
        for num_layers, agg_type, llm_model in product(
            param_dict["layers"],
            param_dict["agg"],
            param_dict["model"],
        ):
            desc_seg_list = (
                param_dict["seg"]
                if agg_type in ["max", "avg", "block"]
                else param_dict["desc"]
            )
            two_layer_base_list = ["S", "Q"] if num_layers == 2 else ["None"]
            for desc_seg in desc_seg_list:
                if not is_good_config(
                    desc_seg=desc_seg,
                    model_name=llm_model,
                    prob_agg=agg_type,
                ):
                    continue
                # alpha_list = (
                #     param_dict["alpha"]
                #     if num_layers in [1, 2]
                #     else [
                #         f"{a2}_{a3}"
                #         for a2, a3 in product(param_dict["alpha"], param_dict["alpha"])
                #     ]
                # )
                # for alpha in alpha_list:
                for two_layer_base in two_layer_base_list:
                    if num_layers == 1 or (num_layers == 2 and two_layer_base == "S"):
                        alpha_list = param_dict["alpha"]
                    else:
                        alpha_list = [
                            f"{a2}_{a3}"
                            for a2, a3 in product(
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
                            / llm_model
                            / desc_seg
                            / f"alpha_{alpha}"
                            / f"repeat_{repeat}"
                            / f"2_layer_base_{two_layer_base}"
                        )

                        # ── shared config labels ──────────────────────────────────
                        meta = dict(
                            repeat=repeat,
                            num_layers=num_layers,
                            agg_type=agg_type,
                            llm_model=llm_model,
                            desc_seg=desc_seg,
                            alpha=str(alpha),
                            two_layer_base=two_layer_base,
                        )

                        ## Preds ##
                        df_preds = pd.read_parquet(import_dir / "output_preds.parquet")
                        pred_list.append(df_preds.assign(**meta))

                        ## IDs ##
                        df_ids = pd.read_parquet(import_dir / "perm_ids.parquet")
                        id_check_list.append(df_ids)
                        id_list.append(df_ids.assign(**meta))

                        ## Coefs ##
                        df_coefs = pd.read_parquet(import_dir / "coefs.parquet")
                        coef_list.append(df_coefs.assign(**meta))

        # Ensure permuted IDs are consistent across all model configurations for this repeat
        assert all(id_check_list[0].equals(s) for s in id_check_list)
    # Concat lists of dfs
    pred_df_combined = pd.concat(pred_list, ignore_index=True)
    id_df_combined = pd.concat(id_list, ignore_index=True)
    coef_df_combined = pd.concat(coef_list, ignore_index=True)
    return pred_df_combined, id_df_combined, coef_df_combined


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
        description="For each alpha, get elasticnet regression results for each fold",
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
    pred_df_combined, id_df_combined, coef_df_combined = aggregate_across_params(
        param_dict=args.param_dict,
        base_import_dir=scratch_dir / perm_dir.name,
        n_repeats=args.config_dict["n_repeats"],
        impact_type=args.impact_type,
    )

    export_path_preds = perm_dir / args.impact_type / "pred_df.parquet"
    export_path_ids = perm_dir / args.impact_type / "id_df.parquet"
    export_path_coefs = perm_dir / args.impact_type / "coef_df.parquet"

    export_df(export_path_preds, pred_df_combined)
    export_df(export_path_ids, id_df_combined)
    export_df(export_path_coefs, coef_df_combined)
    print("DONE!")


if __name__ == "__main__":
    main()
