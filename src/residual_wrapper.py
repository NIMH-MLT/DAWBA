# residual_wrapper.py

from src.utils import get_df_path
from src.residual_preds import loop_through_resid_params

from itertools import product
import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed


class ParquetStreamWriter:
    """

    Append DataFrames to a single parquet file as row groups.

    **Written in collaboration w/ Claude Opus 4.8**

    """

    def __init__(self, path, buffer_rows=200_000, expected_columns=None):
        path.parent.mkdir(exist_ok=True, parents=True)
        if path.exists():
            print(f"REMOVING existing export path at: {path}")
            path.unlink()
        self.path = path
        self.writer = None
        self.schema = None
        self.expected_columns = expected_columns
        self._buf = []
        self._buf_rows = 0
        self.buffer_rows = buffer_rows  # avoid thousands of tiny row groups

    def write(self, df):
        self._buf.append(df)
        self._buf_rows += len(df)
        if self._buf_rows >= self.buffer_rows:
            self._flush()

    def _flush(self):
        """
        Aggregate buffer into single df, write to parquet, then flush buffer
        """
        if not self._buf:
            return
        chunk = pd.concat(self._buf, ignore_index=True)
        self._buf.clear()
        self._buf_rows = 0

        if self.expected_columns is not None:
            target_cols = self.expected_columns
        elif self.schema is not None:
            target_cols = list(self.schema.names)
        else:
            target_cols = None

        if target_cols is not None:
            extra = [c for c in chunk.columns if c not in target_cols]
            if extra:
                raise ValueError(
                    f"{self.path.name}: unexpected columns {extra} not in "
                    f"schema fixed at first flush; pass expected_columns "
                    f"covering the full union."
                )
            chunk = chunk.reindex(columns=target_cols)  # reorder cols + fill NAs
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema)
        else:
            # Guard against dtype drift between configs (e.g. int vs float)
            table = table.cast(self.schema)
        self.writer.write_table(table)

    def close(self):
        self._flush()
        if self.writer is not None:
            self.writer.close()
            assert self.path.exists()


def is_good_config(desc_seg, model_name, prob_agg, temp):
    """
    Only need to check in case of shot. LR case is dealt with at higher level
    """
    ALLOWED_DICT = {
        "impactmeLR": {
            "desc_seg": ["turns", "pt_noshort"],
            "model_name": "Llama-3.3-70B-Instruct",
            "temp": 0.6,
        },
        "new_shot": {
            "desc_seg": ["shot_zero", "shot_few"],
            "model_name": ["Llama-3.3-70B-Instruct", "gemini-2.5-pro", "Qwen3.6-27B"],
            "temp": {
                "gemini-2.5-pro": [0, 0.4, 0.8, 1.2, 1.6, 2.0],
                "Llama-3.3-70B-Instruct": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "Qwen3.6-27B": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            },
        },
        "og_shot": {
            "desc_seg": ["original", "expanded", "combined"],
            "model_name": "Llama-3.3-70B-Instruct",
            "temp": 0.6,
        },
    }
    # impactme LR
    if prob_agg in ["block", "avg", "max"]:
        if (
            desc_seg not in ALLOWED_DICT["impactmeLR"]["desc_seg"]
            or model_name != ALLOWED_DICT["impactmeLR"]["model_name"]
            or temp != ALLOWED_DICT["impactmeLR"]["temp"]
        ):
            return False
    # case of new shot iteration
    elif prob_agg in ["binary-new", "scale-new"]:
        if (
            desc_seg not in ALLOWED_DICT["new_shot"]["desc_seg"]
            or model_name not in ALLOWED_DICT["new_shot"]["model_name"]
            or temp not in ALLOWED_DICT["new_shot"]["temp"][model_name]
        ):
            return False
    # case of old shot iteration
    elif prob_agg in ["proba-limited-og", "binary-og"]:
        if (
            desc_seg not in ALLOWED_DICT["og_shot"]["desc_seg"]
            or model_name != ALLOWED_DICT["og_shot"]["model_name"]
            or temp != ALLOWED_DICT["og_shot"]["temp"]
        ):
            return False
        ## Additionally, skip og binary in case of expanded
        # since its covered by new shot iteration
        elif (desc_seg == "expanded") & (prob_agg == "binary-og"):
            return False
    else:
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
    temp,
):
    df_path = get_df_path(
        df_dir=df_dir,
        llm_model=llm_model,
        agg_type=agg_type,
        desc_seg_type=desc_seg,
        impact_type=impact_type,
        temp=temp,
    )
    save_dir = (
        scratch_dir
        / perm_dir.name
        / f"{num_layers}_layers"
        / "ridge"
        / agg_type
        / impact_type
        / llm_model
        / f"temp_{temp}"
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
        temp: list[float]
            Vals used for 0/few-shot
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
        if agg_type in ["max", "avg", "block"]:
            desc_seg_list = param_dict["seg"]
            temp_list = [0.6]  # temp used for all prelim extraction
        else:
            desc_seg_list = param_dict["desc"]
            temp_list = param_dict["temp"]
            if llm_model == "gemini-2.5-pro":
                temp_list = [val * 2 for val in temp_list]  # gemini on scale [0,2]
        for desc_seg, temp in product(desc_seg_list, temp_list):
            if is_good_config(desc_seg, llm_model, agg_type, temp):
                jobs.append((num_layers, agg_type, llm_model, desc_seg, temp))
    tot_jobs = len(jobs)
    print(f"total jobs: {tot_jobs}")
    n_jobs = min(tot_jobs, 14)
    Parallel(n_jobs=n_jobs, verbose=10)(
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
            temp=temp,
        )
        for num_layers, agg_type, llm_model, desc_seg, temp in jobs
    )


def aggregate_and_export(
    param_dict,
    base_import_dir,
    n_repeats,
    impact_type,
    export_path_preds,
    export_path_ids,
    export_path_coefs,
):
    """
    Aggregate exported files for individual model configs/repeats into single files.
    Labels each sub-df with relevant config values + repeat iteration.
    Also aggregates coefs.parquet alongside preds and perm IDs.

    Exports directly into final parquet files.
    """
    META_COLS = [
        "repeat",
        "split_id",
        "num_layers",
        "agg_type",
        "llm_model",
        "desc_seg",
        "temp",
        "alpha",
        "two_layer_base",
    ]
    PRED_COLS = [
        "DAWBA_ID",
        "true",
        "pred_stage1",
        "pred_stage2_increment",
        "pred_stage2_total",
        "pred_stage3_increment",
        "pred_stage3_total",
    ] + META_COLS
    pred_writer = ParquetStreamWriter(export_path_preds, expected_columns=PRED_COLS)
    id_writer = ParquetStreamWriter(export_path_ids)
    coef_writer = ParquetStreamWriter(export_path_coefs)
    try:
        for repeat in range(n_repeats):
            ref_ids = None
            # id_check_list = []
            for num_layers, agg_type, llm_model in product(
                param_dict["layers"],
                param_dict["agg"],
                param_dict["model"],
            ):
                if agg_type in ["max", "avg", "block"]:
                    desc_seg_list = param_dict["seg"]
                    temp_list = [0.6]  # temp used for all prelim extraction
                else:
                    desc_seg_list = param_dict["desc"]
                    temp_list = param_dict["temp"]
                    if llm_model == "gemini-2.5-pro":
                        temp_list = [
                            val * 2 for val in temp_list
                        ]  # gemini on scale [0,2]
                for desc_seg, temp in product(desc_seg_list, temp_list):
                    if not is_good_config(desc_seg, llm_model, agg_type, temp):
                        continue
                    two_layer_base_list = ["S", "Q"] if num_layers == 2 else ["None"]
                    import_dir_specific = (
                        base_import_dir
                        / f"{num_layers}_layers"
                        / "ridge"
                        / agg_type
                        / impact_type
                        / llm_model
                        / f"temp_{temp}"
                        / desc_seg
                    )

                    for two_layer_base in two_layer_base_list:
                        if num_layers == 1 or (
                            num_layers == 2 and two_layer_base == "S"
                        ):
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
                                import_dir_specific
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
                                temp=temp,
                                alpha=str(alpha),
                                two_layer_base=two_layer_base,
                            )

                            ## Preds ##
                            df_preds = pd.read_parquet(
                                import_dir / "output_preds.parquet"
                            )
                            pred_writer.write(df_preds.assign(**meta))
                            del df_preds  # free space

                            ## IDs ##
                            df_ids = pd.read_parquet(import_dir / "perm_ids.parquet")
                            if ref_ids is None:
                                ref_ids = df_ids
                            else:
                                assert ref_ids.equals(
                                    df_ids
                                ), f"perm ID mismatch at {import_dir}"
                            id_writer.write(df_ids.assign(**meta))
                            del df_ids  # free space

                            ## Coefs ##
                            df_coefs = pd.read_parquet(import_dir / "coefs.parquet")
                            coef_writer.write(df_coefs.assign(**meta))
    finally:
        pred_writer.close()
        id_writer.close()
        coef_writer.close()


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
    parser.add_argument("--df_dir", type=Path)
    parser.add_argument("--scratch_dir", type=Path)
    parser.add_argument("--perm_dir", type=Path)
    parser.add_argument("--impact_type")
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)

    ## Perm shuffle str --> extract from config dict
    args.config_dict["shuffle"] = (
        None if args.config_dict["shuffle"] == "none" else args.config_dict["shuffle"]
    )
    # This runs to get raw preds on each param combo
    print("Looping through params...")
    loop_through_params(
        param_dict=args.param_dict,
        df_dir=args.df_dir,
        scratch_dir=args.scratch_dir,
        perm_dir=args.perm_dir,
        config_dict=args.config_dict,
        impact_type=args.impact_type,
    )
    ### Aggregate + export ###
    print("Aggregating and exporting...")
    aggregate_and_export(
        param_dict=args.param_dict,
        base_import_dir=args.scratch_dir / args.perm_dir.name,
        n_repeats=args.config_dict["n_repeats"],
        impact_type=args.impact_type,
        export_path_preds=args.perm_dir / args.impact_type / "pred_df.parquet",
        export_path_ids=args.perm_dir / args.impact_type / "id_df.parquet",
        export_path_coefs=args.perm_dir / args.impact_type / "coef_df.parquet",
    )
    print("DONE!")


if __name__ == "__main__":
    main()
