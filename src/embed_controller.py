from pathlib import Path
import argparse
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from src.utils import col_config_sensitivity


@dataclass
class RunningJob:
    impact_type: str
    gpu_block: str
    proc: subprocess.Popen
    log_path: Path


def launch_worker(
    gpu_block,
    og_imp_dir,
    impact_type,
    model_id,
    log_dir,
    result_dir,
    text_col_names,
):
    model_name = Path(model_id).name
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_block
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    og_df_path = Path(og_imp_dir) / f"{impact_type}.parquet"
    log_path = Path(log_dir) / f"{impact_type}.log"
    result_path = Path(result_dir) / model_name / f"{impact_type}.parquet"
    cmd = [
        "python",
        "-m",
        "src.embed",
        "--og_df_path",
        str(og_df_path),
        "--text_col_names",
        *text_col_names,
        "--model-id",
        str(model_id),
        "--log-path",
        str(log_path),
        "--result-path",
        str(result_path),
    ]

    proc = subprocess.Popen(cmd, env=env)

    print(
        f"START embedding for impact_type={impact_type} model={model_name} pid={proc.pid} gpus={gpu_block}",
        flush=True,
    )

    return RunningJob(impact_type, gpu_block, proc, log_path)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gpu-groups",
        nargs="+",
        default=["0,1,2,3", "4,5,6,7"],
        help="GPU blocks, e.g. '0,1,2,3 4,5,6,7'",
    )
    p.add_argument("--impact-types", nargs="+", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--og-imp-dir", required=True)
    p.add_argument(
        "--text-col-name",
        required=True,
        help="If set as a column, embed only this column; if `ALL`, embed all configured text columns.",
    )
    p.add_argument("--log-dir", required=True)
    p.add_argument("--result-dir", required=True)
    return p.parse_args()


def main():
    args = build_parser()

    # Dequeu impact types along with gpu blocks
    impact_types = deque(args.impact_types)
    gpu_blocks = deque(args.gpu_groups)
    running = []

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    text_col_name = args.text_col_name

    while impact_types or running:
        # launch
        while impact_types and gpu_blocks:
            impact_type = impact_types.popleft()
            if text_col_name == "ALL":
                original_text_cols = col_config_sensitivity[f"{impact_type}_text"]
                text_col_names = original_text_cols + ["open_text_block"]
            else:
                text_col_names = [text_col_name]
            block = gpu_blocks.popleft()
            running.append(
                launch_worker(
                    gpu_block=block,
                    og_imp_dir=args.og_imp_dir,
                    impact_type=impact_type,
                    text_col_names=text_col_names,
                    model_id=args.model_id,
                    log_dir=log_dir,
                    result_dir=result_dir,
                )
            )
        for idx in reversed(range(len(running))):
            job = running[idx]
            rc = job.proc.poll()
            if rc is not None:
                status = "DONE" if rc == 0 else "FAILED"
                print(
                    f"{status} impact_type={job.impact_type} "
                    f"pid={job.proc.pid} exitcode={rc} gpus={job.gpu_block}",
                    flush=True,
                )
                gpu_blocks.append(job.gpu_block)
                running.pop(idx)

        if running:
            time.sleep(2)

    print("ALL IMPACT TYPES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
