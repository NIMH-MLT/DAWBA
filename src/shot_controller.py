# shot_controller.py
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunningJob:
    impact_type: str
    gpu_block: str
    proc: subprocess.Popen
    log_path: Path


def launch_worker(
    gpu_block,
    outcomes,
    impact_type,
    llm_config,
    base_df_path,
    save_path,
    log_path,
    outcome_desc_type,
    n_shot,
    output_type,
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_block
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        "python",
        "src/shot.py",
        "--outcome_list",
        *outcomes,
        "--impact_type",
        impact_type,
        "--llm_config",
        json.dumps(llm_config),
        "--base_df_path",
        base_df_path,
        "--save_path",
        save_path,
        "--log_path",
        log_path,
        "--outcome_desc_type",
        outcome_desc_type,
        "--n_shot",
        n_shot,
        "--output_type",
        output_type,
    ]

    proc = subprocess.Popen(cmd, env=env)

    print(
        f"START impact_type={impact_type} pid={proc.pid} gpus={gpu_block} log={log_path}",
        flush=True,
    )

    return RunningJob(impact_type, gpu_block, proc, log_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-groups",
        nargs="+",
        default=["0,1,2,3", "4,5,6,7"],
        help="GPU blocks, e.g. '0,1,2,3 4,5,6,7'",
    )
    parser.add_argument("--outcomes", nargs="+", required=True)

    parser.add_argument("--impact_types", nargs="+", required=True)
    parser.add_argument("--llm_config", required=True)
    parser.add_argument("--base_df_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--outcome_desc_type", required=True)
    parser.add_argument("--n_shot", required=True)
    parser.add_argument("--output_type", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # Build deque of all (outcome, seg_type) combinations
    impact_types = deque(args.impact_types)
    gpu_blocks = deque(args.gpu_groups)
    running = []

    config = json.loads(args.llm_config)

    while impact_types or running:
        # launch
        while impact_types and gpu_blocks:
            impact_type = impact_types.popleft()
            block = gpu_blocks.popleft()
            ## Base df
            base_df_path = Path(args.base_df_dir) / f"{impact_type}.parquet"
            ## Save set up
            save_path = Path(args.save_dir) / f"{impact_type}.parquet"
            if save_path.exists():
                save_path.unlink()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ## Log set up
            log_path = Path(args.log_dir) / f"{impact_type}.log"
            if log_path.exists():
                log_path.unlink()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            ### Add to pending list
            running.append(
                launch_worker(
                    gpu_block=block,
                    outcomes=args.outcomes,
                    impact_type=impact_type,
                    llm_config=config,
                    base_df_path=base_df_path,
                    save_path=save_path,
                    log_path=log_path,
                    outcome_desc_type=args.outcome_desc_type,
                    n_shot=args.n_shot,
                    output_type=args.output_type,
                )
            )

        # reap
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

    print("ALL IMPACT_TYPES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
