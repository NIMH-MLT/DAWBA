# swarm.py
import subprocess
from src.config import BASE_PATH
import shlex
import json


####################################################
################## SWARM SPECIFIC ##################
####################################################
def get_len_swarm(swarm_path):
    with open(swarm_path, "r") as f:
        line_count = sum(1 for line in f)
    return min(line_count, 20)


def run_perm_swarm(
    swarm_log_dir,
    cmd_dir,
    scratch_gb=10,
    n_batch=1,
    process_gb=2,
    num_cpu=8,
    partition="quick",
    swarm_time="1:00:00",
):
    for impact_swarm_path in cmd_dir.iterdir():
        impact_name = impact_swarm_path.stem
        log_dir = swarm_log_dir / impact_name
        log_dir.mkdir(parents=True, exist_ok=True)
        swarm_cmd = f"swarm --time={swarm_time} --gres=lscratch:{scratch_gb} -g {process_gb} -t {num_cpu} --logdir={log_dir}  --job-name={impact_name}_preds_perm --partition={partition} -b {n_batch} {impact_swarm_path}"
        print(swarm_cmd)
        subprocess.run(swarm_cmd, shell=True)


def run_perm_r2_swarm(impact_list, log_dir, cmd_dir, swarm_time, n_batch, partition):
    for impact_type in impact_list:
        swarm_log_dir = log_dir / impact_type
        swarm_path = cmd_dir / f"{impact_type}.swarm"
        gb = 6
        num_threads = 6
        swarm_cmd = f"swarm --time={swarm_time} -g {gb} -t {num_threads} --logdir={swarm_log_dir}  --job-name=perm_r2_{impact_type} --partition={partition} -b {n_batch} {swarm_path}"
        subprocess.run(swarm_cmd, shell=True)


def run_lmg_swarm(cmd_path, swarm_time, gb, num_threads, log_dir, partition, n_batch):
    swarm_cmd = f"swarm --time={swarm_time} -g {gb} -t {num_threads} --logdir={log_dir}  --job-name=lmg_imp --partition={partition} -b {n_batch} {cmd_path}"
    subprocess.run(swarm_cmd, shell=True)


####################################################
################# MAKE CMD STRINGS #################
####################################################
SPLIT_TOL = 0.2
N_SPLITS = 3
N_REPEATS = 10


def make_perm_level(
    param_dict, config_dict, df_dir, scratch_dir, perm_dir, impact_type
):
    cmd_str_list = [
        f"export PYTHONPATH={BASE_PATH};",
        "uv run",
        "python -m src.residual_wrapper",
        "--param_dict",
        shlex.quote(json.dumps(param_dict)),
        "--config_dict",
        shlex.quote(json.dumps(config_dict)),
        "--df_dir",
        str(df_dir),
        "--scratch_dir",
        str(scratch_dir),
        "--perm_dir",
        str(perm_dir),
        "--impact_type",
        impact_type,
    ]
    return " ".join(cmd_str_list)


def create_lmg_cmd(
    config_dict,
    n_cv_repeats,
    n_cv_splits,
    seed,
    n_lmg_samples,
    df_path,
    n_jobs,
    export_path,
):
    import shlex
    import json

    cmd_str_list = [
        f"export PYTHONPATH={BASE_PATH};",
        f"export OMP_NUM_THREADS={n_jobs + 4};",
        "uv run",
        "python -m src.lmg_imp",
        "--config_dict",
        shlex.quote(json.dumps(config_dict)),
        "--n_cv_repeats",
        str(n_cv_repeats),
        "--n_cv_splits",
        str(n_cv_splits),
        "--seed",
        str(seed),
        "--n_lmg_samples",
        str(n_lmg_samples),
        "--df_path",
        str(df_path),
        "--n_jobs",
        str(n_jobs),
        "--export_path",
        str(export_path),
    ]
    return " ".join(cmd_str_list)
