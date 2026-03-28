# swarm.py
import subprocess
from pathlib import Path
from src.config import BASE_PATH, SEED


####################################################
################## SWARM SPECIFIC ##################
####################################################
def get_len_swarm(swarm_path):
    with open(swarm_path, "r") as f:
        line_count = sum(1 for line in f)
    return min(line_count, 20)


def run_swarms(*_, agg_list, impact_list, swarm_log_dir, cmd_dir, n_jobs, n_threads):
    for agg_type in agg_list:
        for impact_type in impact_list:
            log_dir = swarm_log_dir / agg_type / impact_type
            log_dir.mkdir(parents=True, exist_ok=True)
            swarm_time = "0:05:0"
            gb = 12
            swarm_path = cmd_dir / agg_type / f"{impact_type}.swarm"
            num_commands = get_len_swarm(swarm_path=swarm_path)
            swarm_cmd = f"swarm --time={swarm_time} -g {gb} -t {(2*n_jobs*n_threads)} --logdir={log_dir}  --job-name={agg_type.upper()}-{impact_type.upper()} --partition=quick -b {num_commands} {swarm_path}"

            subprocess.run(swarm_cmd, shell=True)


####################################################
################# MAKE CMD STRINGS #################
####################################################
SPLIT_TOL = 0.2
N_SPLITS = 3
N_REPEATS = 10


def create_cmd_str(
    impact_type,
    agg_type,
    alpha_val,
    df_path,
    base_save_dir,
    seed,
    n_repeats,
    split_tol,
    n_jobs,
    n_threads,
    max_iter=10000,
):
    """
    Create command for lr_preds

    TODO: Check if good file already exists for passed in configuration
    """
    cmd_str = f"export PYTHONPATH={BASE_PATH}; \
			export OMP_NUM_THREADS={n_threads}; \
			uv run python -m src.elastic_preds \
			--impact_type {impact_type} \
			--agg_type {agg_type}\
			--alpha_val {alpha_val}\
			--df_path {df_path}\
			--base_save_dir {base_save_dir}\
			--seed {seed}\
			--n_repeats {n_repeats}\
			--split_tol {split_tol}\
			--n_splits {N_SPLITS}\
			--n_jobs {n_jobs}\
			--max_iter {max_iter}"
    return " ".join(cmd_str.split())


def make_reduced(impact_type, alpha_val, base_save_dir, n_jobs, n_threads):
    df_path = BASE_PATH / "data" / "cleaned" / "processed" / f"{impact_type}.parquet"
    save_dir = base_save_dir / "reduced" / impact_type / f"alpha_{alpha_val}"
    cmd_str = create_cmd_str(
        impact_type=impact_type,
        alpha_val=alpha_val,
        df_path=df_path,
        base_save_dir=save_dir,
        seed=SEED,
        n_repeats=N_REPEATS,
        split_tol=SPLIT_TOL,
        n_splits=N_SPLITS,
        n_jobs=n_jobs,
        n_threads=n_threads,
        agg_type="reduced",
    )
    return cmd_str


def make_full_lr(
    seg_types,
    model_name,
    impact_type,
    alpha_val,
    agg_type,
    base_save_dir,
    n_jobs,
    n_threads,
    max_iter=50000,  # need more iters to converge bc of higher dimension w/ embeddings
):
    cmd_list = []
    for seg_type in seg_types:
        df_path = (
            BASE_PATH
            / "data"
            / "w_probs"
            / model_name
            / seg_type
            / f"{impact_type}.parquet"
        )
        save_dir = (
            base_save_dir
            / agg_type
            / impact_type
            / model_name
            / seg_type
            / f"alpha_{alpha_val}"
        )
        cmd_str = create_cmd_str(
            impact_type=impact_type,
            alpha_val=alpha_val,
            df_path=df_path,
            base_save_dir=save_dir,
            seed=SEED,
            n_repeats=N_REPEATS,
            split_tol=SPLIT_TOL,
            n_jobs=n_jobs,
            max_iter=max_iter,  # need more iters to converge bc of higher dimension w/ embeddings
            agg_type=agg_type,
            n_threads=n_threads,
        )
        cmd_list.append(cmd_str)
    return cmd_list


def make_full_shot(
    model_name,
    impact_type,
    alpha_val,
    base_save_dir,
    n_jobs,
    n_threads,
    temp=0.6,
    pred_type="proba-limited",
    desc_type="augment",
    max_iter=50000,  # need more iters to converge bc of higher dimension w/ embeddings
):
    df_path = (
        BASE_PATH
        / "data"
        / "w_0_shot_probs"
        / f"temp_{temp}"
        / pred_type
        / model_name
        / desc_type
        / f"{impact_type}.parquet"
    )
    save_dir = (
        base_save_dir
        / "0_shot"
        / impact_type
        / model_name
        / f"output_{pred_type}"
        / f"desc_{desc_type}"
        / f"alpha_{alpha_val}"
    )
    cmd_str = create_cmd_str(
        impact_type=impact_type,
        alpha_val=alpha_val,
        df_path=df_path,
        base_save_dir=save_dir,
        seed=SEED,
        n_repeats=N_REPEATS,
        split_tol=SPLIT_TOL,
        n_jobs=n_jobs,
        n_threads=n_threads,
        max_iter=max_iter,
        agg_type="0_shot",
    )
    return cmd_str
