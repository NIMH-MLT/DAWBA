# swarm.py
import subprocess
from pathlib import Path


def run_swarms(*_, agg_list, impact_list, swarm_log_dir, cmd_dir, n_jobs, n_threads):
    for agg_type in agg_list:
        for impact_type in impact_list:
            log_dir = swarm_log_dir / agg_type / impact_type
            log_dir.mkdir(parents=True, exist_ok=True)
            swarm_time = "1:00:0"
            gb = 12
            swarm_path = cmd_dir / agg_type / f"{impact_type}.swarm"
            swarm_cmd = f"swarm --time={swarm_time} -g {gb} -t {(2*n_jobs*n_threads)} --logdir={log_dir}  --job-name={agg_type.upper()}-{impact_type.upper()} --partition=quick {swarm_path}"
            subprocess.run(swarm_cmd, shell=True)
