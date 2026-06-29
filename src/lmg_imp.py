import pandas as pd
from src.utils import col_config_sensitivity, get_prob_col_outcomes
from src.residual_preds import add_split_col
from src.residual_func import prep_train_test
from src.Regressor import ResidualRegressor
import pandas as pd
from src.utils import get_df_path
import numpy as np
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from functools import reduce
import argparse
import json
from pathlib import Path


def get_starting_r2(model_config_dict, X_train, y_train, X_test, y_test):
    """
    Helper for get_perm_feat_imp(); trains a dummy model to extract r^2 from SQ-level predictions
    """
    model = ResidualRegressor(
        **model_config_dict,
        outcome_sub_list=None,
        two_layer_base=None,
    )
    model.fit(X_train, y_train)
    pred_df = model.predict_all(X_test)
    q_preds = pred_df["pred_stage2_total"].values
    r2_score_q = r2_score(y_test, q_preds)
    return r2_score_q


def lmg_imp_wrapper(
    config_dict, n_cv_repeats, n_cv_splits, seed, n_lmg_samples, df_path, n_jobs
):
    impact_type = config_dict["impact_type"]
    df = pd.read_parquet(df_path)
    all_perm_imps = []
    jobs = []
    for repeat in range(n_cv_repeats):
        rand_state_for_split = seed + repeat
        df_w_splits = add_split_col(
            df=df,
            impact_type=config_dict["impact_type"],
            n_splits=n_cv_splits,
            random_state=rand_state_for_split,
        )
        for split_id in range(n_cv_splits):
            X_train, X_test, y_train, y_test = prep_train_test(
                base_df=df_w_splits,
                split_id=split_id,
                impact_type=impact_type,
                agg_type=config_dict["agg_type"],
                desc_seg_type=config_dict["desc_seg_type"],
            )
            jobs.append((X_train, X_test, y_train, y_test, repeat, split_id))
    all_perm_imps = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(get_perm_feat_imp)(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_config_dict=config_dict,
            n_samples=n_lmg_samples,
            random_state=seed,
            repeat=repeat,
            split_id=split_id,
        )
        for X_train, X_test, y_train, y_test, repeat, split_id in jobs
    )
    # merge dfs
    all_perm_imp_df = reduce(
        lambda left, right: pd.merge(left, right, on="outcome"), all_perm_imps
    )
    all_perm_imp_df = all_perm_imp_df.set_index("outcome")
    all_perm_imp_df = all_perm_imp_df.copy().assign(
        mean=lambda df: df.mean(axis=1).round(4),
        std=lambda df: df.std(axis=1).round(4),
    )
    # all_perm_imp_df['mean'] = all_perm_imp_df.mean(axis=1).round(4)
    # all_perm_imp_df['std'] = all_perm_imp_df.std(axis=1).round(4)

    return all_perm_imp_df


def get_perm_feat_imp(
    X_train,
    y_train,
    X_test,
    y_test,
    model_config_dict,
    n_samples,
    random_state,
    repeat=None,
    split_id=None,
):
    """
    Calculate predictive value of each outcome by permuting the order of text outcomes
    n_samples times. Each permutation loops through a shuffled order of feats and iteratively adds
    features + records added r^2 to previous iteration w/o the outcome. Outputs pandas DF containing
    avg incremental r^2 of each outcome over the n_samples permutations.
    """
    _, outcome_list = get_prob_col_outcomes(
        agg_type=model_config_dict["agg_type"],
        desc_seg_type=model_config_dict["desc_seg_type"],
    )

    rng = np.random.default_rng(random_state)

    perm_list = [tuple(rng.permutation(outcome_list)) for _ in range(n_samples)]

    def run_perm(perm):
        contributions = {outcome: 0.0 for outcome in outcome_list}
        active = []
        r2_prev = get_starting_r2(model_config_dict, X_train, y_train, X_test, y_test)
        tot_r2 = r2_prev

        for outcome in perm:
            active.append(outcome)
            # Fit new model
            new_model = ResidualRegressor(
                **model_config_dict,
                outcome_sub_list=active.copy(),
                two_layer_base=None,
            )
            new_model.fit(X_train, y_train)
            # Get new r2
            pred = new_model.predict(X_test)
            r2_new = r2_score(y_test, pred)
            # Update trackers
            delta = r2_new - r2_prev
            contributions[outcome] += delta
            tot_r2 += delta
            r2_prev = r2_new
        # Sum of all incrimental r2 should = r2 of full model (last run of outer-loop ^)
        assert tot_r2 == r2_new, f"TOT: {tot_r2}; NEW: {r2_new}"
        return contributions

    ## Aggregate across outcome perms
    all_conts = [run_perm(perm) for perm in perm_list]
    lmg = {
        outcome: np.round(np.mean([r[outcome] for r in all_conts]), 4)
        for outcome in outcome_list
    }
    # Make pandas df
    lmg_col = "lmg"
    if repeat is not None and split_id is not None:
        lmg_col += f"_{repeat}_{split_id}"
    res_dict = {
        "outcome": list(lmg.keys()),
        lmg_col: list(lmg.values()),
    }
    res_df = (
        pd.DataFrame(res_dict)
        .sort_values(lmg_col, ascending=False)
        .reset_index(drop=True)
    )
    return res_df


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lmg_importance",
        description="Get avg LMG importance for each text outcome",
    )
    parser.add_argument("--config_dict", type=json.loads)
    parser.add_argument("--n_cv_repeats", type=int)
    parser.add_argument("--n_cv_splits", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_lmg_samples", type=int)
    parser.add_argument("--df_path", type=Path)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--export_path", type=Path)
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)
    all_perm_imp_df = lmg_imp_wrapper(
        config_dict=args.config_dict,
        n_cv_repeats=args.n_cv_repeats,
        n_cv_splits=args.n_cv_repeats,
        seed=args.seed,
        n_lmg_samples=args.n_lmg_samples,
        df_path=args.df_path,
        n_jobs=args.n_jobs,
    )
    export_path = args.export_path
    if export_path.exists():
        print(f"OVER-WRITING result at path: {export_path}")
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    all_perm_imp_df.to_parquet(export_path)


if __name__ == "__main__":
    main()
