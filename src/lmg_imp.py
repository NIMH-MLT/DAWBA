from src.utils import (
    get_prob_col_outcomes,
    get_perm_seed,
    expand_text,
)
from src.residual_preds import add_split_col
from src.residual_func import prep_train_test
from src.Regressor import ResidualRegressor

from joblib import Parallel, delayed
import argparse
from functools import reduce
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def single_train_test_lmg(config, n_repeats, df, starting_seed, scale=False):
    #### PREP DATA  ####
    # Get X/y train/test sets
    target_col = f"impact_score_{config['impact_type']}"
    df_expand, prob_outcomes = expand_text(
        df=df,
        agg_type=config["agg_type"],
        desc_seg_type=config["desc_seg_type"],
    )
    # X
    X_train = df_expand.drop(target_col, axis=1)
    if scale:
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(clip=True, feature_range=(0, 1))
        X_train[prob_outcomes] = scaler.fit_transform(X_train[prob_outcomes])
    X_test = X_train
    # y
    y_train = df_expand[target_col].to_numpy()
    y_test = y_train

    #### RUN PERMS ####
    df_list = []
    for repeat in range(n_repeats):
        rand_state = starting_seed + repeat
        # Run feat imp
        lmg_df = get_perm_feat_imp(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_config_dict=config,
            random_state=rand_state,
        )
        lmg_df = lmg_df.rename(
            columns={
                "original": f"original_{repeat}",
                "reversed": f"reversed_{repeat}",
            }
        )
        df_list.append(lmg_df)
    combined_df = reduce(lambda left, right: left.join(right, how="outer"), df_list)
    final_df = pd.DataFrame(
        {
            # avg for each outcome across all perms
            "avg_increase": combined_df.mean(axis=1).round(4),
            # stdev for each outcome across all perms
            "std_increase": combined_df.std(axis=1).round(4),
        }
    )
    return final_df


def agg_over_perms(all_dfs):
    """
    Collapse per-ordering LMG results into a single allocation of explained
    variance across text outcomes.

    Params
    ------
    all_dfs: list[pandas DF]
        One frame per ordering, indexed by outcome. Each must contain ONLY
        columns whose names carry "mean" or "std", and those names must be unique
        across frames. Passing raw wrapper output trips the
        assertion; passing unsuffixed frames raises on overlapping column names in the join.

    Returns
    -------
    pandas DF indexed by outcome, with columns:
        avg_mean: mean incremental r^2 allocated to that outcome, averaged over
            every ordering and its reverse. Sums across outcomes to the text
            stage's increment over the SQ baseline, so entries are readable as
            shares of that increment. May be negative -- an outcome can lower
            out-of-sample r^2.
        std_mean: spread of those same per-ordering estimates. Monte Carlo error,
            i.e. whether enough orderings were sampled for the allocation to have
            settled; NaN when only one ordering is supplied. The per-ordering
            *_std columns (fold-level noise WITHIN an ordering) are a different
            quantity and are dropped.
    """
    if len(all_dfs) == 0:
        raise ValueError("Input list is empty")
    ref = all_dfs[0].index
    for i, df in enumerate(all_dfs[1:], start=1):
        if not df.index.equals(ref):
            missing = sorted(set(ref) - set(df.index))
            extra = sorted(set(df.index) - set(ref))
            raise ValueError(f"""frame {i} has a different outcome index than frame 0 \n
                (missing: {missing}, extra: {extra}). Every ordering must \n
                decompose the same registry, or the shares will not sum to \n
                the increment.""")

    combined_df = reduce(lambda left, right: left.join(right, how="outer"), all_dfs)
    unexpected = [
        c for c in combined_df.columns if not c.split("_")[1] in ("mean", "std")
    ]
    assert (
        not unexpected
    ), f"unexpected columns (subset to aggregates first): {unexpected}"
    mean_cols = [col for col in combined_df.columns if "mean" in col]
    std_cols = [col for col in combined_df.columns if "std" in col]
    assert len(mean_cols) + len(std_cols) == combined_df.shape[1]
    final_df = pd.DataFrame(
        {
            # global mean
            "avg_mean": combined_df[mean_cols].mean(axis=1).round(4),
            # stdev of mean(avg across folds/repeats for a given permutation)
            "std_mean": combined_df[mean_cols].std(axis=1).round(4),
        }
    )
    return final_df


def get_starting_r2(model_config_dict, X_train, y_train, X_test, y_test):
    """
    Helper for get_perm_feat_imp(); trains a dummy model to extract r^2 from SQ-level predictions
    """
    # Ensure no two_layer_base bc hardcode as None
    if (
        "two_layer_base" in model_config_dict.keys()
        or "outcome_sub_list" in model_config_dict.keys()
    ):
        raise ValueError(
            "two_layer_base and outcome_sub_list must not be in model_config_dict"
        )

    prob_col, prob_outcomes = get_prob_col_outcomes(
        agg_type=model_config_dict["agg_type"],
        desc_seg_type=model_config_dict["desc_seg_type"],
    )
    model = ResidualRegressor(
        **model_config_dict,
        # only use one text outcome for speed bc only need SQ
        outcome_sub_list=[prob_outcomes[0]],
        two_layer_base=None,
    )
    model.fit(X_train, y_train)
    pred_df = model.predict_all(X_test)
    q_preds = pred_df["pred_stage2_total"].values
    r2_score_q = r2_score(y_test, q_preds)
    return r2_score_q


def lmg_imp_wrapper(
    config_dict,
    n_cv_repeats,
    n_cv_splits,
    seed,
    df_path,
    n_jobs,
    perm_dir,
):
    """
    For a given permutation iteration, loop through repeats/splits and get avg/stdev
    added % variance explained (r^2; coefficient of determination) of each outcome
    """
    perm_seed = get_perm_seed(perm_dir=perm_dir)
    impact_type = config_dict["impact_type"]
    df = pd.read_parquet(df_path)

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
    lmg_dfs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(get_perm_feat_imp)(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_config_dict=config_dict,
            random_state=perm_seed,
            repeat=repeat,
            split_id=split_id,
        )
        for X_train, X_test, y_train, y_test, repeat, split_id in jobs
    )
    # # merge dfs across splits/repeats
    all_perm_imp_df = reduce(lambda left, right: left.join(right, how="outer"), lmg_dfs)
    # get original/reversed cols
    og_cols = [c for c in all_perm_imp_df.columns if c.startswith("original")]
    rev_cols = [c for c in all_perm_imp_df.columns if c.startswith("reversed")]
    # get mean/stdev across splits/repeats separately
    all_perm_imp_df = all_perm_imp_df.copy().assign(
        original_mean=lambda df: df[og_cols].mean(axis=1).round(4),
        original_std=lambda df: df[og_cols].std(axis=1).round(4),
        reversed_mean=lambda df: df[rev_cols].mean(axis=1).round(4),
        reversed_std=lambda df: df[rev_cols].std(axis=1).round(4),
    )
    return all_perm_imp_df


def get_perm_feat_imp(
    X_train,
    y_train,
    X_test,
    y_test,
    model_config_dict,
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
    # Base and sublist are hard-coded, cant pass
    if (
        "two_layer_base" in model_config_dict.keys()
        or "outcome_sub_list" in model_config_dict.keys()
    ):
        raise ValueError(
            "two_layer_base and outcome_sub_list must not be in model_config_dict"
        )
    _, outcome_list = get_prob_col_outcomes(
        agg_type=model_config_dict["agg_type"],
        desc_seg_type=model_config_dict["desc_seg_type"],
    )

    rng = np.random.default_rng(random_state)
    og_perm = tuple(rng.permutation(outcome_list))

    og_col_name = "original"
    reversed_col_name = "reversed"
    if repeat is not None and split_id is not None:
        og_col_name += f"_{repeat}_{split_id}"
        reversed_col_name += f"_{repeat}_{split_id}"
    perm_dict = {
        og_col_name: og_perm,
        # add reverse order for full coverage
        reversed_col_name: tuple(reversed(og_perm)),
    }

    def run_perm(perm):
        contributions = {outcome: 0.0 for outcome in outcome_list}
        active = []
        r2_start = get_starting_r2(model_config_dict, X_train, y_train, X_test, y_test)
        r2_prev = r2_start
        tot_r2 = r2_start

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
        # Sum of all incremental r2 should (excluding baseline) should equal the difference between the new r2 and the starting r2
        assert np.isclose(sum(contributions.values()), r2_new - r2_start)
        # Sum of all incremental r2 (including baseline) should = r2 of full model (last run of outer-loop ^)
        assert np.isclose(tot_r2, r2_new, atol=1e-8), f"TOT: {tot_r2}; NEW: {r2_new}"
        return contributions

    ## Aggregate across og/reverse perms
    lmg_dict = {list_type: run_perm(perm) for list_type, perm in perm_dict.items()}
    res_df = pd.DataFrame(lmg_dict).rename_axis("outcome").sort_values("outcome")
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
    parser.add_argument("--df_path", type=Path)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--perm_dir", type=Path)
    parser.add_argument("--export_path", type=Path)
    return parser


def main(argv=None):
    ## Parse args
    args = build_parser().parse_args(argv)
    all_perm_imp_df = lmg_imp_wrapper(
        config_dict=args.config_dict,
        n_cv_repeats=args.n_cv_repeats,
        n_cv_splits=args.n_cv_splits,
        seed=args.seed,
        df_path=args.df_path,
        n_jobs=args.n_jobs,
        perm_dir=args.perm_dir,
    )

    export_path = args.export_path
    if export_path.exists():
        print(f"OVER-WRITING result at path: {export_path}")
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    all_perm_imp_df.to_parquet(export_path)


if __name__ == "__main__":
    main()
