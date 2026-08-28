import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from baycomp import two_on_single

_METRIC_FNS = {
    "mae": mean_absolute_error,
    "r2": r2_score,
}


def export_data(data_to_export, export_path, remove_existing=False):
    if remove_existing and export_path.exists():
        print(f"REMOVING file at path: {export_path}")
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    if export_path.suffix == ".parquet":
        data_to_export.to_parquet(export_path)
    else:
        raise ValueError(f"Unrecognized export type: {export_path.suffix}")


def get_layer1(r2_df):
    r2s_1 = r2_df[r2_df["num_layers"] == 1]
    if len(r2s_1) == 0:
        return None
    # r2s_1
    best_1Q_row = r2s_1.sort_values(by="r2_Q_mean", ascending=False).iloc[0]
    best_1Q_alpha = best_1Q_row["alpha"].split("_")[0]
    best_1Q_val = best_1Q_row["r2_Q_mean"]
    r2s_1["best_Q_alpha"] = f"{best_1Q_val}_{best_1Q_alpha}"
    r2s_1["diff"] = round(r2s_1["r2_T_mean"] - best_1Q_val, 3)
    r2s_1 = r2s_1.sort_values(by="r2_T_mean", ascending=False)
    return r2s_1


def get_layer2(r2_df, filter_best_Q):
    r2s_2 = r2_df[r2_df["num_layers"] == 2]

    r2s_2_S_base = r2s_2[r2s_2["two_layer_base"] == "S"]
    r2s_2_S_base["diff"] = round(
        r2s_2_S_base["r2_T_mean"] - r2s_2_S_base["r2_S_mean"], 3
    )
    r2s_2_S_base = r2s_2_S_base.sort_values(by="diff", ascending=False)

    r2s_2_Q_base = r2s_2[r2s_2["two_layer_base"] == "Q"]
    best_Q = max(r2s_2_Q_base["r2_Q_mean"])
    if filter_best_Q:
        r2s_2_Q_base = r2s_2_Q_base[np.isclose(r2s_2_Q_base["r2_Q_mean"], best_Q)]
    else:
        r2s_2_Q_base["best_Q"] = best_Q
    r2s_2_Q_base["diff"] = round(r2s_2_Q_base["r2_T_mean"] - best_Q, 3)
    r2s_2_Q_base = r2s_2_Q_base.sort_values(by="diff", ascending=False)

    return r2s_2_S_base, r2s_2_Q_base


def get_layer3(r2_df, filter_best_Q):
    r2s_3 = r2_df[r2_df["num_layers"] == 3]

    best_3Q_row = r2s_3.sort_values(by="r2_Q_mean", ascending=False).iloc[0]
    best_3Q_alpha = best_3Q_row["alpha"].split("_")[0]
    best_3Q_val = best_3Q_row["r2_Q_mean"]
    if filter_best_Q:
        r2s_3 = r2s_3[np.isclose(r2s_3["r2_Q_mean"], best_3Q_val)]
    else:
        r2s_3["best_Q_alpha"] = f"{best_3Q_val}_{best_3Q_alpha}"
    r2s_3["diff"] = round(r2s_3["r2_T_mean"] - best_3Q_val, 3)
    r2s_3 = r2s_3.sort_values(by="diff", ascending=False)
    return r2s_3


def compute_r2s(g):
    scores = {
        "r2_S": r2_score(g["true"], g["pred_stage1"]),
        "r2_Q": r2_score(g["true"], g["pred_stage2_total"]),
        "r2_T": r2_score(g["true"], g["pred_stage3_total"]),
    }
    return pd.Series(scores)


def get_avg_r2(df):
    df = df.fillna("None")
    params = [
        "num_layers",
        "agg_type",
        "llm_model",
        "desc_seg",
        "alpha",
        "two_layer_base",
        "temp",
    ]
    r2_df = (
        df.groupby(params + ["repeat"])
        .apply(compute_r2s)  # compute r2 within repeat
        .reset_index()
        .groupby(params)[["r2_S", "r2_Q", "r2_T"]]
        .agg(["mean", "std"])
        .round(3)
        .reset_index()
    )
    r2_df = r2_df.fillna("None")
    r2_df.columns = ["_".join(c).strip("_") for c in r2_df.columns]
    return r2_df


def get_top_models(r2_df, params, top_n=50, num_layers=3, two_layer_base=None):
    if num_layers == 3:
        r2_n_layer = get_layer3(r2_df, filter_best_Q=True)
        print("Running 3 layer...")
    elif num_layers == 2:
        # only consider 2 layer ST (not QT)
        r2_n_layer_S, r2_n_layer_Q = get_layer2(r2_df, filter_best_Q=True)
        if two_layer_base == "Q":
            r2_n_layer = r2_n_layer_Q
        elif two_layer_base == "S":
            r2_n_layer = r2_n_layer_S
        else:
            raise ValueError(
                f"two_layer_base must be one of ['S', 'Q'], got {two_layer_base}"
            )
        print("Running 2 layer...")
    elif num_layers == 1:
        r2_n_layer = get_layer1(r2_df)
        print("Running 1 layer...")
    else:
        raise ValueError(f"num_layers must be one of [1, 2, 3], got {num_layers}")
    r2_n_layer = r2_n_layer.sort_values(by="diff", ascending=False)
    # Take top 50 model configs
    r2_top = r2_n_layer.iloc[:top_n]
    top_model_params = r2_top.iloc[0][params]
    return r2_top, top_model_params


def get_score(pred_df, score_type, param_vals=None):
    """
    Fold- and repeat-level scores for the stage-3 and stage-2 predictions.

    Params
    ------
    df: pandas DF
        Contains predictions for all models
    score_type: str
        One of ['mae', 'r2']
    param_vals: pd.Series
        Mapping params to specific values for filtering


    Returns
    ------
    score_df_fold_repeats: pandas DF
        score_type value of each repeat in T(folds are concat)
    score_df_repeats: pandas DF
        score_type value of each fold-repeat in T
    score_df_fold_repeats_q: pandas DF
        score_type value of each repeat in Q(folds are concat)
    score_df_repeats_q: pandas DF
        score_type value of each fold-repeat in Q
    """
    if param_vals is not None:
        param_vals = param_vals.fillna("None")
        df_sub = pred_df[
            (pred_df[param_vals.keys().to_list()] == param_vals).all(axis=1)
        ]
        if len(df_sub) == 0:
            raise ValueError("No matching rows found for the given parameter values.")
    else:
        df_sub = pred_df
    metric_fn = _METRIC_FNS[score_type]
    score_df_fold_repeats = (
        df_sub.groupby(["repeat", "split_id"])
        .apply(lambda g: metric_fn(y_pred=g["pred_stage3_total"], y_true=g["true"]))
        .rename(score_type)
        .reset_index()
    )
    score_df_repeats = (
        df_sub.groupby(["repeat"])
        .apply(lambda g: metric_fn(y_pred=g["pred_stage3_total"], y_true=g["true"]))
        .rename(score_type)
        .reset_index()
    )
    score_df_fold_repeats_q = (
        df_sub.groupby(["repeat", "split_id"])
        .apply(lambda g: metric_fn(y_pred=g["pred_stage2_total"], y_true=g["true"]))
        .rename(score_type)
        .reset_index()
    )
    score_df_repeats_q = (
        df_sub.groupby(["repeat"])
        .apply(lambda g: metric_fn(y_pred=g["pred_stage2_total"], y_true=g["true"]))
        .rename(score_type)
        .reset_index()
    )

    return (
        score_df_fold_repeats,
        score_df_repeats,
        score_df_fold_repeats_q,
        score_df_repeats_q,
    )


def compare_model_configs(
    base_pred_df,
    r2_df,
    param_list,
    top_n_models,
    score_type,
    num_layers=3,
    two_layer_base=None,
):
    r2_n_top, top_model_params = get_top_models(
        r2_df,
        param_list,
        top_n=top_n_models,
        num_layers=num_layers,
        two_layer_base=two_layer_base,
    )
    best_score_df_rf, _, _, _ = get_score(
        pred_df=base_pred_df, param_vals=top_model_params, score_type=score_type
    )

    new_rows = []
    for i, row in r2_n_top.reset_index(drop=True).iterrows():
        new_row = row[param_list].to_dict()
        cur_score_df_rf, cur_score_df_r, cur_score_df_rf_q, cur_score_df_r_q = (
            get_score(
                pred_df=base_pred_df, param_vals=row[param_list], score_type=score_type
            )
        )
        if i == 0:
            new_row.update(
                {
                    "P(best better)": 0.0,
                    "P(equal)": 1.0,
                    "P(this better)": 0.0,
                }
            )
        else:
            combined_df = pd.merge(
                best_score_df_rf,
                cur_score_df_rf,
                on=["repeat", "split_id"],
                suffixes=("_best", "_cur"),
            )
            # (p_left, p_rope, p_right), fig = two_on_single(
            p_left, p_rope, p_right = two_on_single(
                x=combined_df[f"{score_type}_best"],
                y=combined_df[f"{score_type}_cur"],
                rope=0.01,
                runs=10,
                plot=False,
            )
            new_row.update(
                {
                    "P(best better)": round(p_left, 2),
                    "P(equal)": round(p_rope, 2),
                    "P(this better)": round(p_right, 2),
                }
            )
        # average across n_repeats * n_folds values
        avg_r2_rf = float(cur_score_df_rf[score_type].mean())
        std_r2_rf = float(cur_score_df_rf[score_type].std())
        avg_r2_rf_q = float(cur_score_df_rf_q[score_type].mean())
        std_r2_rf_q = float(cur_score_df_rf_q[score_type].std())
        # average across n_repeats values (each val calculated within a repeat across folds)
        avg_r2_r = float(cur_score_df_r[score_type].mean())
        std_r2_r = float(cur_score_df_r[score_type].std())
        avg_r2_r_q = float(cur_score_df_r_q[score_type].mean())
        std_r2_r_q = float(cur_score_df_r_q[score_type].std())
        new_row.update(
            {
                f"{score_type} repeats-folds": f"{avg_r2_rf:.3f} ({std_r2_rf:.3f})",
                f"{score_type} repeats": f"{avg_r2_r:.3f} ({std_r2_r:.3f})",
                f"{score_type} repeats-folds- Q": f"{avg_r2_rf_q:.3f} ({std_r2_rf_q:.3f})",
                f"{score_type} repeats- Q": f"{avg_r2_r_q:.3f} ({std_r2_r_q:.3f})",
            }
        )
        new_rows.append(new_row)

    compare_df = pd.DataFrame(new_rows)
    return compare_df


def plot_scatter(error_best, error_cur, i):
    fig, ax = plt.subplots()
    ax.scatter(
        error_best,
        error_cur,
        label="prediction error per participant in each fold/repeat",
        s=20,
        alpha=0.7,
    )
    ax.set_xlabel("error_model_0")
    ax.set_ylabel(f"error_model_{i}")
    plt.show()


def get_cur_pred_error(pred_df, i, row, param_list):
    mask = (pred_df[param_list] == row[param_list]).all(axis=1)
    cur_preds = pred_df[mask].reset_index(drop=True)

    label = f"error_model_{i}"
    cur_preds[label] = cur_preds["true"] - cur_preds["pred_stage3_total"]

    cur_preds_sub = cur_preds[["DAWBA_ID", "repeat", "split_id", label]].set_index(
        ["DAWBA_ID", "repeat", "split_id"]
    )
    return cur_preds_sub


def plot_cluster(compare_df, pred_df, prob_threshold, param_list):
    inconclusive_mask = (
        compare_df[["P(best better)", "P(equal)", "P(this better)"]] < prob_threshold
    ).all(axis=1)
    conclusive_mask = compare_df["P(equal)"] > prob_threshold
    eq_param_df = compare_df[inconclusive_mask | conclusive_mask]

    if len(eq_param_df) == 1:
        print("NO models do as well, returning...")
        return None, eq_param_df
    model_frames = []
    for i, (idx, row) in enumerate(eq_param_df.iterrows()):
        cur_preds_sub = get_cur_pred_error(pred_df, idx, row, param_list)
        model_frames.append(cur_preds_sub)
        if i != 0:
            plot_scatter(model_frames[0], cur_preds_sub, idx)

    residual_wide = pd.concat(model_frames, axis=1, join="outer")  # instances x models
    residual_corr = (
        residual_wide.corr()
    )  # pairwise-complete by default, robust to misaligned folds
    sns.clustermap(
        residual_corr, cmap="viridis", annot=True, fmt=".2f", vmin=0.97, vmax=1
    )
    return residual_corr, eq_param_df


def compare_st_sq(r2_df, pred_df, best_T_params, score_type="r2"):
    r2_3_layer = get_layer3(r2_df, filter_best_Q=True)
    assert np.isclose(r2_3_layer["r2_Q_mean"], r2_3_layer["r2_Q_mean"].iloc[0]).all()

    # SQ scores
    best_Q_alpha = r2_3_layer.iloc[0]["alpha"].split("_")[0]
    mask = (pred_df["alpha"].str.split("_").str[0] == best_Q_alpha) & (
        pred_df["num_layers"] == 3
    )
    best_Q_pred_df = pred_df[mask].drop_duplicates(
        subset=["DAWBA_ID", "repeat", "split_id"]
    )
    best_q_score_repeats_folds = (
        best_Q_pred_df.groupby(["repeat", "split_id"])
        .apply(lambda g: r2_score(y_pred=g["pred_stage2_total"], y_true=g["true"]))
        .rename("r2")
        .reset_index()
    )
    ## ST scores
    (
        score_df_fold_repeats,
        _,
        _,
        _,
    ) = get_score(pred_df, score_type, param_vals=best_T_params)

    ## Combine + compare
    suffix_1 = "_ST"
    suffix_2 = "_SQ"
    combined_df = pd.merge(
        score_df_fold_repeats,
        best_q_score_repeats_folds,
        on=["repeat", "split_id"],
        suffixes=(suffix_1, suffix_2),
    )
    (p_left, p_rope, p_right), ax = two_on_single(
        x=combined_df[f"{score_type}{suffix_1}"],
        y=combined_df[f"{score_type}{suffix_2}"],
        rope=0.01,
        runs=10,
        plot=True,
    )
    return p_left, p_rope, p_right, combined_df


def compare_S_Q_T(r2_df, pred_df, score_type, best_T_params):
    # S scores
    S_pred_df = pred_df[pred_df["num_layers"] == 1].drop_duplicates(
        subset=["DAWBA_ID", "pred_stage1", "repeat", "split_id"]
    )
    score_df_fold_repeats_s = (
        S_pred_df.groupby(["repeat", "split_id"])
        .apply(lambda g: r2_score(y_pred=g["pred_stage1"], y_true=g["true"]))
        .rename("r2")
        .reset_index()
    )
    score_df_repeats_s = (
        S_pred_df.groupby(["repeat"])
        .apply(lambda g: r2_score(y_pred=g["pred_stage1"], y_true=g["true"]))
        .rename("r2")
        .reset_index()
    )
    # Q scores
    r2_1_layer = get_layer1(r2_df)
    best_Q_alpha = (
        r2_1_layer.sort_values(by="r2_Q_mean", ascending=False)
        .iloc[0]["alpha"]
        .split("_")[0]
    )
    q_mask = (pred_df["alpha"].str.split("_").str[0] == best_Q_alpha) & (
        pred_df["num_layers"] == 1
    )
    best_Q_pred_df = pred_df[q_mask].drop_duplicates(
        subset=["DAWBA_ID", "repeat", "split_id"]
    )

    # No need to pass in params bs already subset
    (
        _,
        _,
        score_df_fold_repeats_q,
        score_df_repeats_q,
    ) = get_score(best_Q_pred_df, score_type)

    # T scores
    (
        score_df_fold_repeats_t,
        score_df_repeats_t,
        _,
        _,
    ) = get_score(pred_df, score_type, param_vals=best_T_params)

    compare_dict = {
        "S": {
            "repeats/folds": score_df_fold_repeats_s,
            "repeats": score_df_repeats_s,
        },
        "Q": {
            "repeats/folds": score_df_fold_repeats_q,
            "repeats": score_df_repeats_q,
        },
        "T": {
            "repeats/folds": score_df_fold_repeats_t,
            "repeats": score_df_repeats_t,
        },
    }
    ## Make comparison table
    models = list(compare_dict)  # ['S', 'Q', 'T']
    p_better = pd.DataFrame(index=models, columns=models, dtype=float)  # P(row > col)
    p_equal = pd.DataFrame(index=models, columns=models, dtype=float)  # P(row = col)
    model_scores = pd.DataFrame(
        index=models,
        columns=["avg across repeats/folds (std)", "avg within repeats (std)"],
    )  # P(row = col)
    for a in models:
        rf = compare_dict[a]["repeats/folds"][score_type]
        rp = compare_dict[a]["repeats"][score_type]
        model_scores.loc[a, "avg across repeats/folds (std)"] = (
            f"{rf.mean():.3f} ({rf.std():.3f})"
        )
        model_scores.loc[a, "avg within repeats (std)"] = (
            f"{rp.mean():.3f} ({rp.std():.3f})"
        )
        for b in models:
            if a == b:
                p_better.loc[a, b] = np.nan  # diagonal: self-comparison undefined
                p_equal.loc[a, b] = 1.0
                continue

            merged = pd.merge(
                compare_dict[a]["repeats/folds"],
                compare_dict[b]["repeats/folds"],
                on=["repeat", "split_id"],
                suffixes=(f"_{a}", f"_{b}"),
            )
            p_left, p_rope, p_right = two_on_single(
                x=merged[f"{score_type}_{a}"],
                y=merged[f"{score_type}_{b}"],
                rope=0.01,
                runs=10,
            )
            p_better.loc[a, b] = p_left * 100  # P(a > b)
            p_equal.loc[a, b] = p_rope * 100  # P(a = b)

    for a in models:
        for b in models:
            if a != b:
                total = p_better.loc[a, b] + p_equal.loc[a, b] + p_better.loc[b, a]
                assert abs(total - 100) < 1e-6, f"{a},{b}: {total}"
    return p_better, p_equal, model_scores
