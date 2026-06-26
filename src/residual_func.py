## residual_func.py
from src.utils import col_config_sensitivity
from src.Regressor import IndependentRegressor, ResidualRegressor
from src.utils import get_df_path
from src.coef_visual import plot_heatmap
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def get_output_df(num_layers, test_set, y_test, pred_df):
    """
    Aggregates ground truth and model predictions
    """
    if num_layers == 1:
        output_df = pd.DataFrame(
            {
                "DAWBA_ID": test_set["DAWBA_ID"].values,
                "true": y_test.values,
                "pred_stage1": pred_df["pred_stage1"].values,
                "pred_stage2_total": pred_df["pred_stage2_total"].values,
                "pred_stage3_total": pred_df["pred_stage3_total"].values,
            }
        )
    else:
        output_df = pd.DataFrame(
            {
                "DAWBA_ID": test_set["DAWBA_ID"].values,
                "true": y_test.values,
                "pred_stage1": pred_df["pred_stage1"].values,
                "pred_stage2_increment": pred_df["pred_stage2_increment"].values,
                "pred_stage2_total": pred_df["pred_stage2_total"].values,
                "pred_stage3_increment": pred_df["pred_stage3_increment"].values,
                "pred_stage3_total": pred_df["pred_stage3_total"].values,
            }
        )
    return output_df


def prep_train_test(model, base_df, split_id, impact_type):
    """
    Helper for get_lr_output(); Splits data into train/test based on split_id
    and extracts X and y for each
    """
    # Expand text prob vectors to make probs granular cols
    df_expand = model.expand_text(base_df)

    # Get X_train, y_train, X_test, y_test
    test_set = df_expand[df_expand["split_id"] == split_id]
    train_set = df_expand[df_expand["split_id"] != split_id]
    assert len(df_expand) != 0
    assert len(test_set) + len(train_set) == len(df_expand)

    # Clean Train + Test y
    target_col = f"impact_score_{impact_type}"
    y_train = train_reset[target_col]
    y_test = test_reset[target_col]
    # X
    train_reset = train_set.drop(target_col, axis=1).reset_index(drop=True)
    test_reset = test_set.drop(target_col, axis=1).reset_index(drop=True)

    return train_reset, test_reset, y_train, y_test


def get_lr_output(
    df,
    agg_type,
    desc_seg_type,
    num_layers,
    selected_alphas,
    split_id,
    impact_type,
    max_iter,
    seed,
    perm_dir,
    two_layer_base,  # first estimator type for 2 layer iteration; one of 'Q', 'T'
    linear_type="ridge",
    hypothesis_shuffle=None,  # None, 'text'
):
    """
    - For a given split during CV, subset into train/test sets
    - Fit a Regressor on train
        - Depending on num_layers, one of IndependentRegressor or ResidualRegressor
        - Optionally perform stratified permutation on layer3 (text)
    - Get learned regression coefs from training
        - Written to memory as coef_df.parquet
    - Get resulting order of participants from permutation
        - Written to memory as id_df.parquet
    - Get predictions on test set from each layer of Regressor
        - Written to memory as pred_df.parquet
    """
    # ================> Set up model
    model_args = dict(
        num_layers=num_layers,
        impact_type=impact_type,
        quest_cols=col_config_sensitivity[f"{impact_type.lower()}_quest"],
        agg_type=agg_type,
        alpha2=selected_alphas[0],
        alpha3=selected_alphas[1],
        max_iter=max_iter,
        seed=seed,
        linear_type=linear_type,
        desc_seg_type=desc_seg_type,
        outcome_sub_list=None,  # dont subset outcomes for full run
    )
    if num_layers == 1:
        model = IndependentRegressor(**model_args)
    else:
        model = ResidualRegressor(**model_args, two_layer_base=two_layer_base)

    # ================> Set up data
    X_train, X_test, y_train, y_test = prep_train_test(
        model=model, base_df=df, split_id=split_id, impact_type=impact_type
    )
    # ================> Fit + extract info from model
    model.fit(X_train, y_train, shuffle=hypothesis_shuffle, perm_dir=perm_dir)
    pred_df = model.predict_all(X_test)

    # Coefficients
    coef_df = model.get_coef_df()
    coef_df["split_id"] = split_id  # tag with fold so rows stay traceable

    # Permutation IDs
    perm_id_df = pd.DataFrame(
        {f"perm_id_split_{split_id}": model.perm_ids},
        index=list(range(len(model.perm_ids))),
    )

    # Predictions
    output_df = get_output_df(num_layers, X_test, y_test, pred_df)
    return perm_id_df, output_df, coef_df


############################################################
############################################################
############################################################


def build_model(config_dict, llama_model, data_dir):
    """
    - Fit a ResidualRegressor given some model configuration
    - Returns trained model, learned feature coefs, and loaded train DF
    """
    data_path = get_df_path(
        df_dir=data_dir,
        llama_model=llama_model,
        agg_type=config_dict["agg_type"],
        desc_seg_type=config_dict["desc_seg_type"],
        impact_type=config_dict["impact_type"],
    )
    model = ResidualRegressor(**config_dict, two_layer_base=None)
    train_df = pd.read_parquet(data_path)
    print(len(train_df))
    df_expand = model.expand_text(train_df)
    target_col = f"impact_score_{config_dict['impact_type']}"
    X_train = df_expand.drop(target_col, axis=1)
    y_train = df_expand[target_col]
    model.fit(X_train, y_train, shuffle=None, perm_dir=None)
    # Process coef df
    coef_df = model.get_coef_df()
    coef_df["stage_feat"] = coef_df["stage"] + "_" + coef_df["feature"]
    coef_df = coef_df.drop(columns=["stage", "feature"])
    # remove this so can combine self+parent
    replace_str = "p_dep" if config_dict["impact_type"] == "parent" else "s_dep"
    coef_df["stage_feat"] = coef_df["stage_feat"].str.replace(
        replace_str, "", regex=False
    )
    coef_df = coef_df.sort_values("stage_feat")
    coef_df = coef_df.set_index("stage_feat")
    return model, coef_df, train_df


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


def get_perm_feat_imp(
    train_df, test_df, model_config_dict, n_samples, random_state, n_jobs=1
):
    """
    Calculate predictive value of each outcome by permuting the order of text outcomes
    n_samples times. Each permutation loops through a shuffled order of feats and iteratively adds
    features + records added r^2 to previous iteration w/o the outcome. Outputs pandas DF containing
    avg incremental r^2 of each outcome over the n_samples permutations.
    """
    prelim_model = ResidualRegressor(**model_config_dict, two_layer_base=None)
    _, outcome_list = prelim_model._get_prob_col_outcomes()

    # Get X/y train/test sets
    target_col = f"impact_score_{model_config_dict['impact_type']}"

    df_expand_train = prelim_model.expand_text(train_df)
    df_expand_test = prelim_model.expand_text(test_df)

    X_train = df_expand_train.drop(target_col, axis=1)
    X_test = df_expand_test.drop(target_col, axis=1)

    y_train = df_expand_train[target_col].to_numpy()
    y_test = df_expand_test[target_col].to_numpy()

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

    all_conts = Parallel(n_jobs=n_jobs)(delayed(run_perm)(perm) for perm in perm_list)
    lmg = {
        outcome: np.mean([r[outcome] for r in all_conts]) for outcome in outcome_list
    }
    res_dict = {
        "outcome": list(lmg.keys()),
        "lmg": list(lmg.values()),
    }
    res_df = (
        pd.DataFrame(res_dict)
        .sort_values("lmg", ascending=False)
        .reset_index(drop=True)
    )
    return res_df


def get_saliency_map(config_dict, model, df, coef_df):
    """
    Given a df of weights for each feature (coef_df), multiply recorded values for every feature
    (including all of sum, questionnaire, and text feats) by the assigned coef weight. Resulting matrix
    represents the actual addative contribution of each feature to final model output (excluding intercepts).
    Calculates and returns a summary (mean, stdev, min, max) for each feature. Plots summary as a heatmap.

    Use
    ---------
    - Note that the returned summary matrix should be used in conjunction w/ learned coef
    - A feature with a high coef but right skewed distribution (mostly close to 0) will
        have a low avg effect on model output (common in text outcomes) and
        vice versa (common in sum score)
    """
    # Load necessary vars
    impact_type = config_dict["impact_type"]
    quest_pref = "p_dep" if impact_type == "parent" else "s_dep"
    # Expand text prob vector into cols
    df_sub = model.expand_text(df)
    # Add sum score col
    quest_cols = model.quest_cols
    df_sub["sum_score"] = (
        df_sub[quest_cols].to_numpy(dtype=int).sum(axis=1, keepdims=True)
    )
    ## Multiply learned coefs w/ observed values
    new_cols = []
    for idx, row in coef_df.iterrows():
        col_name = "_".join(idx.split("_")[1:])  # remove stage_ prefix
        # stage = idx_split[0]
        if col_name == "intercept":
            continue
        if col_name.startswith("_"):  # add prefix to quest cols
            col_name = quest_pref + col_name
        # Multiply coef scalar by observed vals for current feature
        coef = getattr(row, f"coef_{impact_type}")
        df_col = df_sub[col_name]
        df_col_weighted = df_col * coef
        new_cols.append(pd.DataFrame({col_name: df_col_weighted}))
    # Create combined saliency map
    combined = pd.concat(new_cols, axis=1)
    # Summarize across all participants
    summary = combined.describe().loc[["mean", "std", "min", "max"]]
    # plot
    plot_heatmap(
        summary.T,
        title="Summary of saliency",
        vmin=-0.5,
        vmax=0.5,
        xlabel="Summary type",
    )
    return summary
