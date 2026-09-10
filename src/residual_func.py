## residual_func.py
from src.utils import col_config_sensitivity
from src.Regressor import IndependentRegressor, ResidualRegressor
from src.utils import get_df_path, expand_text
from src.coef_visual import plot_heatmap

import pandas as pd


def prep_train_test(base_df, split_id, impact_type, agg_type, desc_seg_type):
    """
    Helper for get_lr_output(); Splits data into train/test based on split_id
    and extracts X and y for each
    """
    if desc_seg_type in ["shot_zero", "shot_few"]:
        df_expand = base_df
    else:
        # Expand text prob vectors to make probs granular cols
        df_expand, _ = expand_text(
            df=base_df,
            agg_type=agg_type,
            desc_seg_type=desc_seg_type,
        )

    # Get X_train, y_train, X_test, y_test
    test_set = df_expand[df_expand["split_id"] == split_id].reset_index(drop=True)
    train_set = df_expand[df_expand["split_id"] != split_id].reset_index(drop=True)
    assert len(df_expand) != 0
    assert len(test_set) + len(train_set) == len(df_expand)
    assert len(test_set) > 0

    # Clean Train + Test y
    target_col = f"impact_score_{impact_type}"
    y_train = train_set[target_col]
    y_test = test_set[target_col]
    # X
    X_train = train_set.drop(target_col, axis=1)
    X_test = test_set.drop(target_col, axis=1)

    return X_train, X_test, y_train, y_test


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
    # scale=False,
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
    # ================> Set up data
    X_train, X_test, y_train, y_test = prep_train_test(
        base_df=df,
        split_id=split_id,
        impact_type=impact_type,
        agg_type=agg_type,
        desc_seg_type=desc_seg_type,
    )
    outcome_sub_list = None
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
        outcome_sub_list=outcome_sub_list,  # dont subset outcomes for full run
    )
    if num_layers == 1:
        model = IndependentRegressor(**model_args)
    else:
        model = ResidualRegressor(
            **model_args,
            two_layer_base=two_layer_base,
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
    pred_df["true"] = y_test.values
    pred_df["split_id"] = split_id

    return perm_id_df, pred_df, coef_df


############################################################
############################################################
############################################################
def get_X_y(df, config_dict, scale=False):
    """
    Extracts X and y from a given df based on config_dict. Optionally scales the text_derived numerical columns to [0, 1] range.

    Params
    ------
    data_dir : pandas DF
        - contains train/test data for a given LLM model, temp, and config_dict
    llm_model : str
        Name of LLM model used to generate text embeddings. One of [gemini-2.5-pro, Llama-3.3-70B-Instruct, Qwen3.6-27B]
    config_dict : dict
        Dictionary containing model configuration. Must contain keys:
            - agg_type : str
            - desc_seg_type : str
            - impact_type : str
    scale : bool
        Whether to scale the text_derived numerical (prob_cols) columns to [0, 1] range using MinMaxScaler
    """
    df_expand, prob_cols = expand_text(
        df=df,
        agg_type=config_dict["agg_type"],
        desc_seg_type=config_dict["desc_seg_type"],
    )
    target_col = f"impact_score_{config_dict['impact_type']}"
    X = df_expand.drop(target_col, axis=1)
    y = df_expand[target_col]
    if scale:
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(clip=True, feature_range=(0, 1))
        X[prob_cols] = scaler.fit_transform(X[prob_cols])
    return X, y


def build_model(
    config_dict, data_dir, temp, llm_model, two_layer_base=None, scale=False
):
    """
    - Fit a ResidualRegressor given some model configuration on **training data**
    - Returns trained model, learned feature coefs, and loaded train DF
    """
    ## Get train data
    data_path = get_df_path(
        df_dir=data_dir,
        llm_model=llm_model,
        agg_type=config_dict["agg_type"],
        desc_seg_type=config_dict["desc_seg_type"],
        impact_type=config_dict["impact_type"],
        temp=temp,
    )
    train_df = pd.read_parquet(data_path)
    # Split X/y
    X_train, y_train = get_X_y(df=train_df, config_dict=config_dict, scale=scale)

    # Build + fit model
    model = ResidualRegressor(**config_dict, two_layer_base=two_layer_base)
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
    df_sub, _ = expand_text(
        df=df,
        agg_type=config_dict["agg_type"],
        desc_seg_type=config_dict["desc_seg_type"],
    )
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
        title=f"Summary of saliency- {impact_type.upper()}",
        vmin=-0.5,
        vmax=0.5,
        xlabel="Summary type",
    )
    return summary
