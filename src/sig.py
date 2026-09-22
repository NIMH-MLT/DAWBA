# sig.py- significance testing
from src.utils import (
    col_config_sensitivity,
    get_test_df_path,
    get_df_path,
    get_perm_seed,
    order_perm_dir,
)
from src.residual_func import get_X_y
from src.Regressor import ResidualRegressor, IndependentRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def format_p_val(p_value):
    """
    Format p-value into a string

    """
    if p_value < 0.001:
        return "<0.001"
    elif p_value < 0.06:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.2f}"


def prep_train_test_model(
    selected_params, impact_type, train_data_dir, test_data_dir, shared_dict
):
    """
    Import and split train/test sets. Instantiate related model for given params.
    """
    param_dict = {
        **selected_params,
        "impact_type": impact_type,
        "quest_cols": col_config_sensitivity[f"{impact_type}_quest"],
        **shared_dict,
    }
    model_config = {
        k: v for k, v in param_dict.items() if k not in ["llm_model", "temp"]
    }
    train_df_path = get_df_path(
        df_dir=train_data_dir,
        llm_model=param_dict["llm_model"],
        agg_type=param_dict["agg_type"],
        desc_seg_type=param_dict["desc_seg_type"],
        impact_type=impact_type,
        temp=param_dict["temp"],
    )
    train_df = pd.read_parquet(train_df_path)
    X_train, y_train = get_X_y(df=train_df, config_dict=param_dict, scale=False)
    test_df_path = get_test_df_path(
        df_dir=test_data_dir,
        llm_model=param_dict["llm_model"],
        agg_type=param_dict["agg_type"],
        desc_seg_type=param_dict["desc_seg_type"],
        impact_type=impact_type,
        temp=param_dict["temp"],
    )
    test_df = pd.read_parquet(test_df_path)
    X_test, y_test = get_X_y(df=test_df, config_dict=param_dict, scale=False)
    if param_dict["num_layers"] == 1:
        model = IndependentRegressor(**model_config)
    else:
        model = ResidualRegressor(**model_config)
    return X_train, y_train, X_test, y_test, model


def get_baseline_reduced(residual_setup, y_true, pred_df):
    """
        ** Hypothesis 1 **
    Get r2 of reduced residual set-up

    Params
    -----
    residual_setup: str
            One of ['SQT', 'ST', 'QT']
    pred_dir: pandas DF
            Contains predictions for each stage of residual regressor
            Output of ResidualRegressor.predict_all()
    """
    if residual_setup not in ["SQT", "ST", "QT"]:
        raise ValueError(f"Unknown residual_setup: {residual_setup}")
    reduced_pred_name = (
        "pred_stage2_total" if residual_setup in ["SQT", "QT"] else "pred_stage1"
    )
    y_pred = pred_df[reduced_pred_name]
    r2_baseline = r2_score(y_true=y_true, y_pred=y_pred)
    return r2_baseline


def plot_null_observed(null_dist, observed, set_up, impact_type):
    """
    Plots observed statistic in context of null distribution
    """
    more_extreme_vals = (null_dist >= observed).sum()
    plt.hist(null_dist, bins=100)
    plt.title(
        f"{impact_type.upper()}: More extreme for {set_up}: {more_extreme_vals}/{len(null_dist)}"
    )
    plt.axvline(x=observed, color="red", ls="--", lw=2)
    plt.show()
    plt.close()


def get_raw_p(null_dist, observed, add_1_flag):
    """
    Helper function that calculates p-value given an observed value and null distribution
    """
    numerator = (null_dist >= observed).sum()
    denominator = len(null_dist)
    if add_1_flag:
        numerator += 1
        denominator += 1
    p_val = numerator / denominator
    return p_val


def get_p(df_sub, impact_type, add_1_flag=False, plot_histo=False):
    """
    **Hypothesis 1**

    Westfall-young
    """
    col_max = df_sub.iloc[0].idxmax()
    observed = df_sub.iloc[0].max()
    max_list = df_sub.max(axis=1)
    p_val = get_raw_p(null_dist=max_list, observed=observed, add_1_flag=add_1_flag)
    if plot_histo:
        plot_null_observed(
            null_dist=max_list,
            observed=observed,
            set_up=col_max,
            impact_type=impact_type,
        )
    return p_val, col_max


def get_hyp1_null(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    all_perm_dir,
    reduced_r2_no_shuffle,
    start_perm,
    end_perm,
):
    ## RUN THROUGH PERMS
    diff_list = []
    # Order perm_dir, just to have stable ordering of null dist
    ordered_perm_dir = order_perm_dir(
        parent_perm_dir=all_perm_dir, start_perm=start_perm, end_perm=end_perm
    )
    for perm_num, perm_dir in ordered_perm_dir:
        # Get null diff
        model.fit(X_train, y_train, shuffle="text", perm_dir=perm_dir)
        full_preds_shuffled = model.predict(X_test)
        full_r2_shuffle = r2_score(y_true=y_test, y_pred=full_preds_shuffled)
        shuffled_r2_diff = full_r2_shuffle - reduced_r2_no_shuffle
        # Add to null distribution
        diff_list.append(shuffled_r2_diff)
    # TODO: Export perm iteration results to perm folder?
    return diff_list


def get_hyp2_null(
    y_pred_SQ,
    y_pred_ST,
    y_true,
    all_perm_dir,
    start_perm,
    end_perm,
):
    """
    y_pred_SQ, y_pred_ST, and y_true are assumed to be in the same row order
    """
    diff_list = []
    # Order perm_dir, just to have stable ordering of null dist
    ordered_perm_dir = order_perm_dir(
        parent_perm_dir=all_perm_dir, start_perm=start_perm, end_perm=end_perm
    )
    for perm_num, perm_dir in ordered_perm_dir:
        perm_seed = get_perm_seed(perm_dir=perm_dir)
        rng = np.random.RandomState(perm_seed)
        mask = rng.rand(len(y_true)) < 0.5
        perm_SQ = np.where(mask, y_pred_SQ, y_pred_ST)
        perm_ST = np.where(mask, y_pred_ST, y_pred_SQ)

        diff_list.append(
            r2_score(y_true=y_true, y_pred=perm_ST)
            - r2_score(y_true=y_true, y_pred=perm_SQ)
        )
    # TODO: Export perm iteration results to perm folder?
    return diff_list
