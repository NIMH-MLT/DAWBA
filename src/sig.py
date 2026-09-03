# sig.py- significance testing
from src.utils import (
    col_config_sensitivity,
    get_test_df_path,
    get_df_path,
)
from src.residual_func import get_X_y
from src.Regressor import ResidualRegressor, IndependentRegressor
import pandas as pd
from sklearn.metrics import r2_score


def format_p_val(p_value):
    if p_value < 0.001:
        return "<0.001"
    elif p_value < 0.05:
        return float(round(p_value, 3))
    else:
        return float(round(p_value, 2))


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


def get_p(df_sub):
    """
    **Hypothesis 1**

    Westfall-young
    """
    col_max = df_sub.iloc[0].idxmax()
    observed = df_sub.iloc[0].max()
    max_list = df_sub.max(axis=1)
    p_val = (max_list >= observed).sum() / (len(max_list) + 1)
    return p_val, col_max
