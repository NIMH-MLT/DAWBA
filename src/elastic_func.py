## elastic_func.py
from .config import SEED
from src.utils import col_config_sensitivity
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np


def get_lr_output(
    df,
    agg_type,
    alpha_val,
    l1_val,
    split_id,
    impact_type,
    export_path,
    max_iter,
    seed=SEED,
):
    quest_cols = col_config_sensitivity[f"{impact_type.lower()}_quest"]
    # ================> Get X_train, y_train, X_test, y_test
    test_set = df[df["split_id"] == split_id]
    train_set = df[df["split_id"] != split_id]
    assert len(df) != 0
    assert len(test_set) + len(train_set) == len(df)
    ## ========> Set up Train + Test
    train_reset = train_set.reset_index(drop=True)
    test_reset = test_set.reset_index(drop=True)
    # y
    y_train = train_reset[f"impact_score_{impact_type}"]
    y_test = test_reset[f"impact_score_{impact_type}"]
    # X-questionaire answers
    X_train_quest = train_set[quest_cols].to_numpy(dtype=int)
    X_test_quest = test_reset[quest_cols].to_numpy(dtype=int)
    # Add outcome probs to X if prompted
    if agg_type != "reduced":
        if agg_type in ["avg", "max", "block"]:
            col_to_add = f"prob_vector_{agg_type}"
        else:  ## 0-shot
            col_to_add = "open_text_block_0_shot"
        # Train X
        X_train_proba_vector = np.vstack(train_set[col_to_add].values).astype(float)
        X_train = np.hstack([X_train_quest, X_train_proba_vector])
        assert X_train.shape[1] == len(quest_cols) + len(X_train_proba_vector[0])
        # Test X
        X_test_proba_vector = np.vstack(test_set[col_to_add].values).astype(float)
        X_test = np.hstack([X_test_quest, X_test_proba_vector])
        assert X_test.shape[1] == len(quest_cols) + len(X_test_proba_vector[0])
    # Otherwise X just questionaire responses
    else:
        X_train = X_train_quest
        X_test = X_test_quest
    ## ============> Modeling
    model = ElasticNet(
        alpha=alpha_val,
        l1_ratio=l1_val,
        fit_intercept=True,
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ## ===========> Create df

    dawba_id_series = test_reset["DAWBA_ID"]
    # print(f"DAWBA ID SIZE: {len(dawba_id_series)}")
    # print(f"TRUE: {len(y_test)}")
    # print(f"PRED: {len(y_pred)}")
    output_df = pd.DataFrame(
        {"DAWBA_ID": dawba_id_series, "true": y_test, "pred": y_pred}
    )
    ## Export
    if export_path.exists():
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    output_df.to_parquet(export_path)
