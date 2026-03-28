## elastic_func.py
from .config import SEED
from src.utils import col_config_sensitivity
from sklearn.linear_model import ElasticNet, LinearRegression
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


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
    X_train_quest = train_set[quest_cols]
    X_test_quest = test_reset[quest_cols]

    X_train_quest_np = X_train_quest.to_numpy(dtype=int)
    X_test_quest_np = X_test_quest.to_numpy(dtype=int)
    # Add outcome probs to X if prompted
    if agg_type != "reduced":
        if agg_type in ["avg", "max", "block"]:
            col_to_add = f"prob_vector_{agg_type}"
        else:  ## 0-shot
            col_to_add = "open_text_block_0_shot"
        # Train X
        X_train_proba_vector = np.vstack(train_set[col_to_add].values).astype(float)
        X_train = np.hstack([X_train_quest_np, X_train_proba_vector])
        assert X_train.shape[1] == len(quest_cols) + len(X_train_proba_vector[0])
        # Test X
        X_test_proba_vector = np.vstack(test_set[col_to_add].values).astype(float)
        X_test = np.hstack([X_test_quest_np, X_test_proba_vector])
        assert X_test.shape[1] == len(quest_cols) + len(X_test_proba_vector[0])
    # Otherwise X just questionaire responses
    else:
        X_train = X_train_quest_np
        X_test = X_test_quest_np

    ## ============> Modeling
    ## just elasticnet on questionnaire data ##
    elastic_model = ElasticNet(
        alpha=alpha_val,
        l1_ratio=l1_val,
        fit_intercept=True,
        max_iter=max_iter,
        random_state=seed,
    )

    elastic_model.fit(X_train, y_train)
    y_pred = elastic_model.predict(X_test)

    ## PCA --> elasticnet pipeline ##
    # linear_model = LinearRegression()
    # pipeline = Pipeline(
    #     [
    #         ("pca", PCA(random_state=SEED)),
    #         ("regressor", elastic_model),
    #     ]
    # )
    # pipeline.fit(X_train, y_train)
    # y_pred = pipeline.predict(X_test)

    ## Fit just on questionnaire sum scores ##
    # X_train_sum = X_train_quest.sum(axis=1).to_numpy(dtype=int).reshape(-1, 1)
    # X_test_sum = X_test_quest.sum(axis=1).to_numpy(dtype=int).reshape(-1, 1)
    # linear_model.fit(X_train_sum, y_train)
    # y_pred = linear_model.predict(X_test_sum)

    ## ===========> Create df
    dawba_id_series = test_reset["DAWBA_ID"]
    output_df = pd.DataFrame(
        {"DAWBA_ID": dawba_id_series, "true": y_test, "pred": y_pred}
    )
    ## Export
    if export_path.exists():
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    output_df.to_parquet(export_path)
