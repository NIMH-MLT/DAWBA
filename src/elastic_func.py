## elastic_func.py
from .config import SEED
from src.utils import col_config_sensitivity
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ResidualElasticNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        quest_cols,
        agg_type,
        alpha2,
        l1_ratio2,
        alpha3,
        l1_ratio3,
        seed,
        max_iter=5000,
    ):
        self.quest_cols = quest_cols
        self.agg_type = agg_type
        self.alpha2 = alpha2
        self.l1_ratio2 = l1_ratio2
        self.alpha3 = alpha3
        self.l1_ratio3 = l1_ratio3
        self.max_iter = max_iter
        self.seed = seed

    def _get_prob_col(self):
        if self.agg_type in ["avg", "max", "block"]:
            return f"prob_vector_{self.agg_type}"
        elif self.agg_type == "binary":
            return "open_text_block_0_shot_binary"
        elif self.agg_type == "proba-limited":
            # col_to_add = "block_0_shot_proba_limited"
            return "open_text_block_0_shot"
        else:
            raise ValueError(f"Unrecognized agg_type: {self.agg_type}")

    def _build_X_blocks(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        quest_np = X[self.quest_cols].to_numpy(dtype=float)
        quest_sum = quest_np.sum(axis=1, keepdims=True)

        prob_col = self._get_prob_col()
        prob_vec = np.vstack(X[prob_col].values).astype(float)
        return quest_sum, quest_np, prob_vec

    def _make_model(self, alpha, l1_ratio, model_type):
        if model_type == "elastic":
            return ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=self.max_iter,
                random_state=self.seed,
            )

        elif model_type == "regular":
            return LinearRegression()
        else:
            raise ValueError(
                f"Unrecognized model_type. Expected one of ['elastic', 'regular']. Got {model_type}"
            )

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)

        X1, X2, X3 = self._build_X_blocks(X)

        self.stage1_ = self._make_model(alpha=None, l1_ratio=None, model_type="regular")
        self.stage1_.fit(X1, y)
        self.pred1_train_ = self.stage1_.predict(X1)
        resid1 = y - self.pred1_train_

        self.stage2_ = self._make_model(
            alpha=self.alpha2, l1_ratio=self.l1_ratio2, model_type="elastic"
        )
        self.stage2_.fit(X2, resid1)
        self.pred2_train_ = self.stage2_.predict(X2)
        resid2 = resid1 - self.pred2_train_

        self.stage3_ = self._make_model(
            alpha=self.alpha3, l1_ratio=self.l1_ratio3, model_type="elastic"
        )
        self.stage3_.fit(X3, resid2)
        self.pred3_train_ = self.stage3_.predict(X3)

        return self

    def predict(self, X):
        """
        sklearn-compatible predict: returns final cumulative prediction only.
        """
        pred_df = self.predict_all(X)
        return pred_df["pred_stage3_total"].to_numpy()

    def predict_all(self, X):
        """
        Returns all stage predictions and cumulative totals in a DataFrame.
        """
        X1, X2, X3 = self._build_X_blocks(X)

        pred1 = self.stage1_.predict(X1)

        pred2 = self.stage2_.predict(X2)
        pred2_total = pred1 + pred2

        pred3 = self.stage3_.predict(X3)
        pred3_total = pred2_total + pred3

        out = pd.DataFrame(
            {
                "pred_stage1": pred1,
                "pred_stage2_increment": pred2,
                "pred_stage2_total": pred2_total,
                "pred_stage3_increment": pred3,
                "pred_stage3_total": pred3_total,
            }
        )
        return out


def get_lr_output(
    df,
    agg_type,
    selected_alphas,
    selected_l1s,
    split_id,
    impact_type,
    export_path,
    max_iter,
    seed,
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

    ## ============> Modeling
    model = ResidualElasticNetRegressor(
        quest_cols=quest_cols,
        agg_type=agg_type,
        alpha2=selected_alphas[0],
        l1_ratio2=selected_l1s[0],
        alpha3=selected_alphas[1],
        l1_ratio3=selected_l1s[1],
        max_iter=max_iter,
        seed=seed,
    )

    model.fit(train_reset, y_train)

    pred_df = model.predict_all(test_set)

    ## ===========> Create df
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
    ## Export
    if export_path.exists():
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    output_df.to_parquet(export_path)
