## residual_func.py
from src.utils import (
    col_config_sensitivity,
    IMPACTME_OUTCOMES_SHOT,
    IMPACTME_OUTCOMES_LR,
    IMPACTME_OUTCOMES_FULL,
    OUTCOMES_EXPAND,
)
from src.perm import create_perms

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge

## TODO: rename stage1, stage2, etc to stageSum, stageQuest, etc for readability


class BaseRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        num_layers,
        impact_type,
        desc_seg_type,
        quest_cols,
        agg_type,
        alpha2,
        alpha3,
        seed,
        linear_type,
        max_iter=5000,
    ):
        self.num_layers = num_layers
        self.impact_type = impact_type
        self.desc_seg_type = desc_seg_type
        self.quest_cols = quest_cols
        self.agg_type = agg_type
        self.alpha2 = alpha2
        self.max_iter = max_iter
        self.seed = seed
        self.linear_type = linear_type
        self.alpha3 = alpha3
        self.perm_ids = None

    def _build_X_blocks(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        quest_np = X[self.quest_cols].to_numpy(dtype=float)

        quest_sum = quest_np.sum(axis=1, keepdims=True)

        prob_col, prob_outcomes = self._get_prob_col_outcomes()
        prob_vals = X[prob_outcomes].values
        prob_vec = np.vstack(prob_vals).astype(float)

        return quest_sum, quest_np, prob_vec

    def _get_prob_col_outcomes(self):
        if self.agg_type in ["avg", "max", "block"]:
            return f"prob_vector_{self.agg_type}", IMPACTME_OUTCOMES_LR
        elif self.agg_type in ["proba-limited", "binary"]:
            prob_col = "open_text_block_0_shot"
            if self.desc_seg_type == "new":
                prob_outcomes = IMPACTME_OUTCOMES_FULL
            elif self.desc_seg_type == "expanded":
                prob_outcomes = OUTCOMES_EXPAND
            elif self.desc_seg_type in ["original", "augment_neg", "augment"]:
                prob_outcomes = IMPACTME_OUTCOMES_SHOT
            elif self.desc_seg_type == "combined":  # combines expanded + original
                prob_outcomes = [
                    f"{col}_impactme" for col in IMPACTME_OUTCOMES_SHOT
                ] + [f"{col}_expand" for col in OUTCOMES_EXPAND]
            else:
                raise ValueError(f"Unrecognized desc_seg_type: {self.desc_seg_type}")
        else:
            raise ValueError(f"Unrecognized agg_type: {self.agg_type}")
        return prob_col, prob_outcomes

    def get_coef_df(self):
        parts = [self._extract_coefs(self.stage1_, self._feat1, "stage1")]
        if self.num_layers == 1:
            return pd.concat(parts, ignore_index=True)
        ## 2/3 layers
        parts.append(self._extract_coefs(self.stage3_, self._feat3, "stage3"))
        if self.num_layers == 3:
            # only add stage2 info if 3 layers
            # stage2 info == stage1 info when 2 layers
            parts.append(self._extract_coefs(self.stage2_, self._feat2, "stage2"))
        return pd.concat(parts, ignore_index=True)

    def _make_model(
        self,
        alpha,
        model_type,
    ):
        """
        model_type: str
            One of ['elastic', 'regular', 'ridge']
        """
        if model_type == "ridge":
            return Ridge(
                alpha=alpha,
                fit_intercept=True,
                random_state=self.seed,
                max_iter=self.max_iter,
            )
        elif model_type == "regular":
            return LinearRegression()
        else:
            raise ValueError(
                f"Unrecognized model_type. Expected one of ['elastic', 'regular', 'ridge']. Got {model_type}"
            )

    def _extract_coefs(self, model, feature_names, stage_label):
        """
        Extract intercept + coefficients from a fitted sklearn linear model
        into a tidy DataFrame with columns [stage, feature, coef].
        Works for LinearRegression, Ridge, and ElasticNet.
        """
        coef = np.atleast_1d(model.coef_).flatten()
        intercept = np.atleast_1d(model.intercept_).flatten()
        rows = [{"stage": stage_label, "feature": "intercept", "coef": intercept[0]}]
        for name, val in zip(feature_names, coef):
            rows.append({"stage": stage_label, "feature": name, "coef": val})
        return pd.DataFrame(rows)

    def shuffle_vals(self, X, col_to_shuffle_np, perm_dir):
        ## Get permutation seed
        perm_seed_path = perm_dir / f"seed.txt"
        if not perm_seed_path.exists():
            raise ValueError(
                f"No seed file exists at path {perm_seed_path}, but prompted to permute. Aborting..."
            )
        with open(perm_seed_path, "r") as f:
            perm_seed = int(f.read().strip())
        # Sort by DAWBA_ID to ensure canonical order regardless of input row order
        sort_idx = np.argsort(X["DAWBA_ID"].values)
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        col_sorted = col_to_shuffle_np[sort_idx]
        ### Shuffle DAWBA_ID col
        self.perm_ids = create_perms(perm_seed=perm_seed, df_to_permute=X_sorted)
        # Attatch og DAWBA IDs to col_to_shuffle
        df_w_ids = pd.DataFrame(col_sorted, index=X_sorted["DAWBA_ID"].values)
        # Reindex by permuted IDs
        shuffled = df_w_ids.reindex(self.perm_ids)
        return shuffled.to_numpy()

    def predict(self, X):
        """
        sklearn-compatible predict: returns final cumulative prediction only.
        """
        pred_df = self.predict_all(X)
        return pred_df["pred_stage3_total"].to_numpy()

    def expand_text(self, df):
        prob_col, prob_outcomes = self._get_prob_col_outcomes()
        new_cols = pd.DataFrame(
            df[prob_col].tolist(), columns=prob_outcomes, index=df.index
        )
        df = df.join(new_cols)
        df = df.drop(columns=prob_col)
        return df


class IndependentRegressor(BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, shuffle=None, perm_dir=None):
        """
        Create linear/ridge regression for each layer individually
        Layer1--> Use sum scores to predict raw impact
        Layer2--> Use granular questionnaire response to predict raw impact
        Layer3--> Use text probs to predict raw impact
        *NOTE: use alpha3 for both questionnaire and text models
        """
        y = np.asarray(y, dtype=float)
        X1, X2, X3 = self._build_X_blocks(X)

        if shuffle == "text":
            X3 = self.shuffle_vals(X=X, col_to_shuffle_np=X3, perm_dir=perm_dir)
        elif shuffle is None:
            self.perm_ids = X["DAWBA_ID"].values
        else:
            raise ValueError(
                f"Unrecognized `shuffle` in fit method: {shuffle}. Must be one of ['text', None]"
            )

        # Build feature name lists
        self._feat1 = ["sum_score"]
        self._feat2 = list(self.quest_cols)
        _, self._feat3 = self._get_prob_col_outcomes()

        ## Layer 1 — Sum scores
        self.stage1_ = self._make_model(
            alpha=None,
            model_type="regular",
        )
        self.stage1_.fit(X1, y)

        ## Layer 2 - Granular Questionnaire
        self.stage2_ = self._make_model(
            alpha=self.alpha3,
            model_type=self.linear_type,
        )
        self.stage2_.fit(X2, y)

        ## Layer 3 - Text probs
        self.stage3_ = self._make_model(
            alpha=self.alpha3,
            model_type=self.linear_type,
        )
        self.stage3_.fit(X3, y)
        return self

    def predict_all(self, X):
        """
        Returns all stage predictions and cumulative totals in a DataFrame.
        """
        X1, X2, X3 = self._build_X_blocks(X)

        pred1 = self.stage1_.predict(X1)
        pred2 = self.stage2_.predict(X2)
        pred3 = self.stage3_.predict(X3)

        return pd.DataFrame(
            {
                "pred_stage1": pred1,
                "pred_stage2_total": pred2,
                "pred_stage3_total": pred3,
            }
        )


class ResidualRegressor(BaseRegressor):
    def __init__(self, two_layer_base, **kwargs):
        super().__init__(**kwargs)
        if self.num_layers not in [2, 3]:
            raise ValueError(
                f"Num layers must be one of [2,3] for ResidualRegressor. Got {self.num_layers}"
            )
        self.two_layer_base = two_layer_base
        if (self.two_layer_base is not None and self.num_layers != 2) or (
            self.num_layers == 2 and self.two_layer_base is None
        ):
            raise ValueError(
                f"two_layer_base must be not None exactly when num_layers=2. Got: two_layer_base:{self.two_layer_base}; num_layers:{self.num_layers}"
            )

    def fit(self, X, y, shuffle=None, perm_dir=None):
        """
        Use residual method
        Layer1--> Use sum scores to predict raw impact
        """
        y = np.asarray(y, dtype=float)
        X1, X2, X3 = self._build_X_blocks(X)

        if shuffle == "text":
            # only shuffle text in TRAIN
            X3 = self.shuffle_vals(X=X, col_to_shuffle_np=X3, perm_dir=perm_dir)
        elif shuffle is None:
            self.perm_ids = X["DAWBA_ID"].values
        else:
            raise ValueError(
                f"Unrecognized `shuffle` in fit method: {shuffle}. Must be one of ['text', None]"
            )

        # Build feature name lists
        self._feat1 = ["sum_score"]
        self._feat2 = list(self.quest_cols)
        _, self._feat3 = self._get_prob_col_outcomes()

        ## Layer 1 — Sum scores
        self.stage1_ = self._make_model(
            alpha=None,
            model_type="regular",
        )
        self.stage1_.fit(X1, y)
        self.pred1_train_ = self.stage1_.predict(X1)
        resid1 = y - self.pred1_train_

        if self.num_layers == 3:
            """
            Layer2--> Use granular questionnaire responses to predict layer 1 residuals
            Layer3--> Use probs derived from open text responses to predict layer 2 residuals
            """
            self.stage2_ = self._make_model(
                alpha=self.alpha2,
                model_type=self.linear_type,
            )
            self.stage2_.fit(X2, resid1)
            self.pred2_train_ = self.stage2_.predict(X2)
            resid2 = resid1 - self.pred2_train_

        else:  # num_layers = 2
            """
            if two_layer_base = S:
                - use sum layer (layer1) residuals (resid1 == resid2) for text layer (layer3) model
                - questionnaire layer (layer2) will be copy of sum layer (layer1)
            if two_layer_base = Q:
                - questionnaire predicts made on raw impact
                - use questionnaire layer (layer2) residuals (resid2) for text layer (layer3) model
                - questionnaire layer (layer2) will be independent of sum layer (layer1)
            """
            ## Layer2
            if self.two_layer_base == "S":
                self.stage2_ = self.stage1_
                self.pred2_train_ = self.pred1_train_
                resid2 = resid1
            elif self.two_layer_base == "Q":
                self.stage2_ = self._make_model(
                    alpha=self.alpha3,
                    model_type=self.linear_type,
                )
                self.stage2_.fit(X2, y)
                self.pred2_train_ = self.stage2_.predict(X2)
                resid2 = y - self.pred2_train_
            else:
                raise ValueError(
                    f"two_layer_base must be one of ['S', 'Q']. Got: {self.two_layer_base}"
                )

        ## Train with elastic net
        self.stage3_ = self._make_model(
            alpha=self.alpha3,
            model_type=self.linear_type,
        )
        try:
            self.stage3_.fit(X3, resid2)
        except ValueError as e:
            print(X3)
            raise e
        self.pred3_train_ = self.stage3_.predict(X3)
        return self

    def predict_all(self, X):
        """
        Returns all stage predictions and cumulative totals in a DataFrame.
        """
        X1, X2, X3 = self._build_X_blocks(X)

        num_preds = len(X1)
        if self.num_layers == 3:
            pred1 = self.stage1_.predict(X1)
            pred2 = self.stage2_.predict(X2)
        else:  # 2 layers
            if self.two_layer_base == "S":
                pred1 = self.stage1_.predict(X1)
                pred2 = np.zeros(num_preds)
            else:  # Q
                pred1 = np.zeros(num_preds)
                pred2 = self.stage2_.predict(X2)

        # Use layer 3 to predict layer 2 residuals
        pred2_total = pred1 + pred2
        pred3 = self.stage3_.predict(X3)
        pred3_total = pred2_total + pred3

        return pd.DataFrame(
            {
                "pred_stage1": pred1,
                "pred_stage2_increment": pred2,
                "pred_stage2_total": pred2_total,
                "pred_stage3_increment": pred3,
                "pred_stage3_total": pred3_total,
            }
        )


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
    )
    if num_layers == 1:
        model = IndependentRegressor(**model_args)
    else:
        model = ResidualRegressor(**model_args, two_layer_base=two_layer_base)

    # ================> Set up data
    # Expand text prob vectors to make probs granular cols
    df_expand = model.expand_text(df)

    # Get X_train, y_train, X_test, y_test
    test_set = df_expand[df_expand["split_id"] == split_id]
    train_set = df_expand[df_expand["split_id"] != split_id]
    assert len(df_expand) != 0
    assert len(test_set) + len(train_set) == len(df_expand)
    # Clean Train + Test X
    train_reset = train_set.reset_index(drop=True)
    test_reset = test_set.reset_index(drop=True)
    # y
    y_train = train_reset[f"impact_score_{impact_type}"]
    y_test = test_reset[f"impact_score_{impact_type}"]
    # ================> Fit + extract info from model
    model.fit(train_reset, y_train, shuffle=hypothesis_shuffle, perm_dir=perm_dir)

    pred_df = model.predict_all(test_reset)

    # Coefficients
    coef_df = model.get_coef_df()
    coef_df["split_id"] = split_id  # tag with fold so rows stay traceable

    # Permutation IDs
    perm_id_df = pd.DataFrame(
        {f"perm_id_split_{split_id}": model.perm_ids},
        index=list(range(len(model.perm_ids))),
    )

    # Predictions
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
    return perm_id_df, output_df, coef_df
