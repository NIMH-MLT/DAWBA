# Regressor.py
from src.perm import create_perms
from src.utils import get_prob_col_outcomes, get_perm_seed


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge


# TODO: Write test for export/import of model
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
        outcome_sub_list=None,  # to run with only a subset of text-level outcomes
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
        self.outcome_sub_list = outcome_sub_list

    def _build_X_blocks(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        # Sum
        quest_np = X[self.quest_cols].to_numpy(dtype=float)
        # Quest
        quest_sum = quest_np.sum(axis=1, keepdims=True)
        # Text
        prob_col, prob_outcomes = get_prob_col_outcomes(
            agg_type=self.agg_type,
            desc_seg_type=self.desc_seg_type,
        )
        # Subset psychological text feats if prompted
        if self.outcome_sub_list is not None:
            # ensure passed outcomes are part of whole outcome list
            # print(prob_outcomes)
            if not set(self.outcome_sub_list).issubset(prob_outcomes):
                raise ValueError(
                    f"""Some specified outcomes are not in the available outcome list: \n
                    In sublist but not available outcome list: {set(self.outcome_sub_list) - set(prob_outcomes)} \n
                    Prob column name: {prob_col}\n
                    Description/Segmentation type: {self.desc_seg_type}\n
                    Available outcomes: {prob_outcomes}\n
                    Specified outcomes: {self.outcome_sub_list}\n
                    """
                )
            prob_outcomes = self.outcome_sub_list
        prob_vals = X[prob_outcomes].values
        prob_vec = np.vstack(prob_vals).astype(float)
        # Return all
        return quest_sum, quest_np, prob_vec

    def _reported_stages(self):
        """[(label, fitted_model, feature_names)] for stages that contribute to
        the final prediction. Subclasses define which those are."""
        raise NotImplementedError

    def get_coef_df(self):
        return pd.concat(
            [
                self._extract_coefs(model, feats, label)
                for label, model, feats in self._reported_stages()
            ],
            ignore_index=True,
        )

    def _make_model(
        self,
        alpha,
        model_type,
    ):
        """
        model_type: str
            One of [''regular', 'ridge']
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
                f"Unrecognized model_type. Expected one of ['regular', 'ridge']. Got {model_type}"
            )

    def _extract_coefs(self, model, feature_names, stage_label):
        """
        Extract intercept + coefficients from a fitted sklearn linear model
        into a tidy DataFrame with columns [stage, feature, coef].
        Works for LinearRegression and Ridge.
        """
        coef = np.atleast_1d(model.coef_).flatten()
        intercept = np.atleast_1d(model.intercept_).flatten()
        rows = [{"stage": stage_label, "feature": "intercept", "coef": intercept[0]}]
        for name, val in zip(feature_names, coef):
            rows.append({"stage": stage_label, "feature": name, "coef": val})
        return pd.DataFrame(rows)

    def shuffle_vals(self, X, col_to_shuffle_np, perm_dir):
        """
        Used for hypothesis 1A/1b to shuffle TRAINING DATA
        """
        ## Get permutation seed
        perm_seed = get_perm_seed(perm_dir=perm_dir)
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


class IndependentRegressor(BaseRegressor):
    # Explicit kwargs for sklearn compatibility
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
        outcome_sub_list=None,
        max_iter=5000,
    ):
        super().__init__(
            num_layers=num_layers,
            impact_type=impact_type,
            desc_seg_type=desc_seg_type,
            quest_cols=quest_cols,
            agg_type=agg_type,
            alpha2=alpha2,
            alpha3=alpha3,
            seed=seed,
            linear_type=linear_type,
            outcome_sub_list=outcome_sub_list,
            max_iter=max_iter,
        )

    def _reported_stages(self):
        """Each stage is a standalone model on raw y, and fit() builds all three
        regardless of num_layers -- so all three are reported."""
        return [
            ("stage1", self.stage1_, self._feat1),
            ("stage2", self.stage2_, self._feat2),
            ("stage3", self.stage3_, self._feat3),
        ]

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
        _, self._feat3 = get_prob_col_outcomes(
            agg_type=self.agg_type,
            desc_seg_type=self.desc_seg_type,
        )
        if self.outcome_sub_list is not None:
            self._feat3 = list(self.outcome_sub_list)

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
                "DAWBA_ID": X["DAWBA_ID"].values,
                "pred_stage1": pred1,
                "pred_stage2_total": pred2,
                "pred_stage3_total": pred3,
            }
        )


class ResidualRegressor(BaseRegressor):
    # explicit kwargs for sklearn compatibility
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
        two_layer_base=None,
        outcome_sub_list=None,
        max_iter=5000,
    ):
        super().__init__(
            num_layers=num_layers,
            impact_type=impact_type,
            desc_seg_type=desc_seg_type,
            quest_cols=quest_cols,
            agg_type=agg_type,
            alpha2=alpha2,
            alpha3=alpha3,
            seed=seed,
            linear_type=linear_type,
            outcome_sub_list=outcome_sub_list,
            max_iter=max_iter,
        )
        self.two_layer_base = two_layer_base

    def _validate_params(self):
        if self.num_layers not in (2, 3):
            raise ValueError(f"num_layers must be one of [2, 3]. Got {self.num_layers}")

        if (self.two_layer_base is None) != (self.num_layers == 3):
            raise ValueError(
                f"two_layer_base must be non-None exactly when num_layers=2. "
                f"Got two_layer_base={self.two_layer_base}, num_layers={self.num_layers}"
            )
        if self.num_layers == 2 and self.two_layer_base not in ("S", "Q"):
            raise ValueError(
                f"two_layer_base must be 'S' or 'Q'. Got {self.two_layer_base}"
            )

    def _reported_stages(self):
        """A stage is reported iff predict_all adds its prediction:
        3 layers      -> S, Q, T
        2 layers, "S" -> S, T   (stage2_ == stage1_; reporting both would
                                 double-count one set of coefficients)
        2 layers, "Q" -> Q, T   (stage1_ is fit but zeroed in predict_all)
        """
        stages = []
        if self.num_layers == 3:
            stages.append(("stage1", self.stage1_, self._feat1))
            stages.append(("stage2", self.stage2_, self._feat2))
        elif self.two_layer_base == "S":
            stages.append(("stage1", self.stage1_, self._feat1))
        elif self.two_layer_base == "Q":
            stages.append(("stage2", self.stage2_, self._feat2))
        else:
            raise ValueError(
                f"two_layer_base must be one of ['S','Q']. Got {self.two_layer_base}"
            )
        stages.append(("stage3", self.stage3_, self._feat3))

        seen = []
        for _, model, _ in stages:
            assert not any(model is s for s in seen), "stage reported twice"
            seen.append(model)
        return stages

    def fit(self, X, y, shuffle=None, perm_dir=None):
        """
        Use residual method
        Layer1--> Use sum scores to predict raw impact
        """
        # Validate params
        self._validate_params()
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
        _, self._feat3 = get_prob_col_outcomes(
            agg_type=self.agg_type,
            desc_seg_type=self.desc_seg_type,
        )
        if self.outcome_sub_list is not None:
            self._feat3 = list(self.outcome_sub_list)

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
                    alpha=self.alpha2,
                    model_type=self.linear_type,
                )
                self.stage2_.fit(X2, y)
                self.pred2_train_ = self.stage2_.predict(X2)
                resid2 = y - self.pred2_train_
            else:
                raise ValueError(
                    f"two_layer_base must be one of ['S', 'Q']. Got: {self.two_layer_base}"
                )

        ## Train with ridge
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
                "DAWBA_ID": X["DAWBA_ID"].values,
                "pred_stage1": pred1,
                "pred_stage2_increment": pred2,
                "pred_stage2_total": pred2_total,
                "pred_stage3_increment": pred3,
                "pred_stage3_total": pred3_total,
            }
        )
