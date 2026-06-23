## residual_func.py
from .config import SEED
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
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

## TODO: rename stage1, stage2, etc to stageSum, stageQuest, etc for readability


class _CoefHolder:
    """
    Minimal model-like object carrying back-mapped stage3 coefficients.
    Allows _extract_coefs to work without modification after PCA back-mapping.
    coef_ shape: (p,) — one weight per original probability feature.
    intercept_ shape: (1,) — scalar intercept, wrapped in array for consistency.
    """

    def __init__(self, coef_, intercept_):
        self.coef_ = np.asarray(coef_).flatten()
        self.intercept_ = np.atleast_1d(intercept_)


class BaseRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        num_layers,
        impact_type,
        desc_seg_type,
        quest_cols,
        agg_type,
        alpha2,
        l1_ratio2,
        alpha3,
        l1_ratio3,
        seed,
        impactme_outcome,
        linear_type,
        pca_k=None,  # can be int for # of PCs or float for % var explained or None for no PCA
        pca_scale=None,  # whether or not to apply standard scaling before PCA
        max_iter=5000,
    ):
        self.num_layers = num_layers
        self.impact_type = impact_type
        self.desc_seg_type = desc_seg_type
        self.quest_cols = quest_cols
        self.agg_type = agg_type
        self.alpha2 = alpha2
        self.l1_ratio2 = l1_ratio2
        self.alpha3 = alpha3
        self.l1_ratio3 = l1_ratio3
        self.max_iter = max_iter
        self.seed = seed
        self.linear_type = linear_type
        self.impactme_outcome = impactme_outcome
        self.pca_k = pca_k
        self.pca_scale = pca_scale
        self.perm_ids = None

    def _fit_pca(self, X3):
        """
        Fit PCA (and optionally StandardScaler) on X3 train, return scores
        Called once during fit(), stores self.pca_ and (and optionally self.pca_scaler_)
        If self.pca_k is None, returns X3 unchanged
        """
        if self.pca_k is None:
            return X3
        if self.pca_scale:
            self.pca_scaler_ = StandardScaler().fit(X3)
            X3 = self.pca_scaler_.transform(X3)
        self.pca_ = PCA(n_components=self.pca_k).fit(X3)
        X3_pcs = self.pca_.transform(X3)
        return X3_pcs

    def _transform_pca(self, X3):
        """
        Apply fitted PCA (and optional scaler) to X3 test.
        If self.pca_k is None, return X3 unchanged.
        Similar to _fit_pca() w/o PCA/Scaler fitting.
        """
        if self.pca_k is None:
            return X3
        if self.pca_scale:
            X3 = self.pca_scaler_.transform(X3)
        X3_pcs = self.pca_.transform(X3)
        return X3_pcs

    def _finalize_stage3_coefs(self):
        """
        After stage3_ (text) is fitted on PCA scores,back-map regression weights
        to original feat space and store self._stage3_coef_rep fot use in get_coef_df

        If pca_k is None, self._stage3_coef_rep aliases stage3_ directly.

        Backmapping:
            w_caled
        """
        if self.pca_k is None:
            self._stage3_coef_rep = self.stage3_
            return
        # loading matrix w/ dimension (#PCs, #feats)
        V = self.pca_.components_
        # weights assigned to each PC
        beta = np.atleast_1d(self.stage3_.coef_).flatten()
        # weights transformed back to original input space
        w = V.T @ beta
        # original intercept
        intercept = np.atleast_1d(self.stage3_.intercept_).flatten()[0]
        if self.pca_scale:
            # undo StandardScaler
            w_raw = w / self.pca_scaler_.scale_
            intercept -= np.sum(w_raw * self.pca_scaler_.mean_)
        else:
            w_raw = w

        assert len(w_raw) == len(
            self._feat3
        ), f"Back-mapped weights ({len(w_raw)} don't match _feat3 ({len(self._feat3)}))"

        self._stage3_coef_rep = _CoefHolder(w_raw, intercept)

    def _build_X_blocks(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        quest_np = X[self.quest_cols].to_numpy(dtype=float)

        quest_sum = quest_np.sum(axis=1, keepdims=True)

        prob_cols = self._get_prob_col()
        prob_vals = X[prob_cols].values
        prob_vec = np.vstack(prob_vals).astype(float)

        return quest_sum, quest_np, prob_vec

    def _get_prob_col(self):
        if self.agg_type in ["avg", "max", "block"]:
            return f"prob_vector_{self.agg_type}"
        elif self.agg_type in ["proba-limited", "binary"]:
            if self.desc_seg_type == "new":
                return IMPACTME_OUTCOMES_FULL
            elif self.desc_seg_type == "expanded":
                return OUTCOMES_EXPAND
            elif self.desc_seg_type in ["original", "augment_neg", "augment"]:
                return IMPACTME_OUTCOMES_SHOT
            elif self.desc_seg_type == "combined":  # combines expanded + original
                return [f"{col}_impactme" for col in IMPACTME_OUTCOMES_SHOT] + [
                    f"{col}_expand" for col in OUTCOMES_EXPAND
                ]
            else:
                raise ValueError(f"Unrecognized desc_seg_type: {self.desc_seg_type}")
        raise ValueError(f"Unrecognized agg_type: {self.agg_type}")

    def get_coef_df(self):
        parts = [self._extract_coefs(self.stage1_, self._feat1, "stage1")]
        if self.num_layers == 1:
            return pd.concat(parts, ignore_index=True)
        ## 2/3 layers
        parts.append(self._extract_coefs(self._stage3_coef_rep, self._feat3, "stage3"))
        if self.num_layers == 3:
            # only add stage2 info if 3 layers
            # stage2 info == stage1 info when 2 layers
            parts.append(self._extract_coefs(self.stage2_, self._feat2, "stage2"))
        return pd.concat(parts, ignore_index=True)

    def _make_model(self, alpha, l1_ratio, model_type):
        """
        model_type: str
            One of ['elastic', 'regular', 'ridge']
        """
        if model_type == "elastic":
            return ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
        elif model_type == "ridge":
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


class IndependentRegressor(BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, shuffle=None, perm_dir=None):
        """
        Create linear/ridge regression for each layer individually
        Layer1--> Use sum scores to predict raw impact
        Layer2--> Use granular questionnaire response to predict raw impact
        Layer3--> Use text probs to predict raw impact
        *NOTE: use alpha3/l1_ratio3 for both questionnaire and text models
        *NOTE: PCA can optionally be applied to layer3
        """
        y = np.asarray(y, dtype=float)
        X1, X2, X3 = self._build_X_blocks(X)

        # if shuffle is not None:
        #     raise ValueError(f"Single model does not accept shuffling. Got: {shuffle}")
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
        prob_cols = self._get_prob_col()

        if self.agg_type in ["avg", "max", "block"]:
            # LR output doesnt have a8,e3,34
            self._feat3 = IMPACTME_OUTCOMES_LR
        else:
            self._feat3 = prob_cols

        ## Layer 1 — Sum scores
        self.stage1_ = self._make_model(alpha=None, l1_ratio=None, model_type="regular")
        self.stage1_.fit(X1, y)

        ## Layer 2 - Granular Questionnaire
        self.stage2_ = self._make_model(
            alpha=self.alpha3,
            l1_ratio=self.l1_ratio3,
            model_type=self.linear_type,
        )
        self.stage2_.fit(X2, y)

        ## Layer 3 - Text probs
        X3 = self._fit_pca(X3)  # will not use PCA if self.pca_k is None
        self.stage3_ = self._make_model(
            alpha=self.alpha3,
            l1_ratio=self.l1_ratio3,
            model_type=self.linear_type,
        )
        self.stage3_.fit(X3, y)
        self._finalize_stage3_coefs()  # map coefs back to og feat space if PCA used
        return self

    def predict_all(self, X):
        """
        Returns all stage predictions and cumulative totals in a DataFrame.
        """
        X1, X2, X3 = self._build_X_blocks(X)
        X3 = self._transform_pca(X3)  # will not use PCA if self.pca_k is None

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

        ## Shuffle before PCA
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
        prob_cols = self._get_prob_col()

        if self.agg_type in ["avg", "max", "block"]:
            # LR output doesnt have a8,e3,34
            self._feat3 = IMPACTME_OUTCOMES_LR
        else:
            self._feat3 = prob_cols
        # self._feat3 = IMPACTME_OUTCOMES_SHOT

        ## Layer 1 — Sum scores
        self.stage1_ = self._make_model(alpha=None, l1_ratio=None, model_type="regular")
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
                l1_ratio=self.l1_ratio2,
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
                    l1_ratio=self.l1_ratio3,
                    model_type=self.linear_type,
                )
                self.stage2_.fit(X2, y)
                self.pred2_train_ = self.stage2_.predict(X2)
                resid2 = y - self.pred2_train_
            else:
                raise ValueError(
                    f"two_layer_base must be one of ['S', 'Q']. Got: {self.two_layer_base}"
                )

        ## Layer 3 — Text probabilities (optional PCA)
        X3 = self._fit_pca(X3)  # PCA not used if self.pca_k is None
        if self.impactme_outcome is not None:
            pass  # TODO: do we even need this?
            # # Train with regular univariable Linear Regression
            # self.stage3_ = self._make_model(
            #     alpha=None, l1_ratio=None, model_type="regular"
            # )
        else:
            ## Train with elastic net
            self.stage3_ = self._make_model(
                alpha=self.alpha3,
                l1_ratio=self.l1_ratio3,
                model_type=self.linear_type,
            )
        try:
            self.stage3_.fit(X3, resid2)
        except ValueError:
            print(X3)
        self.pred3_train_ = self.stage3_.predict(X3)
        self._finalize_stage3_coefs()
        return self

    def predict_all(self, X):
        """
        Returns all stage predictions and cumulative totals in a DataFrame.
        """
        X1, X2, X3 = self._build_X_blocks(X)
        X3 = self._transform_pca(X3)  # PCA not used if self.pca_k is None

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
    selected_l1s,
    split_id,
    impact_type,
    impactme_outcome,
    max_iter,
    seed,
    linear_type,  # ridge
    perm_dir,
    two_layer_base,  # first estimator type for 2 layer iteration; one of 'Q', 'T'
    pca_k=None,  # None to disable; int/float to enable
    pca_scale=True,  # standardize
    hypothesis_shuffle=None,  # None, 'text'. ' or 'resid1'
):
    quest_cols = col_config_sensitivity[f"{impact_type.lower()}_quest"]
    ## Make prob cols granular ######################
    ## TODO: Use this instead of granularizing myself
    # new_cols = pd.DataFrame(
    #     df["open_text_block_0_shot"].tolist(),
    #     columns=IMPACTME_OUTCOMES_FULL,
    #     index=df.index,
    # )
    # df = df.join(new_cols)
    # df = df.drop(columns="open_text_block_0_shot")
    ########################################
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
    shared_args = dict(
        num_layers=num_layers,
        impact_type=impact_type,
        quest_cols=quest_cols,
        agg_type=agg_type,
        alpha2=selected_alphas[0],
        l1_ratio2=selected_l1s[0],
        alpha3=selected_alphas[1],
        l1_ratio3=selected_l1s[1],
        max_iter=max_iter,
        seed=seed,
        impactme_outcome=impactme_outcome,
        linear_type=linear_type,
        desc_seg_type=desc_seg_type,
        pca_k=pca_k,
        pca_scale=pca_scale,
    )
    if num_layers == 1:
        model = IndependentRegressor(**shared_args)
    else:
        model = ResidualRegressor(**shared_args, two_layer_base=two_layer_base)

    model.fit(train_reset, y_train, shuffle=hypothesis_shuffle, perm_dir=perm_dir)

    pred_df = model.predict_all(test_reset)

    # ── Coefficients ──────────────────────────────────────────────────────────
    coef_df = model.get_coef_df()
    coef_df["split_id"] = split_id  # tag with fold so rows stay traceable

    # ── Permutation IDs ───────────────────────────────────────────────────────
    perm_id_df = pd.DataFrame(
        {f"perm_id_split_{split_id}": model.perm_ids},
        index=list(range(len(model.perm_ids))),
    )

    # ── Predictions ───────────────────────────────────────────────────────────
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
