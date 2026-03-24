from src.utils import col_config_sensitivity
from src.config import LR_MODEL_BASE_DIR
import joblib
import numpy as np
import pandas as pd
import torch


def validate_input(impact_type, model_name, impactme_seg_type):
    if impact_type not in ["self", "parent"]:
        raise ValueError(
            f"Impact type must be one of ['self', 'parent'], got {impact_type}"
        )
    if impactme_seg_type not in ["turns", "pt_noshort"]:
        raise ValueError(
            f"Seg type must be one of ['turns', 'pt_noshort'], got {impactme_seg_type}"
        )
    if model_name not in ["Llama-3.3-8B-Instruct", "Llama-3.3-70B-Instruct"]:
        raise ValueError(
            f"model_name must be one of ['Llama-3.3-8B-Instruct', 'Llama-3.3-70B-Instruct'], got {model_name}"
        )


def add_prob_vector_avg_max(embed_df, impact_type, outcome_order, lr_model_dict):
    text_cols = col_config_sensitivity[f"{impact_type}_text"]
    probs_by_text = {}  # {text_col: DF(n_participants x n_outcomes)}
    for text_col in text_cols:
        embed_col = f"{text_col}_embedding"
        ## ==========> Get probs
        X = np.stack(embed_df[embed_col].values).astype(np.float16)
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        this_probs = {}
        for outcome_name in outcome_order:
            lr_model = lr_model_dict[outcome_name]
            y_proba = lr_model.predict_proba(X)[:, 1]
            this_probs[outcome_name] = y_proba
        probs_df = pd.DataFrame(this_probs, index=embed_df.index)
        probs_by_text[text_col] = probs_df

    # 3D tensor: [participant, outcome, question]
    # (axis=0 for participant, axis=1 for outcome, axis=2 for question)
    arr = np.stack([probs_by_text[c].to_numpy() for c in text_cols], axis=2)

    # Aggregate over questions (axis 2)
    probs_avg = pd.DataFrame(
        arr.mean(axis=2), index=embed_df.index, columns=outcome_order
    )
    probs_max = pd.DataFrame(
        arr.max(axis=2), index=embed_df.index, columns=outcome_order
    )

    # Add cols to df
    embed_df["prob_vector_avg"] = probs_avg.to_numpy().tolist()
    embed_df["prob_vector_max"] = probs_max.to_numpy().tolist()
    return embed_df


def add_prob_vector_block(embed_df, outcome_order, lr_model_dict):
    proba_dict = {}
    ## ==========> Get probs
    X = np.stack(embed_df["open_text_block_embedding"].values).astype(np.float16)
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    for outcome_name in outcome_order:
        lr_model = lr_model_dict[outcome_name]
        y_proba = lr_model.predict_proba(X)[:, 1]
        proba_dict[outcome_name] = y_proba
    # Df w/ each row is participant and col is prob of some outcome
    probs_df = pd.DataFrame(proba_dict)
    # Ensure correctly ordered
    probs_df = probs_df[outcome_order]
    # Validate number of rows matches og df
    if len(probs_df) != len(embed_df):
        raise ValueError("Number of probability rows doesn't match embed_df length")
    ### Format probs + add to df (concat all probs in a row in a single list)
    embed_df["prob_vector_block"] = probs_df.to_numpy().tolist()
    return embed_df


def add_prob_vector(
    embed_path, impact_type, model_name, impactme_seg_type, export_path, make_0=True
):
    """
    Params
    ------

    make_0: Boolean
        Flag indicating whether or not to fill in prob vectors with np.zeros() for participants who weren't prompted to fill open text responses
    """
    ## ========> Validate input
    validate_input(impact_type, model_name, impactme_seg_type)
    ## ========== Load df w/ text embeddings
    embed_df = pd.read_parquet(embed_path)
    ## ============> Load ImpactMe LR models
    lr_model_dir = LR_MODEL_BASE_DIR / impactme_seg_type / model_name
    lr_model_dict = {}
    for outcome_file in lr_model_dir.iterdir():
        outcome_name = outcome_file.stem
        # Skips `any` and any comprehensive `a` or `c` outcomes
        if len(outcome_name) != 2:
            continue
        lr_model_dict[outcome_name] = joblib.load(outcome_file)
    outcome_order = sorted(lr_model_dict.keys())
    if len(outcome_order) != 28:
        raise ValueError(
            f"outcome_order should be length 28, got {len(outcome_order)}. \nOutcomes: {outcome_order}"
        )
    ## Add avg, max prob vectors from segments
    embed_df = add_prob_vector_avg_max(
        embed_df, impact_type, outcome_order, lr_model_dict
    )
    ## Add raw prob vector from block text
    embed_df = add_prob_vector_block(embed_df, outcome_order, lr_model_dict)

    # Replace prob embeddings w/ vector of all 0's if question not answered (designated by `Not applicable` or `Not answered`)
    mask = embed_df["no_text"].astype(bool)
    n_matches = mask.sum()
    if (n_matches != 0) and make_0:
        zero_vectors = [np.zeros(len(outcome_order)) for _ in range(n_matches)]
        embed_df.loc[mask, "prob_vector_block"] = pd.Series(
            zero_vectors, index=embed_df.index[mask]
        )
        embed_df.loc[mask, "prob_vector_avg"] = pd.Series(
            zero_vectors, index=embed_df.index[mask]
        )
        embed_df.loc[mask, "prob_vector_max"] = pd.Series(
            zero_vectors, index=embed_df.index[mask]
        )
    # ====> Export
    if export_path.exists():
        export_path.unlink()
    export_path.parent.mkdir(exist_ok=True, parents=True)
    embed_df.to_parquet(export_path)
