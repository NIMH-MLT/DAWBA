# shot.py
import os
from src.config import BASE_PATH
from src.utils import (
    OUTCOMES_EXPAND,
    IMPACTME_OUTCOMES_SHOT,
    OUTCOMES_COMBINED,
)

import json
import argparse
import math
import logging
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np


def make_shot_cmd(
    cur_model,
    outcome_list,
    impact_type_list,
    base_df_dir,
    json_config,
    gpu_group,
    outcome_desc_type,
    n_shot,
    output_type,
):
    temp = json_config["temp"]
    log_dir = (
        BASE_PATH
        / "logs"
        / f"shot_{n_shot}"
        / f"temp_{temp}"
        / output_type
        / cur_model
        / outcome_desc_type
    )
    save_dir = (
        BASE_PATH
        / "data"
        / f"w_{n_shot}_shot_probs"
        / f"temp_{temp}"
        / output_type
        / cur_model
        / outcome_desc_type
    )
    cmd = [
        "python",
        "src/shot_controller.py",
        "--gpu-groups",
        *gpu_group,
        "--outcomes",
        *outcome_list,
        "--impact_types",
        *impact_type_list,
        "--llm_config",
        json.dumps(json_config),
        "--base_df_dir",
        str(base_df_dir),
        "--save_dir",
        str(save_dir),
        "--log_dir",
        str(log_dir),
        "--outcome_desc_type",
        outcome_desc_type,
        "--n_shot",
        str(n_shot),
        "--output_type",
        output_type,
    ]
    return cmd


def get_outcome_list(desc_type):
    if desc_type == "original":
        return IMPACTME_OUTCOMES_SHOT
    elif desc_type == "expanded":
        return OUTCOMES_EXPAND
    elif desc_type == "combined":
        return OUTCOMES_COMBINED
    else:
        raise ValueError(f"Unrecognized desc_type: {desc_type}")


def setup_logger(log_path):
    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logging.getLogger("0-shot").setLevel(logging.WARNING)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filemode="w",
        force=True,
    )


def combine_original_expanded(save_path, impact_type, log_path):
    """
    Combine the original and expanded outcome descriptions for each participant.
    """
    setup_logger(log_path=log_path)
    logging.info(
        f"Combining original and expanded outcome descriptions for impact_type={impact_type}..."
    )
    base_path = save_path.parent.parent
    file_name = save_path.name
    original_output_path = base_path / "original" / file_name
    expanded_output_path = base_path / "expanded" / file_name

    if not original_output_path.exists():
        raise ValueError(f"Original output file does not exist: {original_output_path}")

    if not expanded_output_path.exists():
        raise ValueError(f"Expanded output file does not exist: {expanded_output_path}")

    shot_col = "open_text_block_0_shot"

    original_df = pd.read_parquet(original_output_path)
    original_df_sub = original_df[["DAWBA_ID", "open_text_block_0_shot"]].copy()
    original_df_sub[shot_col] = original_df_sub[shot_col].apply(
        lambda x: np.array(x) if isinstance(x, list) else x
    )

    expanded_df = pd.read_parquet(expanded_output_path)
    expanded_df_sub = expanded_df[["DAWBA_ID", "open_text_block_0_shot"]].copy()
    expanded_df_sub[shot_col] = expanded_df_sub[shot_col].apply(
        lambda x: np.array(x) if isinstance(x, list) else x
    )

    combined_df = pd.merge(
        original_df_sub,
        expanded_df_sub,
        on="DAWBA_ID",
        suffixes=("_original", "_expanded"),
        how="inner",
        validate="one_to_one",
    )
    # Concat such that expanded is original + expanded (in that order)
    combined_df["open_text_block_0_shot"] = [
        np.concatenate([o, e])
        for o, e in zip(
            combined_df["open_text_block_0_shot_original"],
            combined_df["open_text_block_0_shot_expanded"],
        )
    ]
    final_df = original_df.drop(columns=[shot_col]).merge(
        combined_df[["DAWBA_ID", shot_col]],
        on="DAWBA_ID",
        how="inner",
        validate="one_to_one",
    )
    missing = set(original_df["DAWBA_ID"]) - set(expanded_df["DAWBA_ID"])
    if missing:
        raise ValueError(f"IDs missing from expanded output: {sorted(missing)}")
    assert len(final_df) == len(original_df), "Length mismatch after merging"
    assert shot_col in final_df.columns, f"{shot_col} not found in final_df columns"
    num_res = len(final_df[shot_col].iloc[0])
    assert num_res == len(original_df[shot_col].iloc[0]) + len(
        expanded_df[shot_col].iloc[0]
    ), "Length mismatch in concatenated shot results"
    for row in final_df.itertuples():
        assert (
            len(row.open_text_block_0_shot) == num_res
        ), f"Length mismatch for DAWBA_ID {row.DAWBA_ID}"

    export_df(save_path=save_path, df=final_df, impact_type=impact_type)


########################################################################
############################## Prompting ###############################
########################################################################
def do_zero_prompt_parent(outcome_proper, outcome_desc, text_):
    prompt = f"""### Task
Your task is to analyze the provided text and determine whether it contains clear evidence of the psychological outcome "{outcome_proper}".

The text is taken from the open-response section of the The Development and Well-Being Assessment, answered by a parent of an adolescent participant.

---

#### Input format:
- Each new line pertains to a single `Question:... Answer:...` response sequence from a parent.
- All provided Question-Answer pairs pertain to the same parent.

---

### Instructions
- Respond using a single token only: 1 or 0.
- Do not generate any additional text.
- Do not include labels, hyphens, or parentheses.
- Base your judgment on the parent’s description of the adolescent, including the parent’s observations, reports, and perceptions of the adolescent’s thoughts, feelings, behavior, or functioning.
- If the outcome is about parenting, family dynamics, or the parent’s own wellbeing, include the parent’s own feelings, behavior, or experiences as relevant.
- Count both direct and clearly implied evidence when it matches the definition below.
---

### Definition of the psychological outcome "{outcome_proper}"
{outcome_desc}

---

### Question
Does the text contain evidence of the psychological outcome "{outcome_proper}" in the parent’s report?

---

Response options:
- 1 = Yes
- 0 = No

---
### Text:
{text_}

---

### Response:
"""
    return prompt


def do_zero_prompt_self(outcome_proper, outcome_desc, text_):
    prompt = f"""### Task
Your task is to analyze the provided text and determine whether it contains clear evidence of the psychological outcome "{outcome_proper}".

The text is taken from the open-response section of the The Development and Well-Being Assessment, answered by an adolescent participant. 

---

#### Input format:
- Each new line pertains to a single `Question:... Answer:...` response sequence from the adolescent participant.
- All provided Question-Answer pairs pertain to the same participant.

---

### Instructions
- Respond using a single token only: 1 or 0.
- Do not generate any additional text.
- Do not include labels, hyphens, or parentheses.
- Base your judgment only on the adolescent's own statements and experiences.
- Ignore statements about the parent, family, or other people unless the outcome definition explicitly includes them.
- Count both direct and clearly implied evidence when it matches the definition below.

---

### Definition of the psychological outcome "{outcome_proper}"
{outcome_desc}

---

### Question
Does the text contain evidence of the psychological outcome "{outcome_proper}" in the adolescent's own response?

---

Response options:
- 1 = Yes
- 0 = No

---
### Text:
{text_}

---

### Response:
"""
    return prompt


def get_prompt_meta(outcome_desc_type):
    """
    outcome_desc_type: str
        One of ['expanded', 'original']
    """
    if outcome_desc_type == "expanded":  # 80 new outcomes
        outcome_desc_path = (
            BASE_PATH / "data" / "complete_raw" / "80_Depression_Outcomes.xlsx"
        )
        descr_col = "Definition"
        outcome_proper_col = "Outcome"
    elif outcome_desc_type == "original":  # impactme outcomes
        outcome_desc_path = (
            BASE_PATH / "data" / "complete_raw" / "impactme_outcome_desc.xlsx"
        )
        outcome_proper_col = "Name"
        descr_col = "Description_Original"
    else:
        raise ValueError(
            f"`outcome_desc_type` must be one of ['expanded', 'original']. Got {outcome_desc_type}."
        )
    return outcome_desc_path, descr_col, outcome_proper_col


def gen_prompt(cur_outcome, text_, impact_type, outcome_desc_type, n_shot=0):
    """
    Parameters
    ---------
    outcome_desc_type: str
        One of ['expanded', 'original']
    """
    ## Param validation
    if impact_type not in ["self", "parent"]:
        raise ValueError(f"Unrecognized impact type: {impact_type}")
    ## Load information on outcome
    outcome_desc_path, descr_col, outcome_proper_col = get_prompt_meta(
        outcome_desc_type
    )
    outcome_desc_df = pd.read_excel(outcome_desc_path, index_col=0)
    outcome_proper = outcome_desc_df.loc[cur_outcome.upper()][outcome_proper_col]
    outcome_desc = outcome_desc_df.loc[cur_outcome.upper()][descr_col]
    ## Zero-shot prompting
    if n_shot == 0:
        if impact_type == "parent":
            prompt = do_zero_prompt_parent(
                outcome_proper=outcome_proper, outcome_desc=outcome_desc, text_=text_
            )
        else:  # self
            prompt = do_zero_prompt_self(
                outcome_proper=outcome_proper, outcome_desc=outcome_desc, text_=text_
            )
    ## N-shot prompting
    ## TODO: fill this out later if want to do n-shot
    else:
        raise ValueError("n_shot should be 0")
    return prompt


########################################################################
############################### Helpers ################################
########################################################################
def get_output_binary(
    pipeline,
    prompt,
    eos_token_id,
    temp,
    top_p,
):
    message = [{"role": "user", "content": prompt}]
    sequences = pipeline(
        message,
        max_new_tokens=1,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        do_sample=True,
        temperature=temp,
        top_p=top_p,
        use_cache=True,
        num_beams=1,  # Ensure not using beam search accidentally
    )
    raw_output = sequences[0]["generated_text"][-1]["content"].strip()
    try:
        output = int(raw_output)
        # logging.info("Success! Adding prediction...")
    except ValueError:
        output = -1
        logging.error(f"\t Failure! Output: {raw_output}; Adding -1...")
    return output


def get_output_proba(model, tokenizer, prompt, token_ids, limit_output, temp):
    """
    Params
    ------
    limit_output: boolean
        If true, limit output of model to 0/1, otherwise allow all tokens
    """
    model_input = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    id_0 = token_ids["0"]
    id_1 = token_ids["1"]
    ## Move model input to devide (cuda)
    device = next(model.parameters()).device
    model_input = {k: v.to(device) for k, v in model_input.items()}
    import transformers

    transformers.logging.set_verbosity_info()
    with torch.no_grad():
        if limit_output:
            valid_tokens = [id_0, id_1]
            bad_words = [
                [token_id]
                for token_id in range(len(tokenizer))
                if [token_id] not in valid_tokens
            ]
            assert len(bad_words) == (len(tokenizer) - 2)
            output_probs = model.generate(
                **model_input,  # input_ids & attention mask
                max_new_tokens=1,
                num_return_sequences=1,
                do_sample=True,  # needed for dynamic temp
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                suppress_tokens=[tokenizer.eos_token_id],
                bad_words_ids=bad_words,
                temperature=temp,
                output_scores=True,  # return logit scores at each generation step
                return_dict_in_generate=True,
            )
        else:  # allow all tokens to be explored
            output_probs = model.generate(
                **model_input,  # input_ids & attention mask
                max_new_tokens=1,
                do_sample=True,  # needed for dynamic temp
                temperature=temp,
                output_scores=True,  # return logit scores at each generation step
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    logits = output_probs.scores[0]  # (1, vocab_size)
    probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
    p0 = probs[0, id_0].sum().item()  # sum prob mass over all "0" surface forms
    p1 = probs[0, id_1].sum().item()  # sum prob mass over all "1" surface forms
    total = p0 + p1
    # Normalized: probability of "1" given only the two classes
    p1_norm = p1 / total if total > 0 else 0.5
    if limit_output and (not math.isclose(p1_norm, p1, rel_tol=1e-5)):
        raise ValueError(
            f"Normalized P(1) should ~= raw P(1) when limiting output. Got: \nRaw P(1)={1} \nNormalized P(1)={p1_norm}"
        )
    ############## EXPLORE OUTPUT
    res = {
        "prob_0_raw": p0,
        "prob_1_raw": p1,
        "prob_1_normalized": p1_norm,
        "pred": 1 if p1 >= p0 else 0,
    }
    return res


## get_results_proba() helper
def get_binary_token_ids(tokenizer):
    logging.info(f"Searching for 0/1 token ids...")
    variants = {"0": [], "1": []}
    for token_id in range(tokenizer.vocab_size):
        surface = tokenizer.decode([token_id])
        for digit in ["0", "1"]:
            if surface.strip() == digit:  # strip whitespace, check if core char matches
                variants[digit].append(token_id)
                logging.info(
                    f"Found '{repr(surface)}' (id={token_id}) for digit '{digit}'"
                )
    assert variants["0"] and variants["1"], "Could not find valid token IDs for 0 or 1"
    return variants


## get_shot_results() helper
def export_df(save_path, df, impact_type):
    try:
        save_path = Path(save_path)
        if save_path.exists():
            save_path.unlink()
        save_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_parquet(save_path)
        ## Ensure saved
        assert save_path.exists()
        logging.info("!" * 100)
        logging.info("!" * 100)
        logging.info("!" * 100)
        logging.info("!" * 100)
        logging.info(f"Completed impact_type: {impact_type}!!")
    except Exception as e:
        logging.error("=" * 100)
        logging.error("=" * 100)
        logging.error("=" * 100)
        logging.error(f"ERROR SAVING 0-shot DF: {e}")


########################################################################
######################### Sequential loopers ###########################
########################################################################
## Binary
def get_results_binary(
    text_df,
    outcome_names,
    llm_model,
    eos_token_id,
    temp,
    top_p,
    outcome_desc_type,
    impact_type,
    n_shot,
):
    """
    Parameters
    ---------
    outcome_desc_type: str
        One of ['original', 'short', 'augment']
    output_type: str
        One of ['binary', 'proba', ''proba-limited']
    """
    df_rows = []
    tot_rows = len(text_df)
    for counter, (_, row) in enumerate(text_df.iterrows()):
        logging.info(f"{counter}/{tot_rows}")
        text_ = row.open_text_block
        outcome_probs = []
        outcomes_ordered = sorted(outcome_names)
        for outcome_name in outcomes_ordered:
            prompt = gen_prompt(
                cur_outcome=outcome_name,
                text_=text_,
                impact_type=impact_type,
                outcome_desc_type=outcome_desc_type,
                n_shot=n_shot,
            )
            output = get_output_binary(
                pipeline=llm_model,
                prompt=prompt,
                eos_token_id=eos_token_id,
                temp=temp,
                top_p=top_p,
            )
            outcome_probs.append(output)
        new_row = row.to_dict()
        new_row["open_text_block_0_shot"] = np.array(outcome_probs)
        df_rows.append(new_row)
    return pd.DataFrame(df_rows)


## Probability
def get_results_proba(
    tokenizer,
    model,
    text_df,
    outcome_names,
    outcome_desc_type,
    n_shot,
    temp,
    impact_type,
    limit_output,
):
    token_ids = get_binary_token_ids(tokenizer)
    df_rows = []
    tot_rows = len(text_df)
    for counter, (_, row) in enumerate(text_df.iterrows()):
        logging.info(f"{counter}/{tot_rows}")
        text_ = row.open_text_block
        outcome_probs = []
        # Ensure outcome order
        outcomes_ordered = sorted(outcome_names)
        for outcome_name in outcomes_ordered:
            prompt = gen_prompt(
                cur_outcome=outcome_name,
                text_=text_,
                impact_type=impact_type,
                outcome_desc_type=outcome_desc_type,
                n_shot=n_shot,
            )
            output = get_output_proba(
                model,
                tokenizer,
                prompt,
                token_ids,
                limit_output=limit_output,
                temp=temp,
            )
            proba = output["prob_1_normalized"]
            outcome_probs.append(proba)
        new_row = row.to_dict()
        new_row[f"open_text_block_0_shot"] = np.array(outcome_probs)
        df_rows.append(new_row)
    return pd.DataFrame(df_rows)


########################################################################
############################# Main functions ###########################
########################################################################
def get_shot_results(
    outcome_names,
    impact_type,
    llm_config,
    base_df_path,
    save_path,
    log_path,
    outcome_desc_type,
    n_shot,
    output_type,
):
    """
    Parameters
    ---------
    outcome_desc_type: str
        One of ['original', 'expanded', 'combined']
    output_type: str
        One of ['binary', 'proba', 'proba-limited']
    """
    model_id = llm_config["model_id"]
    temp = llm_config["temp"]
    # ==================> Set up env
    ## CUDA imports
    import transformers

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set")
    ## =========================> Logger set up
    setup_logger(log_path=log_path)
    pid = os.getpid()
    logging.info(f"PID={pid} Impat Type={impact_type} GPUs={visible}")
    logging.info(f"n-shot: {n_shot} temp: {temp} model_id: {model_id}")
    ## =================> Load df
    text_df = pd.read_parquet(base_df_path)
    logging.info(f"Loaded df with shape: {text_df.shape}")
    ## =============> Set up model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="balanced",
    )
    logging.info(f"RAW Model: {Path(model_id).name} loaded")
    if output_type == "binary":
        top_p = llm_config["top_p"]
        ## pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="balanced",
        )
        eos_token_id = pipeline.tokenizer.eos_token_id
        logging.info(f"Running model for binary output...")
        ## Run model
        df_w_preds = get_results_binary(
            text_df=text_df,
            outcome_names=outcome_names,
            llm_model=pipeline,
            eos_token_id=eos_token_id,
            temp=temp,
            top_p=top_p,
            outcome_desc_type=outcome_desc_type,
            impact_type=impact_type,
            n_shot=n_shot,
        )
    else:  ## probability output, no need for pipeline wrapper
        limit_output = True if output_type == "proba-limited" else False
        logging.info(
            f"Running model for probability output with limited output={limit_output}, temp={temp}..."
        )
        df_w_preds = get_results_proba(
            tokenizer=tokenizer,
            model=model,
            text_df=text_df,
            outcome_names=outcome_names,
            outcome_desc_type=outcome_desc_type,
            n_shot=n_shot,
            temp=temp,
            impact_type=impact_type,
            limit_output=limit_output,
        )
    # ========================>  Export
    export_df(
        save_path=save_path,
        df=df_w_preds,
        impact_type=impact_type,
    )
    assert Path(save_path).exists()
    logging.info(f"Successfully exported to path: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome_list", required=True, nargs="+")
    parser.add_argument("--impact_type", required=True, choices=["parent", "self"])
    parser.add_argument("--llm_config", required=True)
    parser.add_argument("--base_df_path", required=True, type=Path)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--log_path", required=True, type=Path)
    parser.add_argument(
        "--outcome_desc_type",
        required=True,
        choices=[
            "original",
            "combined",
            "expanded",
        ],
    )
    parser.add_argument(
        "--n_shot",
        required=True,
        help="Must be convertable to int",
    )
    parser.add_argument(
        "--output_type", required=True, choices=["binary", "proba", "proba-limited"]
    )

    args = parser.parse_args()
    # Validate args
    llm_config = json.loads(args.llm_config)
    n_shot = int(args.n_shot)
    if args.outcome_desc_type == "combined":
        combine_original_expanded(
            save_path=args.save_path,
            impact_type=args.impact_type,
            log_path=args.log_path,
        )
    else:
        get_shot_results(
            outcome_names=args.outcome_list,
            impact_type=args.impact_type,
            llm_config=llm_config,
            base_df_path=args.base_df_path,
            save_path=args.save_path,
            log_path=args.log_path,
            outcome_desc_type=args.outcome_desc_type,
            n_shot=n_shot,
            output_type=args.output_type,
        )


if __name__ == "__main__":
    main()
