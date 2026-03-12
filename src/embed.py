import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np


def import_df(import_path):
    og_df_path = Path(import_path)
    if og_df_path.suffix == ".csv":
        df_import = pd.read_csv(og_df_path, index_col=0)
    elif og_df_path.suffix == ".tsv":
        df_import = pd.read_csv(og_df_path, sep="\t", index_col=0)
    elif og_df_path.suffix == ".parquet":
        df_import = pd.read_parquet(og_df_path)
    else:
        raise ValueError(f"Unrecognized og_df_path file type: {og_df_path.suffix}")
    return df_import


def gen_embeds(
    og_df_path,
    text_col_name,
    model_id,
    log_path,
    result_path,
):
    """
    Params
    -------
    og_df_path: pandas DF
        Original df containing text to be embeded
    text_col_name: str
        Name of text column to be embedded, will usually be:
        - question_answer --> for block-level embeddings/preds
    model_id: str
        path to local model to be used
    log_path: str
        path to log file
    result_path: str
        path to where resulting df with embeddings will be saved
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set")
    ## =========================> Logger set up
    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logging.getLogger("embeder").setLevel(logging.WARNING)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filemode="w",
        force=True,
    )
    pid = os.getpid()
    logging.info(f"PID={pid} GPUs={visible}")
    ## ========================> Encoder set up
    from transformers import LlamaModel, PreTrainedTokenizerFast
    import torch

    llama3_tokenizer_fast = PreTrainedTokenizerFast.from_pretrained(model_id)
    llama3_model = LlamaModel.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="balanced",
    )
    assert llama3_model.device.type == "cuda"
    logging.info(f"Successfully loaded model: {model_id}")
    ## Load df containing text to be embeded
    df_import = import_df(og_df_path)
    num_embeds = len(df_import)
    logging.info(f"Successfully loaded df of length: {num_embeds}")
    ## =======================> Loop through rows and embed
    new_row_list = []
    for counter, (_, row) in enumerate(df_import.iterrows()):
        ## ===================> Output progress
        prog = f"{counter} / {num_embeds}"
        logging.info(prog)
        # output_progress(counter, num_embeds)
        text = getattr(row, text_col_name)
        inputs = llama3_tokenizer_fast(text, return_tensors="pt")
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = llama3_model(**inputs, return_dict=True)  # [0]
            ## Avgs last hidden state's weights over all tokens
            version1 = (
                outputs.last_hidden_state[:,].to("cpu").detach().numpy().mean(axis=1)[0]
            )
            ## Append embedding
            new_row = row.to_dict().copy()
            new_row["embedding"] = version1.tolist()
            new_row_list.append(new_row)
    embed_df = pd.DataFrame(new_row_list)
    assert (
        len(embed_df) == num_embeds
    ), f""" Generated df with embeddings does not match original length:
        Original length: {num_embeds}
        Embed df length: {len(embed_df)}
    """.strip()
    ## =======================> Export
    result_path = Path(result_path)
    if result_path.exists():
        result_path.unlink()
    result_path.parent.mkdir(exist_ok=True, parents=True)
    embed_df.to_parquet(result_path)
    logging.info("Successfully generated all embeddings!!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--og_df_path", required=True)
    parser.add_argument("--text_col_name", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--result-path", required=True)

    args = parser.parse_args()

    gen_embeds(
        og_df_path=args.og_df_path,
        text_col_name=args.text_col_name,
        model_id=args.model_id,
        log_path=args.log_path,
        result_path=args.result_path,
    )


if __name__ == "__main__":
    main()
