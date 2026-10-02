import numpy as np
import pandas as pd
import os
import transformers
import torch
import gc
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
import time
import argparse
import re

transformers.logging.set_verbosity_error()


start_time = time.time()

date_string = datetime.now().strftime("%Y-%m-%d")
print(date_string)

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Torch version: {torch.__version__}")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ===============================
# Reading questions / outcomes
# ===============================
df_questions = pd.read_excel("80_Outcomes_Enhanced.V2.xlsx")

# l_questions = df_questions.Question_V2.to_list()

prompt_type = "few"  # few

if prompt_type == "zero":
    l_questions = df_questions.Question_V1.to_list()
elif prompt_type == "few":
    l_questions = df_questions.Question_V2.to_list()


l_outcomes = df_questions.Outcome.to_list()

binary_features = []
ind_ = 1
for emo_ in l_outcomes:
    emo_ = emo_.replace(" / ", " ")
    emo_ = emo_.replace(" ", "_")
    binary_features.append(f"S{ind_}_{emo_}")
    ind_ += 1

print(f"Binary features: {len(binary_features)}")


# ===============================
# Prompt functions (parent + self)
# ===============================
BINARY_OPTIONS_TEMPLATE = """- 1 - Yes: the parent's responses contain an implicit or explicit indication of "{symptom} associated with the child".
- 0 - No: the parent's responses do not contain an implicit or explicit indication of "{symptom} associated with the child"."""

SCALE_OPTIONS_TEMPLATE = """- 0 - Absent: the parent's responses contain no implicit or explicit indication of "{symptom} associated with the child".
- 1 - Very low: the parent's responses contain a faint, indirect, or highly ambiguous indication of "{symptom} associated with the child".
- 2 - Low to moderate: the parent's responses contain a clear but subdued indication of "{symptom} associated with the child", present but not salient.
- 3 - Moderate to high: the parent's responses contain a strong and clearly expressed indication of "{symptom} associated with the child", salient but not dominant.
- 4 - Very high: the parent's responses contain an explicit, dominant, or pervasive indication of "{symptom} associated with the child".
Note: Rate the severity of the symptom, not the number of mentions.
Implicit cues (tone, wording, context) should be considered."""


def do_prompt_question(question_, symptom_, clinical_note_, scale=False):
    if scale:
        task_line = f'Your task is to analyze the parent\'s responses to a questionnaire (at most 10 questions) and rate the severity of "{symptom_}" associated with his/her child, by answering the question below.'
        options = SCALE_OPTIONS_TEMPLATE.format(symptom=symptom_)
    else:
        task_line = f'Your task is to analyze the parent\'s responses to a questionnaire (at most 10 questions) and determine whether they contain an implicit or explicit indication of "{symptom_}" associated with his/her child, by answering the question below.'
        options = BINARY_OPTIONS_TEMPLATE.format(symptom=symptom_)

    prompt = f"""### Task
{task_line}

The input contains responses from a parent, structured as question/answer pairs (up to Q10/A10). Each answer is written by the parent and may contain relevant behavioral observations about the child.

---

### Instructions
- Consider both explicit statements and reasonable implicit indications.
- Base your judgment only on the information provided in the parent's responses.
- Evaluate the responses either collectively or each question independently, as appropriate.
- Use reasonable clinical inference when evidence is indirect or implied.
- Respond with exactly one token from the provided options.
- Do not generate any additional text.
- The output must exactly match one of the provided response options.

---

{question_}

---

### Question: 
#### Select the most appropriate response
{options}

---

### Questionnaire:  
{clinical_note_}

---

### Response:  
    """
    return prompt


BINARY_OPTIONS_TEMPLATE_SELF = """- 1 - Yes: the teenager's responses contain an implicit or explicit indication of "{symptom} associated with him/her".
- 0 - No: the teenager's responses do not contain an implicit or explicit indication of "{symptom} associated with him/her"."""

SCALE_OPTIONS_TEMPLATE_SELF = """- 0 - Absent: the teenager's responses contain no implicit or explicit indication of "{symptom} associated with him/her".
- 1 - Very low: the teenager's responses contain a faint, indirect, or highly ambiguous indication of "{symptom} associated with him/her".
- 2 - Low to moderate: the teenager's responses contain a clear but subdued indication of "{symptom} associated with him/her", present but not salient.
- 3 - Moderate to high: the teenager's responses contain a strong and clearly expressed indication of "{symptom} associated with him/her", salient but not dominant.
- 4 - Very high: the teenager's responses contain an explicit, dominant, or pervasive indication of "{symptom} associated with him/her".
Note: Rate the severity of the symptom, not the number of mentions.
Implicit cues (tone, wording, context) should be considered."""


def do_prompt_question_self(question_, symptom_, clinical_note_, scale=False):
    if scale:
        task_line = f'Your task is to analyze the teenager\'s own responses to a questionnaire (at most 10 questions) and rate the severity of "{symptom_}" associated with him/her, by answering the question below.'
        options = SCALE_OPTIONS_TEMPLATE_SELF.format(symptom=symptom_)
    else:
        task_line = f'Your task is to analyze the teenager\'s own responses to a questionnaire (at most 10 questions) and determine whether they contain an implicit or explicit indication of "{symptom_}" associated with him/her, by answering the question below.'
        options = BINARY_OPTIONS_TEMPLATE_SELF.format(symptom=symptom_)

    prompt = f"""### Task
{task_line}

The input contains responses from the teenager (self-report), structured as question/answer pairs (up to Q10/A10). Each answer is written by the teenager and may contain relevant behavioral observations about him/herself.

---

### Instructions
- Consider both explicit statements and reasonable implicit indications.
- Base your judgment only on the information in the teenager's responses.
- Evaluate the responses either collectively or each question independently, as appropriate.
- Use reasonable clinical inference when evidence is indirect or implied.
- Respond with exactly one token from the provided options.
- Do not generate any additional text.
- The output must exactly match one of the provided response options.

---

{question_}

---

### Question: 
#### Select the most appropriate response
{options}

---

### Questionnaire:  
{clinical_note_}

---

### Response:  
    """
    return prompt


################################
################################
# Llama
# Models
### ====
## Llama 3.3 and Llama 3.1
################################
################################
PATH_TO_MODELS = "meta-llama/"
MODEL_STR = "Llama-3.3-70B-Instruct"
# MODEL_STR = "Llama-3.1-8B-Instruct"

MODEL_ = PATH_TO_MODELS + MODEL_STR
print("Model: {}\n".format(MODEL_))

## Pipeline to generate text
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_,
    model_kwargs={"dtype": torch.bfloat16},
    device_map="auto",
)

## Token for ending a sequence
eos_token_id_ = pipeline.tokenizer.eos_token_id

## Avoid spurious "Both max_new_tokens and max_length seem to have been set" warnings:
## the model's generation_config ships with a default max_length=20, which conflicts
## with max_new_tokens even though max_new_tokens correctly takes precedence.
pipeline.model.generation_config.max_length = None

TOP_P = 0.8
TOP_K = 10
NUM_SEQ = 1  # number of sequences desired to generate


def query_LLM(prompt, temperature):
    MAX_NEW_TOKENS = 1
    response_ = ""
    messages = [{"role": "user", "content": prompt}]

    try:
        sequences = pipeline(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_token_id_,
            pad_token_id=eos_token_id_,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0 else None,
            top_p=TOP_P,
        )

        response_ = sequences[0]["generated_text"][-1]["content"].strip()

    except Exception as err:
        print(f"Error: {err} \nType: {type(err).__name__}")

    return response_


# ===============================
# Datasets to process: parent + self
# Each entry: (name, path_in, prompt_function)
# ===============================
datasets_config = [
    {
        "name": "parent",
        "path_in": "question_answer/df_parent_formatted.csv",
        "prompt_fn": do_prompt_question,
    },
    {
        "name": "self",
        "path_in": "question_answer/df_self_formatted.csv",
        "prompt_fn": do_prompt_question_self,
    },
]

# Columns to keep in the final output
keep_cols_base = ["DAWBA_ID", "formatted_text"]

# ===============================
# Save outputs
# ===============================
PATH_OUT = f"results/lancet/{prompt_type}/"
PATH_OUT_ = Path(PATH_OUT)
PATH_OUT_.mkdir(parents=True, exist_ok=True)

# TEMPERATURES = [0.0, 0.2] #, 0.4, 0.6, 0.8, 1.0]
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for TEMPERATURE in TEMPERATURES:
    print(f"\n{'='*50}")
    print(f"Testing with TEMPERATURE = {TEMPERATURE}")
    print(f"{'='*50}\n")

    for cfg in datasets_config:
        name_ = cfg["name"]
        path_in_ = cfg["path_in"]
        prompt_fn_ = cfg["prompt_fn"]

        print(f"\n=== Processing dataset: {name_} (temp={TEMPERATURE}) ===")
        df_full = pd.read_csv(path_in_)  # [:1]

        # Keep only the required base columns + open_text_block, for prompting
        df_base = df_full[keep_cols_base].copy()

        # Run BINARY then SCALE for this dataset
        for TYPE_RESPONSE, scale_flag in [("BIN", False), ("SC", True)]:
            print(f"--- {name_} | {TYPE_RESPONSE} | temp={TEMPERATURE} ---")

            df_out = df_base.copy()

            list_data = []
            for text_ in df_out["formatted_text"].to_list():
                text_ = str(text_).strip()

                list_row = []
                for index_ in range(len(l_outcomes)):
                    question_ = l_questions[index_]
                    symptom_ = l_outcomes[index_]

                    prompt = prompt_fn_(question_, symptom_, text_, scale=scale_flag)
                    # print(prompt)

                    response_raw = query_LLM(prompt, TEMPERATURE)
                    try:
                        response_ = int(response_raw)
                    except (ValueError, TypeError):
                        response_ = response_raw

                    # print(str(response_) + "\n\n******\n\n")
                    list_row.append(response_)

                list_data.append(list_row)

            df_out[binary_features] = list_data

            # ===============================
            # Save outputs
            # ===============================
            name_out_file_ = (
                f"{prompt_type}_{MODEL_STR}_{date_string}_{name_}_temp{TEMPERATURE}"
            )
            df_out.to_csv(
                PATH_OUT_ / f"{name_out_file_}_{TYPE_RESPONSE}.csv", index=None
            )

            print(
                f"Saved results for {name_} | {TYPE_RESPONSE} | temperature {TEMPERATURE}"
            )


# ===============================
# Execution time
# ===============================

end_time = time.time()
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60
execution_time_hours = execution_time_seconds / 3600

print("+++++++++++")
print(f"Start time: {start_time}")
print(f"End time: {end_time}")
print(f"Execution time: {execution_time_seconds:.2f} seconds")
print(f"Execution time: {execution_time_minutes:.2f} minutes")
print(f"Execution time: {execution_time_hours:.2f} hours")


print("***********************\nEND\n***********************")
