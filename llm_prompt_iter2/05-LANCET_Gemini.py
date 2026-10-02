import numpy as np
import pandas as pd
import os
import gc
import random
import time
import torch
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from google.genai import types
from pathlib import Path
from datetime import datetime
import argparse
import re

start_time = time.time()

date_string = datetime.now().strftime("%Y-%m-%d")
print(date_string)

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Torch version: {torch.__version__}")


os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ===============================
# Reading questions / outcomes
# ===============================
df_questions = pd.read_excel("80_Outcomes_Enhanced.V2.xlsx")

# l_questions = df_questions.Question_V2.to_list()

prompt_type = "few"  # zero   few

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
    # binary_features.append(f"Symptom_{ind_}_{emo_}")
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
# GEMINI
# Model
################################
################################
# ===============================
# GEMINI Model setup
# ===============================
key_path = "../17-NER/dataset/nih-nimh-irp-mlt-ab50f8b00127.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

PROJECT_ID = "nih-nimh-irp-mlt"
# LOCATION = "global"
LOCATION = "us-central1"

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(
        api_version="v1",
        headers={"X-Vertex-AI-LLM-Request-Type": "shared"},
    ),
)

model_id = "gemini-2.5-pro"
MODEL_STR = model_id
# gemini-2.5-pro
# gemini-2.5-flash
# gemini-2.5-flash-lite
# gemini-2.0-flash-001
# gemini-3.1-pro-preview

TOP_P = 0.7
TOP_K = 10
NUM_SEQ = 1

# MAX_NEW_TOKENS per mode
MAX_NEW_TOKENS_BIN = 150  # binary  → short response (0 or 1)
MAX_NEW_TOKENS_SC = 150  # scale   → short response (0–4)

print(f"Model: {model_id}\n")


def query_LLM(prompt, temperature, mode="BIN", max_retries=7):
    """Query Gemini with backoff on 429 errors.
    Forces an integer response; retries if the response is None or non-parseable."""
    MAX_NEW_TOKENS = MAX_NEW_TOKENS_BIN if mode == "BIN" else MAX_NEW_TOKENS_SC
    response_ = ""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=max(
                        temperature, 0.01
                    ),  # keep >0 for consistency with prior behavior
                    candidate_count=NUM_SEQ,
                    top_p=TOP_P,
                    max_output_tokens=MAX_NEW_TOKENS,
                ),
            )

            if response.text is None:
                print(
                    f"None response on attempt {attempt + 1}/{max_retries}, retrying..."
                )
                wait_time = 2 + random.uniform(0, 1)
                time.sleep(wait_time)
                continue

            raw = str(response.text).strip()
            try:
                int(raw)
            except (ValueError, TypeError):
                print(
                    f"Non-integer response on attempt {attempt + 1}/{max_retries}: '{raw}', retrying..."
                )
                wait_time = 2 + random.uniform(0, 1)
                time.sleep(wait_time)
                continue

            response_ = raw
            return response_  # success — exit immediately

        except Exception as err:
            error_name = type(err).__name__
            is_rate_limit = "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err)
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = 2 + random.uniform(0, 1)
                print(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                print(f"Error: {err} \nType: {error_name}")
                break  # non-429 error or max retries reached

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
PATH_OUT = f"results/lancet/gemini/{prompt_type}/"
PATH_OUT_ = Path(PATH_OUT)
PATH_OUT_.mkdir(parents=True, exist_ok=True)

# TEMPERATURES = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
# TEMPERATURES = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
TEMPERATURES = [2.0]

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

                    response_raw = query_LLM(prompt, TEMPERATURE, mode=TYPE_RESPONSE)
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
