# Capturing additional variance in impact of adolescent depression from free text in the Development and Well-Being Assessment (DAWBA)

[![Preregistration](https://img.shields.io/badge/Preregistration-OSF-blue)](https://osf.io/fs6bg)

Code for the preregistered study _Capturing additional variance in impact of adolescent depression from free text: proof of concept and preliminary results_. `DAWBA_Explore_Description.pdf` contains a preliminary write-up detailing content in the preregistration.

No participant data is included in this repository. Data from participants who enrolled in CAT-D are available at https://github.com/transatlantic-comppsych/CATD-study; data from participants who only screened are available upon request.

## Table of contents

- [Summary](#summary)
- [1 Data processing and setup](#1-data-processing-and-setup)
- [2 Text representations](#2-text-representations)
- [3 Exploratory modeling](#3-exploratory-modeling)
- [4 Confirmatory analyses](#4-confirmatory-analyses)
- [5 Running the code](#5-running-the-code)
- [6 MLstatkit fork](#6-mlstatkit-fork)

## Summary
Notebooks and scripts should be executed in the order outlined here.
### Step 1: data processing and setup

- `notebooks/01_clean_process.ipynb`: apply exclusion criteria, separate adolescent and parent response samples, create development (train) and held-out (test) sets. Additionally process development set (calls `src/clean.py`).

### Step 2: text representations

- `notebooks/02_embed.ipynb`: format text responses as question-answer, embed free text (calls `embed.py`, `embed_controller.py`).
- `llm_prompt_iter2/01-ReSructureDataset-Ext.ipynb`: restructure processed data into the input format for LLM prompting scripts.
- `notebooks/03_impactme_probs.ipynb`: generate outcome probabilities from logistic regression classifiers previously trained on IMPACT-ME data (calls `impactme_probs.py`).
- `notebooks/03A_0shot.ipynb`: iteration 1 of LLM prompting with Llama 3.3 70B Instruct for outcome presence, only uses zero-shot (calls `shot.py`, `shot_controller.py`).
- `llm_prompt_iter2/02-LANCET_Llama.V2.py`, `03-LANCET_Qwen.V2.py`, `05-LANCET_Gemini.py`: iteration 2 of LLM prompting with Llama 3.3 70B Instruct, Qwen 3.6 27B, and Gemini 2.5 Pro for outcome presence and severity, explores both zero- and few-shot.

### Step 3: exploratory modeling (development set)

- `notebooks/04_perm_folders.ipynb`: set up permutation directories (calls `perm.py`). Originally meant for exploratory permutation-based significance testing in development set, repurposed for non-parametric permutation-based significance testing and LMG decomposition, which are covered later.
- `notebooks/05_residual_preds.ipynb`: generate cross-validated residual stacking predictions across all configurations (calls `swarm.py`, `residual_preds.py`, `residual_wrapper.py`, `residual_func.py`, `Regressor.py`).
- `notebooks/06_model_compare.ipynb`: compare configurations and select frozen models (calls `compare_res.py`).
- `notebooks/07_coef_analysis.ipynb`: inspect learned coefficients, create graphs (calls `coef_visual.py`).
- `notebooks/08_full_model.ipynb`: fit frozen configurations on the full development set and extract coefficients, derive saliency maps (not included in manuscript) and LMG text feature contributions.

### Step 4: confirmatory analyses (held-out set)

- `notebooks/confirmatory/01_clean_process.ipynb`: apply Step 1 processing to the held-out set (calls `src/clean.py`). Additionally formats text into question-answer pairs.
- rerun `llm_prompt_iter2/` on the cleaned held-out set with frozen configurations
- `notebooks/confirmatory/02_format_LLM_output.ipynb`: format held-out LLM output from `llm_prompt_iter2/02-LANCET_Llama.V2.py`, `03-LANCET_Qwen.V2.py`, `05-LANCET_Gemini.py` into text vectors, also transform scale output from {0,1,2,3,4} to {0, 0.25, 0.5, 0.75, 1}.
- `notebooks/confirmatory/03_eval.ipynb`: evaluate frozen models and frozen model differences on the held-out set, bootstrap confidence intervals (calls `src/eval_test.py` and `src/sig.py`). The additional `add_new_st` flag should be set to True only for the post-hoc sensitivity analysis with the Qwen parent _ST_ model.
- `notebooks/confirmatory/04_sig_test.ipynb`: permutation tests for Hypotheses 1A/1B and 2A/2B (calls `src/sig.py`). A refers to tests within the adolescent-response sample, and B refers to that in the parent-response sample. Folders containing permutation random seeds are created by `notebooks/04_perm_folders.ipynb`.
- `notebooks/confirmatory/05_feat_cont.ipynb`: exploratory LMG feature contribution analysis for held-out set predictions and saliency maps (calls `src/sig.py`, `src/lmg_imp.py`, `src/residual_func.py`). Saliency maps are left out of the associated manuscript.
- `notebooks/confirmatory/06_summarize.ipynb`: create participant and free-text summarization tables (calls `src/clean.py`, `src/summarize.py`).

Shared helpers are in `src/utils.py`, `src/clean.py`, and `src/config.py`. Installation and tests are described in [5 Running the code](#5-running-the-code).

## 1 Data processing and setup

### 1.1 Cleaning and exclusions

This step is performed in `01_clean_process.ipynb`.

Responses to the depression section of the DAWBA were obtained for 347 paired adolescent-parent records. Twenty pairs were removed due to withdrawal of consent. Free-text responses were manually de-identified prior to inclusion.

Each adolescent is the subject of both a self-report and parent-report response set. These were separated into two response samples (adolescent and parent), and all subsequent processing and analyses were conducted separately within each. Exclusion criteria were applied to each response independently, so the final samples contain partially overlapping, rather than matched, sets of adolescents.

The following exclusions and imputations were applied:

- Symptom-scale items 18i-18l (self-harm and suicide ideation and attempt) were largely missing and were dropped.
- Because the DAWBA uses skip logic, responses from informants not prompted to answer free-text questions were removed.
- Symptom-scale items missing because the informant was not prompted were set to 0. Responses with any other missing symptom-scale items were removed.
- Responses with 6 or more missing free-text answers were removed. Missing answers in the remaining responses were filled with "Not answered".

### 1.2 Create development and held-out sets

The 327 paired records were split into development (n=219, ~67%) and held-out (n=108, ~33%) sets, stratified by CAT-D eligibility. Splitting occurred before separating adolescent and parent samples, so no adolescent appears in both splits. The held-out set was not accessed until confirmatory analyses after preregistration.

After cleaning, the development set contained 103 adolescent and 71 parent responses, and the held-out set contained 50 adolescent and 35 parent responses.

### 1.3 Features and target

- **Target:** impact score, the sum of the five impact items at the end of the depression section.
- **S:** sum of the 23 retained symptom-scale items (range 0-25).
- **Q:** 23-dimensional vector of item-level symptom-scale answers.
- **T:** 80-dimensional vector of text-derived outcome scores (Section 2).

## 2 Text representations

Two methods were explored for representing free text as a vector over a taxonomy of outcomes.

### 2.1 Embeddings and logistic regression

Performed in `02_embed.ipynb` and `03_impactme_probs.ipynb`. Text was embedded, and embeddings were passed to logistic regression classifiers previously trained on human-coded IMPACT-ME interview data (Goodyer et al., 2011; Midgley et al., 2014) for a taxonomy of psychological outcomes (Krause et al., 2021). Each classifier outputs the probability that a given outcome is present. This method was explored but not selected.

### 2.2 LLM prompting

The second iteration of prompting is in `llm_prompt_iter2/`. The "LANCET" scripts refer to the taxonomy of 80 depression outcomes from Chevance et al. (2020), published in _The Lancet Psychiatry_. It is referred to as _Expanded_ in the table below. The first iteration of prompting was performed in `03A_0shot.ipynb` for the IMPACT-ME taxonomy (referred to as _Original_ in the table), along with its combination with the Lancet taxonomy (referred to as _Combined_).

LLMs were prompted independently for each outcome and each response, and outputs were concatenated into a single vector. The following pseudo-hyperparameters were explored:

|                                  | **Logistic Regression** | **LLM Prompt iteration 1** | **LLM Prompt iteration 2**                                                               |
| -------------------------------- | ----------------------- | -------------------------- | ---------------------------------------------------------------------------------------- |
| **LLM model**                    | Llama 3.3 70B Instruct  | Llama 3.3 70B Instruct     | Llama 3.3 70B Instruct,<br>Gemini 2.5 Pro,<br>Qwen 3.6 27B                               |
| **Segmentation / prompt method** | Monologue / Dialogue    | zero-shot                  | zero-shot / few-shot                                                                     |
| **Temperature**                  | NA                      | 0.6                        | {0, 0.2, 0.4, 0.6, 0.8, 1.0} for Llama, Qwen;<br>{0, 0.4, 0.8, 1.2, 1.6, 2.0} for Gemini |
| **Aggregation / output method**  | Max / Avg / Block       | Binary / probability       | Binary / scale                                                                           |
| **Outcome taxonomy**             | Original                | Original / Combined        | Expanded                                                                                 |
| **Ridge penalty λ**              | 11-value grid†          | 11-value grid†             | 11-value grid†                                                                           |
| ***Total configs***              | **6 × 11 = 66**         | **4 × 11 = 44**            | **72 × 11 = 792**                                                                        |

† λ ∈ {0.1, 0.5, 0.75, 1, 2.5, 5, 15, 25, 50, 75, 100}

Llama and Qwen are open-weight models and were run on local GPUs. Gemini was run through HHS Enterprise Gemini with FedRAMP High authorization and cannot be reproduced the same way outside HHS. Outputs at temperatures above 0 are stochastic, so re-running extraction will not reproduce text vectors exactly. Note that raw scale output ({0,1,2,3,4}) is transformed into {0, 0.25, 0.5, 0.75, 1}.

## 3 Exploratory modeling

This step was performed on the NIH Biowulf computing cluster. Jobs were parallelized using the swarm command.

### 3.1 Residual stacking

Residual stacking is implemented in `residual_func.py` and the `ResidualRegressor` / `IndependentRegressor` classes in `Regressor.py`. Each layer is fit to the residuals of the layers before it, and the final prediction is the sum of all layer outputs. For `SQT`:

1. A linear regression is fit on S to predict impact score.
2. A ridge regression is fit on Q to predict the residuals of the S model.
3. A ridge regression is fit on T to predict the residuals of the SQ model.

This is visualized in the following table, where ε<sub>X</sub> denotes the residual after layer *X* and ε̂<sub>X</sub> denotes its prediction by the next layer.

|                        | **S-layer**                       | **Q-layer**                                    | **T-layer**                                       |
| ---------------------- | --------------------------------- | ---------------------------------------------- | ------------------------------------------------- |
| **Fit on**             | Sum-score *S*                     | Response vector *Q*                            | Text vector *T*                                   |
| **Predict**            | Impact score *y*                  | ε<sub>S</sub>                                  | ε<sub>SQ</sub>                                    |
| **Incremental output** | ŷ<sub>S</sub>                     | ε̂<sub>S</sub>                                  | ε̂<sub>SQ</sub>                                    |
| **Final output**       | ŷ<sub>S</sub>                     | ŷ<sub>SQ</sub> = ŷ<sub>S</sub> + ε̂<sub>S</sub> | ŷ<sub>SQT</sub> = ŷ<sub>SQ</sub> + ε̂<sub>SQ</sub> |
| **Residual**           | ε<sub>S</sub> = y − ŷ<sub>S</sub> | ε<sub>SQ</sub> = y − ŷ<sub>SQ</sub>            | ε<sub>SQT</sub> = y − ŷ<sub>SQT</sub>             |

Single-layer (S, Q, T), two-layer (SQ, ST, QT), and three-layer (SQT) setups were evaluated. S-layers are always linear regressions, and Q- and T-layers are always ridge regressions. Because shared variance is assigned to earlier layers, the difference in out-of-sample R² between a model with and without T reflects only the variance uniquely explained by T.

### 3.2 Configuration search

`05_residual_preds.ipynb` calls `residual_wrapper.py` to produce out-of-fold predictions using 10-time repeated 3-fold cross-validation, stratified by CAT-D eligibility, for each configuration. The number of regression fits is a product of:

- Number of response samples (adolescent, parent). Note that each fit is performed on a single training fold, containing 2/3 of the sample
- Number of residual setups (1: S, Q, T; 2: SQ, ST, QT; 3: SQT)
- Number of LLM pseudo-hyperparameter combinations (for setups including T)
- Number of ridge penalties (λ) for Q- and T-layers
- Number of folds (3)
- Number of CV repeats (10)

### 3.3 Model selection

Performed in `06_model_compare.ipynb` (calls `compare_res.py`). Out-of-fold R² was averaged over all 30 folds, and the best pseudo-hyperparameters and λ values were selected separately for each residual setup within each response sample. Selected configurations were frozen and preregistered, then fit on the entire development set in `08_full_model.ipynb`.

## 4 Confirmatory analyses

### 4.1 Held-out evaluation

Performed in `confirmatory/03_eval.ipynb`. Frozen models were evaluated on the held-out set. 95% confidence intervals for R² and paired ΔR² were derived via 1000-iteration bootstrapping using MLstatkit (see [6 MLstatkit fork](#6-mlstatkit-fork)). A post-hoc sensitivity analysis exploring _Qwen ST_ performance is also featured in `confirmatory/03_eval.ipynb`.

### 4.2 Significance testing

Performed in `confirmatory/04_sig_test.ipynb`.

**Hypotheses 1A/1B** test whether T adds variance explained over S, Q, and SQ (comparisons ST/S, QT/Q, SQT/SQ). Over 1000 iterations, T vectors in the development set were shuffled, stratified by CAT-D eligibility, while the target and reduced-model sources were held fixed. The same shuffle was used for all three full models within each iteration. Full models were refit on the shuffled development set and evaluated on the un-shuffled held-out set. Family-wise error rate was controlled with the Westfall-Young step-down minP procedure. The preregistered adaptation of the step-down maxT procedure is reported for completeness (see the manuscript for this deviation from preregistration). **A post-hoc sensitivity analysis including _Qwen ST_ in significance testing (including FWER correction) is also featured in `confirmatory/04_sig_test.ipynb`.**

**Hypotheses 2A/2B** test whether ST outperforms SQ. Held-out predictions of ST and SQ were randomly swapped over 1000 permutations, and uncorrected p-values were computed.

### 4.3 Feature contribution analysis

Performed in `confirmatory/05_feat_cont.ipynb` (calls `lmg_imp.py`). The R² contribution of T was apportioned to each of the 80 outcomes via LMG decomposition, averaged over 2000 random feature orderings. This was computed on the held-out set and, using out-of-fold predictions from the 10-time repeated 3-fold CV, on the development set in `notebooks/08_full_model.ipynb`. This analysis is exploratory.

### 4.4 Summary

Participant and free-text summary tables for the manuscript are produced in `confirmatory/06_summarize.ipynb`.

## 5 Running the code

### 5.1 Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The Python version is pinned in `.python-version` (3.12).

```bash
git clone https://github.com/NIMH-MLT/DAWBA.git
cd DAWBA
uv sync
```

`uv sync` installs all dependencies at the exact versions recorded in `uv.lock`, including the pinned MLstatkit fork (see [6 MLstatkit fork](#6-mlstatkit-fork)).

Running Llama 3.3 70B Instruct or Qwen 3.6 27B locally requires one or more GPUs with enough memory for the chosen model and precision.

### 5.2 Data

Input and output paths are set in `src/config.py`. 

### 5.3 Running on Biowulf

The configuration search, permutation tests, and LMG decomposition were run on the [NIH HPC Biowulf cluster](https://hpc.nih.gov). `src/swarm.py` writes [swarm](https://hpc.nih.gov/apps/swarm.html) files that parallelize these jobs.

On other systems, the same functions can be called serially.

### 5.4 Tests

Unit tests are in `tests/` and can be run with:

```bash
uv run pytest
```

## 6 MLstatkit fork

Bootstrap confidence intervals use a forked copy of [MLstatkit](https://github.com/AnthonyMatarr/MLstatkit), installed from GitHub. The forked copy adds additional metrics and paired bootstrapping.
