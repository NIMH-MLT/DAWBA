# utils.py
import joblib
import pandas as pd
from pathlib import Path


def get_test_df_path(df_dir, llm_model, agg_type, desc_seg_type, impact_type, temp):
    """
    Helper returning path to test df given model configuration
    """
    if desc_seg_type not in ["shot_zero", "shot_few"]:
        raise ValueError(
            f"Unrecognized desc_seg_type: {desc_seg_type}, must be one of ['shot_zero', 'shot_few']"
        )
    # remove -
    agg_type_clean = agg_type.split("-")[0]
    return (
        Path(df_dir)
        / llm_model
        / f"temp_{temp}"
        / desc_seg_type
        / agg_type_clean
        / f"{impact_type}.parquet"
    )


def get_df_path(df_dir, llm_model, agg_type, desc_seg_type, impact_type, temp):
    """
    Helper returning path to train df given model configuration
    """
    if desc_seg_type in ["shot_zero", "shot_few"]:
        # remove -
        agg_type_clean = agg_type.split("-")[0]
        return (
            Path(df_dir)
            / "final_cleaned"
            / llm_model
            / f"temp_{temp}"
            / desc_seg_type
            / agg_type_clean
            / f"{impact_type}.parquet"
        )
    elif agg_type in ["avg", "max", "block"]:
        intermediate_dir = "w_probs"
    elif agg_type in ["binary-og", "proba-limited-og"]:
        agg_type_clean = "-".join(agg_type.split("-")[:-1])
        intermediate_dir = f"w_0_shot_probs/temp_0.6/{agg_type_clean}"
    else:
        raise ValueError(f"Unreccognized agg type: {agg_type}")
    return (
        Path(df_dir)
        / intermediate_dir
        / llm_model
        / desc_seg_type
        / f"{impact_type}.parquet"
    )


def get_prob_col_outcomes(agg_type, desc_seg_type):
    if desc_seg_type in ["shot_zero", "shot_few"]:
        return "", ADVANCED_EXTRACTION_OUTCOMES
    elif agg_type in ["avg", "max", "block"]:
        return f"prob_vector_{agg_type}", IMPACTME_OUTCOMES_LR
    elif agg_type in ["proba-limited-og", "binary-og"]:
        prob_col = "open_text_block_0_shot"
        # if desc_seg_type == "new":
        #     prob_outcomes = IMPACTME_OUTCOMES_FULL
        if desc_seg_type == "expanded":
            prob_outcomes = OUTCOMES_EXPAND
        elif desc_seg_type in ["original"]:  # "augment_neg", "augment"]:
            prob_outcomes = IMPACTME_OUTCOMES_SHOT
        elif desc_seg_type == "combined":  # combines expanded + original
            prob_outcomes = [f"{col}_impactme" for col in IMPACTME_OUTCOMES_SHOT] + [
                f"{col}_expand" for col in OUTCOMES_EXPAND
            ]
        else:
            raise ValueError(f"Unrecognized desc_seg_type: {desc_seg_type}")
    else:
        raise ValueError(
            f"Unrecognized agg_type: {agg_type}--- DESC SEG: {desc_seg_type}"
        )
    return prob_col, prob_outcomes


def expand_text(df, agg_type, desc_seg_type):
    prob_col, prob_outcomes = get_prob_col_outcomes(
        agg_type=agg_type, desc_seg_type=desc_seg_type
    )
    if desc_seg_type in ["shot_zero", "shot_few"]:
        return df, prob_outcomes
    else:
        new_cols = pd.DataFrame(
            df[prob_col].tolist(), columns=prob_outcomes, index=df.index
        )
        df = df.join(new_cols)
        df = df.drop(columns=prob_col)

        return df, prob_outcomes


def get_perm_seed(perm_dir):
    perm_seed_path = perm_dir / f"seed.txt"
    if not perm_seed_path.exists():
        raise ValueError(f"No seed file exists at path {perm_seed_path}. Aborting...")
    with open(perm_seed_path, "r") as f:
        perm_seed = int(f.read().strip())
    return perm_seed


def export_data(data_to_export, export_path, remove_existing=False):
    export_path = Path(export_path)
    if export_path.exists():
        if remove_existing:
            print(f"REMOVING file at path: {export_path}")
            export_path.unlink()
        else:
            print(f"File already exists at path: {export_path}")
            print(f"ABORTING")
            raise ValueError(
                "`remove_existing` set to False and file already exists at path"
            )
    export_path.parent.mkdir(exist_ok=True, parents=True)
    if export_path.suffix == ".parquet":
        data_to_export.to_parquet(export_path)
    elif export_path.suffix == ".csv":
        data_to_export.to_csv(export_path)
    elif export_path.suffix == ".tsv":
        data_to_export.to_csv(export_path, sep="\t")
    elif export_path.suffix == ".joblib":
        joblib.dump(data_to_export, export_path)
    else:
        raise ValueError(f"Unrecognized export type: {export_path.suffix}")


## Column tyoes
col_config_og = {
    "self_impact": [
        "s_dep_19_distress",
        "s_dep_20a_impact_family",
        "s_dep_20b_impact_friendships",
        "s_dep_20c_impact_school_work",
        "s_dep_20d_impact_leisure",
    ],
    "parent_impact": [
        "p_dep_19_distress",
        "p_dep_20a_impact_family",
        "p_dep_20b_impact_friendships",
        "p_dep_20c_impact_school_work",
        "p_dep_20d_impact_leisure",
    ],
    "self_text": [
        "s_dep_25a_text_mood_anhedonia",
        "s_dep_25b_text_else_changed",
        "s_dep_25c_text_duration",
        "s_dep_25d_text_severity",
        "s_dep_25e_text_ep_begin",
        "s_dep_25f_text_ep_trigger",
        "s_dep_25g_text_ep_past",
        "s_dep_25h_text_ep_high",
        "s_dep_25i_text_quality_life",
        "s_dep_25j_text_tried_fix",
        # 's_dep_26_describe_self_harm', # not included in depr section?
    ],
    "parent_text": [
        "p_dep_25a_text_mood_anhedonia",
        "p_dep_25b_text_else_changed",
        "p_dep_25c_text_duration",
        "p_dep_25d_text_severity",
        "p_dep_25e_text_ep_begin",
        "p_dep_25f_text_ep_trigger",
        "p_dep_25g_text_ep_past",
        "p_dep_25h_text_ep_high",
        "p_dep_25i_text_quality_life",
        "p_dep_25j_text_tried_fix",
        # 'p_dep_26_describe_self_harm', # not included in depr section?
    ],
    "self_quest": [
        "s_dep_1_sad",
        "s_dep_2_miserable_daily",
        "s_dep_3_miserable_mostly",
        "s_dep_4_miserable_cheered_up",
        "s_dep_5_miserable_duration",
        "s_dep_7_irritable",
        "s_dep_8_irritable_daily",
        "s_dep_9_irratable_mostly",
        "s_dep_10_irritable_improved",
        "s_dep_11_irritable_duration",
        "s_dep_13_anhedonia",
        "s_dep_14_anhedonia_daily",
        "s_dep_15_anhedonia_mostly",
        "s_dep_16_anhedonia_duration",
        "s_dep_17_coincided_mis_irrit",
        "s_dep_18a_tired",
        "s_dep_18b_appetite",
        "s_dep_18c_weight",
        "s_dep_18d_insomnia",
        "s_dep_18e_hypersomnia",
        "s_dep_18f_agitation",
        "s_dep_18g_worthless",
        "s_dep_18h_concentration",
        "s_dep_18i_thoughts_death",
        "s_dep_18j_self_harm_talk_recent",
        "s_dep_18k_self_harm_act_recent",
        "s_dep_18l_self_harm_act_ever",
    ],
    "parent_quest": [
        "p_dep_1_sad",
        "p_dep_2_miserable_daily",
        "p_dep_3_miserable_mostly",
        "p_dep_4_miserable_cheered_up",
        "p_dep_5_miserable_duration",
        "p_dep_7_irritable",
        "p_dep_8_irritable_daily",
        "p_dep_9_irratable_mostly",
        "p_dep_10_irritable_improved",
        "p_dep_11_irritable_duration",
        "p_dep_13_anhedonia",
        "p_dep_14_anhedonia_daily",
        "p_dep_15_anhedonia_mostly",
        "p_dep_16_anhedonia_duration",
        "p_dep_17_coincided_mis_irrit",
        "p_dep_18a_tired",
        "p_dep_18b_appetite",
        "p_dep_18c_weight",
        "p_dep_18d_insomnia",
        "p_dep_18e_hypersomnia",
        "p_dep_18f_agitation",
        "p_dep_18g_worthless",
        "p_dep_18h_concentration",
        "p_dep_18i_thoughts_death",
        "p_dep_18j_self_harm_talk_recent",
        "p_dep_18k_self_harm_act_recent",
        "p_dep_18l_self_harm_act_ever",
    ],
    "meta": ["DAWBA_ID", "eligible_group", "Participant_Type2"],
}


col_config_sensitivity = {
    "self_impact": [
        "s_dep_19_distress",
        "s_dep_20a_impact_family",
        "s_dep_20b_impact_friendships",
        "s_dep_20c_impact_school_work",
        "s_dep_20d_impact_leisure",
    ],
    "parent_impact": [
        "p_dep_19_distress",
        "p_dep_20a_impact_family",
        "p_dep_20b_impact_friendships",
        "p_dep_20c_impact_school_work",
        "p_dep_20d_impact_leisure",
    ],
    "self_text": [
        "s_dep_25a_text_mood_anhedonia",
        "s_dep_25b_text_else_changed",
        "s_dep_25c_text_duration",
        "s_dep_25d_text_severity",
        "s_dep_25e_text_ep_begin",
        "s_dep_25f_text_ep_trigger",
        "s_dep_25g_text_ep_past",
        "s_dep_25h_text_ep_high",
        "s_dep_25i_text_quality_life",
        "s_dep_25j_text_tried_fix",
        # 's_dep_26_describe_self_harm', # not included in depr section?
    ],
    "parent_text": [
        "p_dep_25a_text_mood_anhedonia",
        "p_dep_25b_text_else_changed",
        "p_dep_25c_text_duration",
        "p_dep_25d_text_severity",
        "p_dep_25e_text_ep_begin",
        "p_dep_25f_text_ep_trigger",
        "p_dep_25g_text_ep_past",
        "p_dep_25h_text_ep_high",
        "p_dep_25i_text_quality_life",
        "p_dep_25j_text_tried_fix",
        # 'p_dep_26_describe_self_harm', # not included in depr section?
    ],
    "self_quest": [
        "s_dep_1_sad",
        "s_dep_2_miserable_daily",
        "s_dep_3_miserable_mostly",
        "s_dep_4_miserable_cheered_up",
        "s_dep_5_miserable_duration",
        "s_dep_7_irritable",
        "s_dep_8_irritable_daily",
        "s_dep_9_irratable_mostly",
        "s_dep_10_irritable_improved",
        "s_dep_11_irritable_duration",
        "s_dep_13_anhedonia",
        "s_dep_14_anhedonia_daily",
        "s_dep_15_anhedonia_mostly",
        "s_dep_16_anhedonia_duration",
        "s_dep_17_coincided_mis_irrit",
        "s_dep_18a_tired",
        "s_dep_18b_appetite",
        "s_dep_18c_weight",
        "s_dep_18d_insomnia",
        "s_dep_18e_hypersomnia",
        "s_dep_18f_agitation",
        "s_dep_18g_worthless",
        "s_dep_18h_concentration",
        ## Removed bc missing for >40
        # "s_dep_18i_thoughts_death",
        # "s_dep_18j_self_harm_talk_recent",
        # "s_dep_18k_self_harm_act_recent",
        # "s_dep_18l_self_harm_act_ever",
    ],
    "parent_quest": [
        "p_dep_1_sad",
        "p_dep_2_miserable_daily",
        "p_dep_3_miserable_mostly",
        "p_dep_4_miserable_cheered_up",
        "p_dep_5_miserable_duration",
        "p_dep_7_irritable",
        "p_dep_8_irritable_daily",
        "p_dep_9_irratable_mostly",
        "p_dep_10_irritable_improved",
        "p_dep_11_irritable_duration",
        "p_dep_13_anhedonia",
        "p_dep_14_anhedonia_daily",
        "p_dep_15_anhedonia_mostly",
        "p_dep_16_anhedonia_duration",
        "p_dep_17_coincided_mis_irrit",
        "p_dep_18a_tired",
        "p_dep_18b_appetite",
        "p_dep_18c_weight",
        "p_dep_18d_insomnia",
        "p_dep_18e_hypersomnia",
        "p_dep_18f_agitation",
        "p_dep_18g_worthless",
        "p_dep_18h_concentration",
        # Removed bc missing for >40
        # "p_dep_18i_thoughts_death",
        # "p_dep_18j_self_harm_talk_recent",
        # "p_dep_18k_self_harm_act_recent",
        # "p_dep_18l_self_harm_act_ever",
    ],
    "meta": ["DAWBA_ID", "eligible_group"],
}


ALPHA_LIST = [
    0.1,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    15.0,
    25.0,
    50.0,
    75.0,
    100.0,
]

# all impactme outcomes + 2 new (f4,f5) (used for "new" 0-shot)
IMPACTME_OUTCOMES_FULL = [
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "a8",  # --> NOT INCLDUED in LR
    "b1",
    "b2",
    "b3",
    "c1",
    "c2",
    "c3",
    "c4",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "e1",
    "e2",
    "e3",
    "e4",  # --> NOT INCLDUED in LR
    "e5",  # --> NOT INCLDUED in LR
    "f1",
    "f2",
    "f3",
    "f4",  # --> NOT INCLDUED in SHOT (new)
    "f5",  # --> NOT INCLDUED in SHOT (new)
    "g1",
    "g2",
]

# all impactme outcomes (used for "original" 0-shot)
IMPACTME_OUTCOMES_SHOT = [
    outcome for outcome in IMPACTME_OUTCOMES_FULL if outcome not in ["f4", "f5"]
]

# all impactme outcomes used in og study (used for max, block, avg)
IMPACTME_OUTCOMES_LR = [
    outcome for outcome in IMPACTME_OUTCOMES_SHOT if outcome not in ["a8", "e4", "e5"]
]

# new taxonomy of outcomes (used for "expanded" 0-shot)
OUTCOMES_EXPAND = [
    "A1",
    "A2",
    "A3",
    "A4",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "B11",
    "B12",
    "B13",
    "B14",
    "B15",
    "B16",
    "B17",
    "B18",
    "B19",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",
    "E9",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "H1",
    "H2",
    "H3",
    "I1",
    "I2",
    "I3",
]

OUTCOMES_COMBINED = IMPACTME_OUTCOMES_SHOT + OUTCOMES_EXPAND

# NEWEST taxonomy of outcomes (used for zero_shot, few_shot w/ qwen + llama)
ADVANCED_EXTRACTION_OUTCOMES = [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S19",
    "S20",
    "S21",
    "S22",
    "S23",
    "S24",
    "S25",
    "S26",
    "S27",
    "S28",
    "S29",
    "S30",
    "S31",
    "S32",
    "S33",
    "S34",
    "S35",
    "S36",
    "S37",
    "S38",
    "S39",
    "S40",
    "S41",
    "S42",
    "S43",
    "S44",
    "S45",
    "S46",
    "S47",
    "S48",
    "S49",
    "S50",
    "S51",
    "S52",
    "S53",
    "S54",
    "S55",
    "S56",
    "S57",
    "S58",
    "S59",
    "S60",
    "S61",
    "S62",
    "S63",
    "S64",
    "S65",
    "S66",
    "S67",
    "S68",
    "S69",
    "S70",
    "S71",
    "S72",
    "S73",
    "S74",
    "S75",
    "S76",
    "S77",
    "S78",
    "S79",
    "S80",
]


BEST_PARAM_DICT = {
    "SQT": {
        "parent": {
            "agg_type": "scale-new",
            "llm_model": "Qwen3.6-27B",
            "desc_seg_type": "shot_zero",
            "alpha2": 15.0,
            "alpha3": 75.0,
            "temp": 0.0,
            "num_layers": 3,
            "two_layer_base": None,
        },
        "self": {
            "agg_type": "scale-new",
            "llm_model": "gemini-2.5-pro",
            "desc_seg_type": "shot_few",
            "alpha2": 50.0,
            "alpha3": 15.0,
            "temp": 0.4,
            "num_layers": 3,
            "two_layer_base": None,
        },
    },
    "ST": {
        "parent": {
            "agg_type": "scale-new",
            "llm_model": "Llama-3.3-70B-Instruct",
            "desc_seg_type": "shot_few",
            "alpha2": None,
            "alpha3": 15.0,
            "temp": 0.0,
            "num_layers": 2,
            "two_layer_base": "S",
        },
        "self": {
            "agg_type": "scale-new",
            "llm_model": "gemini-2.5-pro",
            "desc_seg_type": "shot_few",
            "alpha2": None,
            "alpha3": 25.0,
            "temp": 0.4,
            "num_layers": 2,
            "two_layer_base": "S",
        },
    },
    "QT": {
        "parent": {
            "agg_type": "scale-new",
            "llm_model": "Qwen3.6-27B",
            "desc_seg_type": "shot_zero",
            "alpha2": 5.0,
            "alpha3": 50.0,
            "temp": 0.0,
            "num_layers": 2,
            "two_layer_base": "Q",
        },
        "self": {
            "agg_type": "scale-new",
            "llm_model": "gemini-2.5-pro",
            "desc_seg_type": "shot_few",
            "alpha2": 15.0,
            "alpha3": 5.0,
            "temp": 0.4,
            "num_layers": 2,
            "two_layer_base": "Q",
        },
    },
    "T": {
        "parent": {
            "agg_type": "scale-new",
            "llm_model": "Llama-3.3-70B-Instruct",
            "desc_seg_type": "shot_few",
            "alpha2": None,
            "alpha3": 5.0,
            "temp": 1.0,
            "num_layers": 1,
        },
        "self": {
            "agg_type": "scale-new",
            "llm_model": "gemini-2.5-pro",
            "desc_seg_type": "shot_few",
            "alpha2": None,
            "alpha3": 5.0,
            "temp": 0.4,
            "num_layers": 1,
        },
    },
}
