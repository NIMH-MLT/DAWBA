# clean.py


def create_checkpoint_cols(df, impact_type, impact_cols):
    """
    Create checkpoint columns (Depression, Irritability, Loss of Interest);
    used downstream to determine what questionnaires are prompted. Also create
    target column as sum of impact_cols.

    Params
    ------
    df: pandas DF
        Dataframe containing raw data for participant responses to the DAWBA
    impact_type: str
        One of {self, parent}
    impact_cols: List[str]
        Name of columns relating to impact questions in DAWBA
        Their sum is the target impact score

    Returns
    -------
    df: pandas DF
        Input dataframe with the three additional checkpoint columns

    """
    if impact_type not in ["self", "parent"]:
        raise ValueError(
            f"impact_type must be one of ['self', 'parent'], got {impact_type}"
        )
    prefix = "s" if impact_type == "self" else "p"
    ## Impact score
    df[f"impact_score_{impact_type}"] = df[impact_cols].sum(axis=1)
    # Depr --> H1, H2, AND H3
    df[f"Depression_Check_{impact_type}"] = (
        (df[f"{prefix}_dep_1_sad"] == 1)
        & (df[f"{prefix}_dep_2_miserable_daily"] == 1)
        & (df[f"{prefix}_dep_3_miserable_mostly"] == 1)
    )
    # Irr --> H7, H8, AND H9
    df[f"Irritable_Check_{impact_type}"] = (
        (df[f"{prefix}_dep_7_irritable"] == 1)
        & (df[f"{prefix}_dep_8_irritable_daily"] == 1)
        & (df[f"{prefix}_dep_9_irratable_mostly"] == 1)
    )
    # Loss of Interest --> H13 AND H14
    df[f"Loss_Interest_Check_{impact_type}"] = (
        df[f"{prefix}_dep_13_anhedonia"] == 1
    ) & (df[f"{prefix}_dep_14_anhedonia_daily"] == 1)
    return df


def mark_NA_0_when_not_prompted_helper(df, skipped_cols, prefix, mask_col):
    """
    Helper for mark_NA_0_when_not_prompted; creates mask given some column
    and marks NAs in skipped_cols as 0 for participants whose mask_col is 0

    Params
    ------
    df: pandas DF
        Dataframe containing raw data for participant responses to the DAWBA
    skipped_cols: List[str]
        Columns pertaining to questions in the DAWBA that are skipped when mask_col is 0 (no)
    prefix: str
        One of {'s', 'p'}, pertaining to self and parent
    mask_col: str
        Column pertaining to question in the DAWBA that determines whether skipped_cols are prompted

    Returns
    -------
    df: pandas DF
        Input dataframe whose skipped cols have been adjusted according to mask_col
    """
    skipped_cols_w_prefix = [f"{prefix}_{col}" for col in skipped_cols]
    mask = df[f"{prefix}_{mask_col}"] == 0

    df.loc[mask, skipped_cols_w_prefix] = df.loc[mask, skipped_cols_w_prefix].fillna(0)
    return df


def mark_NA_0_when_not_prompted(df, impact_type, impact_cols):
    """
    Fill NAs w/ 0 when DAWBA did not prompt a participant to answer, based on
    states of [Miserable, Irritable, Anhedonia].

    Params
    ------
    df: pandas DF
        Dataframe containing raw data for participant responses to the DAWBA
    impact_type: str
        One of {self, parent}
    impact_cols: List[str]
        Name of columns relating to impact questions in DAWBA
        Their sum is the target impact score

    Returns
    -------
    df: pandas DF
        Input dataframe w/ NAs marked as 0 where appropriate
    """
    if impact_type not in ["self", "parent"]:
        raise ValueError(
            f"impact_type must be one of ['self', 'parent'], got {impact_type}"
        )
    prefix = "s" if impact_type == "self" else "p"

    ## ====> Miserable (2-5); For 2-5, change NAs to 0s when H1 is 0
    skipped_cols1 = [
        "dep_2_miserable_daily",
        "dep_3_miserable_mostly",
        "dep_4_miserable_cheered_up",
        "dep_5_miserable_duration",
    ]
    df = mark_NA_0_when_not_prompted_helper(
        df=df, skipped_cols=skipped_cols1, prefix=prefix, mask_col="dep_1_sad"
    )

    ## ====> Irritiable (8-11); For 8-11, change NAs to 0s when H7 is 0
    skipped_cols2 = [
        "dep_8_irritable_daily",
        "dep_9_irratable_mostly",
        "dep_10_irritable_improved",
        "dep_11_irritable_duration",
    ]
    df = mark_NA_0_when_not_prompted_helper(
        df=df, skipped_cols=skipped_cols2, prefix=prefix, mask_col="dep_7_irritable"
    )

    ## ====> anhedonia (14-16); for 14-16, change NAs to 0s when H13 is 0
    skipped_cols3 = [
        "dep_14_anhedonia_daily",
        "dep_15_anhedonia_mostly",
        "dep_16_anhedonia_duration",
    ]
    df = mark_NA_0_when_not_prompted_helper(
        df=df, skipped_cols=skipped_cols3, prefix=prefix, mask_col="dep_13_anhedonia"
    )

    ## ====> H17; Answer only if H13 AND (Depr OR Irr); Change NA to 0 if conditition not met
    mask = (df[f"{prefix}_dep_13_anhedonia"] == 0) | (
        (df[f"Depression_Check_{impact_type}"] == 0)
        & (df[f"Irritable_Check_{impact_type}"] == 0)
    )
    df.loc[mask, f"{prefix}_dep_17_coincided_mis_irrit"] = df.loc[
        mask, f"{prefix}_dep_17_coincided_mis_irrit"
    ].fillna(0)

    ## ====> Depr + Irr + LoI (18A-18L, 19, 20A-20D, 21)
    # Only need 1 to be True to answer
    mask = (
        (df[f"Loss_Interest_Check_{impact_type}"] == 0)
        & (df[f"Depression_Check_{impact_type}"] == 0)
        & (df[f"Irritable_Check_{impact_type}"] == 0)
    )
    skipped_at_18 = [
        f"{prefix}_dep_18a_tired",
        f"{prefix}_dep_18b_appetite",
        f"{prefix}_dep_18c_weight",
        f"{prefix}_dep_18d_insomnia",
        f"{prefix}_dep_18e_hypersomnia",
        f"{prefix}_dep_18f_agitation",
        f"{prefix}_dep_18g_worthless",
        f"{prefix}_dep_18h_concentration",
        f"{prefix}_dep_18i_thoughts_death",
        f"{prefix}_dep_18j_self_harm_talk_recent",
        f"{prefix}_dep_18k_self_harm_act_recent",
        f"{prefix}_dep_18l_self_harm_act_ever",
    ] + impact_cols

    # Fill questions with 0
    df.loc[mask, skipped_at_18] = df.loc[mask, skipped_at_18].fillna(0)
    return df


def remove_0_impact_participants(df, impact_type, impact_cols):
    """
    Some participants will have been healthy (w/o any of the checkpoints),
    and thus not prompted to respond to text open-questions. Remove these
    participants.
    """
    df[f"impact_score_{impact_type}"] = df[impact_cols].sum(axis=1)
    mask = (
        (df[f"Depression_Check_{impact_type}"] == 0)
        & (df[f"Irritable_Check_{impact_type}"] == 0)
        & (df[f"Loss_Interest_Check_{impact_type}"] == 0)
    )
    ## remove participants without text
    return df[~mask].reset_index(drop=True)


def remove_remaining_missing(df, impact_type, quest_cols, text_cols):
    """
    Removes 18i-18l due to large-scale missingness. If a participant has 5 or less (out of 10)
    missing text responses, fill those responses with "Not answered". For all other participants
    with missing entries in either questionnaire or text (6+ missing), remove the entire participant.

    Params
    ------
    df: pandas DF
        Dataframe containing raw data for participant responses to the DAWBA
    impact_type: str
        One of {self, parent}
    quest_cols: List[str]
        Columns in df pertaining to standardized questionnaire prompts in DAWBA
    text_cols: List[str]
        Columns in df pertaining to open-text prompts in DAWBA

    Returns
    -------
    df: padnas DF
        Input DF with no NAs, filled or dropped as appropriate
    """
    if impact_type not in ["self", "parent"]:
        raise ValueError(
            f"impact_type must be one of ['self', 'parent'], got {impact_type}"
        )
    prefix = "s" if impact_type == "self" else "p"

    # Remove death/self-harm questions bc of missingness
    drop_cols = [
        f"{prefix}_dep_18i_thoughts_death",
        f"{prefix}_dep_18j_self_harm_talk_recent",
        f"{prefix}_dep_18k_self_harm_act_recent",
        f"{prefix}_dep_18l_self_harm_act_ever",
    ]
    print(f"Dropping high-missing columns: \n{df[drop_cols].isna().sum()}")
    df_sub = df.drop(drop_cols, axis=1)

    quest_cols_sub = [col for col in quest_cols if col not in drop_cols]
    # Remove NAs resulting from true missingness rather than structure
    ## From questionnaire cols
    og_len = len(df_sub)
    df_sub = df_sub.dropna(subset=quest_cols_sub)
    new_len = len(df_sub)
    print(f"Total dropped bc questionnaire NAs: {og_len - new_len}")
    ## From text cols
    og_len = len(df_sub)
    df_sub = df_sub.dropna(subset=text_cols, thresh=5)
    new_len = len(df_sub)
    print(f"Total dropped bc text NAs: {og_len - new_len}")

    # Fill remaining text NAs w/ "Not answered"
    print(f"Remaining participants w/ text NAs: \n {df_sub[text_cols].isna().sum()}")
    df_sub[text_cols] = df_sub[text_cols].fillna("Not answered")

    assert df_sub.isna().sum().sum() == 0
    return df_sub.reset_index(drop=True)
