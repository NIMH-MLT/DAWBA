# summarize.py
import numpy as np
import pandas as pd


def get_count_col_entries(count_col_entries, col_name, df):
    res_df_rows = []
    for count_col in count_col_entries.keys():
        res_df_rows.append({"Feature": f"{count_col.upper()} n (%)", col_name: ""})
        ## Add missing entries
        for entry in count_col_entries[count_col]:
            if entry not in df[count_col].unique():
                res_df_rows.append(
                    {"Feature": f"{count_col.lower()}--{entry}", col_name: "0 (0)"}
                )
        ## Add percentages for each instance in the count column
        counts = df[count_col].value_counts(dropna=False)
        for instance, count in counts.items():
            res_df_rows.append(
                {
                    "Feature": f"{count_col.lower()}--{instance}",
                    col_name: f"{count} ({count/len(df):.1%})",
                }
            )
    return res_df_rows


def summarize_text(text_cols, df, col_name, return_lists=False):
    sub_df = df[text_cols]

    ## Num responded
    num_responded_list = (sub_df != "Not answered").sum(axis=1).to_list()
    avg_resp = np.mean(num_responded_list)
    std_resp = np.std(num_responded_list, ddof=1)

    ## Length
    sub_df = sub_df.mask(sub_df.eq("Not answered"))
    text_len_df = sub_df.apply(lambda x: x.str.split().str.len())
    row_lengths = text_len_df.sum(axis=1).dropna()
    assert np.isclose(avg_resp, text_len_df.notna().sum().sum() / len(sub_df))
    if return_lists:
        return row_lengths.to_list()

    avg_len = row_lengths.mean()
    std_len = row_lengths.std(ddof=1)
    q25, q50, q75 = row_lengths.quantile([0.25, 0.5, 0.75])
    res_list = [
        {
            "Feature": "Questions Responded To Avg (std)",
            col_name: f"{avg_resp:.2f} ({std_resp:.2f})",
        },
        {
            "Feature": "Total Words in entire response Avg (std) [25%, 50%, 75%]",
            col_name: f"{avg_len:.2f} ({std_len:.2f}) [ {q25:.2f}, {q50:.2f}, {q75:.2f} ]",
        },
    ]
    return res_list


def summarize_impact(df, col_name, impact_type):
    ## Summarize impact score
    impact_score_avg = df[f"impact_score_{impact_type}"].mean()
    impact_score_quartiles = df[f"impact_score_{impact_type}"].quantile(
        [0.25, 0.5, 0.75]
    )
    res_list = [
        {
            "Feature": "Impact Score Avg [25%, 50%, 75%]",
            col_name: f"{impact_score_avg:.2f} [{impact_score_quartiles[0.25]:.2f}, {impact_score_quartiles[0.5]:.2f}, {impact_score_quartiles[0.75]:.2f}]",
        },
    ]
    return res_list


def summarize_sum_score(df, quest_cols, col_name):
    quest_np = df[quest_cols].to_numpy(dtype=float)
    # Quest
    quest_sum = quest_np.sum(axis=1, keepdims=True)
    quest_sum_avg = np.mean(quest_sum)
    quest_sum_quartiles = np.quantile(quest_sum, [0.25, 0.5, 0.75])
    res_list = [
        {
            "Feature": "Quest Sum Score Avg [25%, 50%, 75%]",
            col_name: f"{quest_sum_avg:.2f} [{quest_sum_quartiles[0]:.2f}, {quest_sum_quartiles[1]:.2f}, {quest_sum_quartiles[2]:.2f}]",
        },
    ]
    return res_list


def get_col(df_merge, set_type, impact_type, count_col_entries, text_cols, quest_cols):
    """
    Params
    ------
    set_type: str
                    One of ['train', 'test']
    """
    col_name = f"{impact_type}_{set_type} (n={len(df_merge)})"
    ## Summarize counts
    res_df_rows = get_count_col_entries(count_col_entries, col_name, df=df_merge)
    ## Summarize quest sum-scores
    res_df_rows += summarize_sum_score(df_merge, quest_cols, col_name)

    ## Summarize text responses
    res_df_rows += summarize_text(text_cols, df_merge, col_name)
    ## Summarize impact score
    res_df_rows += summarize_impact(df_merge, col_name, impact_type)
    return pd.DataFrame(res_df_rows).set_index("Feature")
