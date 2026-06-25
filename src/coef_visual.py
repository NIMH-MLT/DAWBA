import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def get_pivot(
    df,
    agg_type,
    llama,
    desc_seg,
    alpha,
    num_layers,
):
    df = df.fillna("None")
    df["repeat_split_id"] = df["repeat"].astype(str) + "_" + df["split_id"].astype(str)
    sub_df = df[
        (df["agg_type"] == agg_type)
        & (df["llama_model"] == llama)
        & (df["desc_seg"] == desc_seg)
        & (df["alpha"] == alpha)
        & (df["num_layers"] == num_layers)
    ][["repeat_split_id", "stage", "feature", "coef"]]
    sub_df = sub_df.reset_index(drop=True)
    sub_df["stage_feat"] = sub_df["stage"] + "_" + sub_df["feature"]  # y-axis
    # Pivot: rows = stage_feat, columns = repeat, values = avg_coef
    pivot = sub_df.pivot(index="stage_feat", columns="repeat_split_id", values="coef")
    return pivot


def plot_heatmap(
    pivot, title="avg_val by Repeat and Feature", vmin=-3, vmax=3, xlabel="Repeat"
):
    ## PLOT
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_green",
        ["#08306b", "white", "#00441b"],  # dark blue, white, dark green
    )
    abs_max = pivot.abs().max().max()
    fig, ax = plt.subplots(
        figsize=(len(pivot.columns) + 10, max(12, len(pivot.index) * 0.6 + 2)),
        constrained_layout=True,
    )
    sns.heatmap(
        pivot,
        annot=True,  # show values in cells
        fmt=".2f",  # format to 2 decimal places
        cmap=cmap,
        center=0,
        # vmin=-abs_max,
        # vmax=abs_max,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    plt.show()
