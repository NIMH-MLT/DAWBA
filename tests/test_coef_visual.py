# test_coef_visual.py
# Run w/ uv run pytest tests/test_coef_visual.py -v
# Written in collaboration w/ Claude Opus 5


import pytest
import pandas as pd
from src.coef_visual import get_pivot
import numpy as np


class TestGetPivot:
    @pytest.fixture
    def coef_df(self):
        """Coefficient rows as aggregate_and_export writes them: one per feature
        per fold per repeat, tagged with the full config."""
        rows = []
        for repeat in range(2):
            for split_id in range(2):
                for stage, feature in [
                    ("stage1", "sum_score"),
                    ("stage3", "t1"),
                ]:
                    rows.append(
                        {
                            "repeat": repeat,
                            "split_id": split_id,
                            "stage": stage,
                            "feature": feature,
                            "coef": 0.1 * repeat + 0.01 * split_id,
                            "agg_type": "scale-new",
                            "llm_model": "gemini-2.5-pro",
                            "desc_seg": "shot_few",
                            "alpha": "15.0_5.0",
                            "num_layers": 3,
                            "temp": 0.4,
                            "two_layer_base": "None",
                        }
                    )
        return pd.DataFrame(rows)

    @pytest.fixture
    def config(self):
        return dict(
            agg_type="scale-new",
            llm_model="gemini-2.5-pro",
            desc_seg="shot_few",
            alpha="15.0_5.0",
            num_layers=3,
            temp=0.4,
            two_layer_base="None",
        )

    def test_one_column_per_repeat_fold_one_row_per_feature(self, coef_df, config):
        pivot = get_pivot(coef_df, **config)
        assert list(pivot.index) == ["stage1_sum_score", "stage3_t1"]
        assert list(pivot.columns) == ["0_0", "0_1", "1_0", "1_1"]
        assert pivot.loc["stage3_t1", "1_1"] == pytest.approx(0.11)

    def test_underspecified_config_raises_on_duplicates(self, coef_df, config):
        """The pivot has no aggfunc, so if the filter leaves more than one row per
        (feature, repeat_split_id) it raises rather than silently averaging. Here
        two alphas survive because the frame carries a config the filter doesn't
        distinguish."""
        other = coef_df.copy()
        other["alpha"] = "5.0_15.0"
        combined = pd.concat([coef_df, other], ignore_index=True)
        combined["alpha"] = "15.0_5.0"  # collapse the distinguishing column
        with pytest.raises(ValueError, match="duplicate"):
            get_pivot(combined, **config)

    def test_nonmatching_config_raises(self, coef_df, config):
        with pytest.raises(
            ValueError, match="No data matches the specified configuration."
        ):
            get_pivot(coef_df, **{**config, "alpha": "does_not_exist"})

    def test_two_layer_base_none_placeholder_matches(self, coef_df, config):
        """aggregate_and_export writes the string "None"; the fillna makes a NaN
        in the frame match it."""
        coef_df["two_layer_base"] = np.nan
        assert not get_pivot(coef_df, **config).empty
