# test_gen_representations.py
# Run w/ uv run pytest tests/test_gen_representations.py -v
# Written in collaboration w/ Claude Opus 5

import pytest
import pandas as pd
import numpy as np

from src.embed import concat_question_answer, import_df
import src.shot as shot
from src.shot import combine_original_expanded


# Data fixtures
@pytest.fixture
def answer_df():
    return pd.DataFrame(
        {
            "DAWBA_ID": [1, 2, 3],
            "impact_score": [4, 6, 8],
            "q1": ["worry", "cannot sleep", "sad most days"],
            "q2": ["school is hard here", "no", "friends are fine"],
        }
    )


@pytest.fixture
def question_df():
    return pd.DataFrame(
        {
            "question_col": ["q1", "q2"],
            "question_text": ["How do you feel?", "What about school?"],
        }
    )


@pytest.fixture
def text_cols():
    return ["q1", "q2"]


@pytest.fixture
def input_path(answer_df, tmp_path):
    path = tmp_path / "self.parquet"
    answer_df.to_parquet(path)
    return path


# CONCAT
class TestConcatQuestionAnswer:
    def test_prefixes_each_question_column(self, answer_df, question_df):
        out = concat_question_answer(answer_df, question_df)
        assert out.loc[0, "q1"] == "Question: How do you feel? Answer: worry"
        assert out.loc[1, "q1"] == "Question: How do you feel? Answer: cannot sleep"
        assert out.loc[2, "q1"] == "Question: How do you feel? Answer: sad most days"

        assert (
            out.loc[0, "q2"]
            == "Question: What about school? Answer: school is hard here"
        )
        assert out.loc[1, "q2"] == "Question: What about school? Answer: no"
        assert (
            out.loc[2, "q2"] == "Question: What about school? Answer: friends are fine"
        )

    def test_creates_open_text_block(self, answer_df, question_df):
        out = concat_question_answer(answer_df, question_df)
        assert "open_text_block" in out.columns

    def test_open_text_block_joins_with_newline(self, answer_df, question_df):
        out = concat_question_answer(answer_df, question_df)
        expected = (
            "Question: How do you feel? Answer: worry\n"
            "Question: What about school? Answer: school is hard here"
        )
        assert out.loc[0, "open_text_block"] == expected

    def test_open_text_block_order_follows_question_df(self, answer_df, question_df):
        """Row order in question_df determines concatenation order, which changes
        the embedded string -- so it must be deterministic."""
        reversed_qdf = question_df.iloc[::-1].reset_index(drop=True)
        out = concat_question_answer(answer_df, reversed_qdf)
        first_line = out.loc[0, "open_text_block"].split("\n")[0]
        assert first_line.startswith("Question: What about school?")

    def test_non_question_columns_untouched(self, answer_df, question_df):
        out = concat_question_answer(answer_df, question_df)
        pd.testing.assert_series_equal(out["DAWBA_ID"], answer_df["DAWBA_ID"])
        pd.testing.assert_series_equal(out["impact_score"], answer_df["impact_score"])

    def test_row_count_preserved(self, answer_df, question_df):
        out = concat_question_answer(answer_df, question_df)
        assert len(out) == len(answer_df)

    def test_single_question(self, answer_df, question_df):
        out = concat_question_answer(answer_df, question_df.iloc[[0]])
        assert out.loc[2, "open_text_block"] == (
            "Question: How do you feel? Answer: sad most days"
        )

    def test_missing_question_yields_raises(self, answer_df, question_df):
        bad = question_df.copy()
        bad.loc[0, "question_col"] = np.nan
        with pytest.raises(AssertionError):
            concat_question_answer(answer_df, bad)

    def test_missing_answer_yields_raises(self, answer_df, question_df):
        answer_df.loc[1, "q1"] = np.nan
        with pytest.raises(AssertionError):
            concat_question_answer(answer_df, question_df)


# IMPORT
class TestImportDf:
    @pytest.mark.parametrize("suffix", [".csv", ".tsv", ".parquet"])
    def test_round_trips_each_format(self, answer_df, tmp_path, suffix):
        path = tmp_path / f"data{suffix}"
        if suffix == ".csv":
            answer_df.to_csv(path)
        elif suffix == ".tsv":
            answer_df.to_csv(path, sep="\t")
        else:
            answer_df.to_parquet(path)
        out = import_df(path)
        pd.testing.assert_frame_equal(
            out.reset_index(drop=True), answer_df.reset_index(drop=True)
        )

    @pytest.mark.parametrize("suffix", [".txt", ".xlsx", ".json", ""])
    def test_unrecognized_suffix_raises(self, tmp_path, suffix):
        path = tmp_path / f"data{suffix}"
        path.write_text("whatever")
        with pytest.raises(ValueError, match="Unrecognized"):
            import_df(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_df(tmp_path / "nope.parquet")


SHOT_COL = "open_text_block_0_shot"


@pytest.fixture
def captured(monkeypatch):
    """combine_original_expanded returns None -- the result is only observable
    through export_df."""
    calls = []

    def fake_export_df(save_path, df, impact_type):
        calls.append(
            {"save_path": save_path, "df": df.copy(), "impact_type": impact_type}
        )

    monkeypatch.setattr(shot, "export_df", fake_export_df)
    return calls


def shot_array(prefix, dawba_id, n):
    return np.array([f"{prefix}-{dawba_id}-{i}" for i in range(n)], dtype=object)


@pytest.fixture
def original_df():
    return pd.DataFrame(
        {
            "DAWBA_ID": [1, 2, 3],
            "impact_score": [4, 6, 8],
            "open_text_block": ["a", "b", "c"],
            SHOT_COL: [shot_array("orig", i, 2) for i in (1, 2, 3)],
        }
    )


@pytest.fixture
def expanded_df():
    return pd.DataFrame(
        {
            "DAWBA_ID": [1, 2, 3],
            "impact_score": [4, 6, 8],
            "open_text_block": ["a", "b", "c"],
            SHOT_COL: [shot_array("exp", i, 3) for i in (1, 2, 3)],
        }
    )


@pytest.fixture
def write_pair(tmp_path):
    """Lay out base_path/{original,expanded}/<file_name> and return save_path."""

    def _write(original=None, expanded=None, file_name="self.parquet"):
        for name, df in (("original", original), ("expanded", expanded)):
            d = tmp_path / name
            d.mkdir(exist_ok=True)
            if df is not None:
                df.to_parquet(d / file_name)
        (tmp_path / "combined").mkdir(exist_ok=True)
        return tmp_path / "combined" / file_name

    return _write


@pytest.fixture
def combine(write_pair, tmp_path, captured):
    def _combine(original, expanded, impact_type="self"):
        save_path = write_pair(original, expanded)
        combine_original_expanded(
            save_path=save_path,
            impact_type=impact_type,
            log_path=tmp_path / "run.log",
        )
        return captured[-1]["df"]

    return _combine


# CONCATENATION
class TestConcatenation:
    def test_order_is_original_then_expanded(self, combine, original_df, expanded_df):
        out = combine(original_df, expanded_df)
        row = out.loc[out.DAWBA_ID == 1, SHOT_COL].iloc[0]
        assert list(row) == [
            "orig-1-0",
            "orig-1-1",
            "exp-1-0",
            "exp-1-1",
            "exp-1-2",
        ]

    def test_pairs_by_id_not_position(self, combine, original_df, expanded_df):
        """The two jobs write independently, so row order in the two parquet files
        need not agree."""
        shuffled = expanded_df.iloc[[2, 0, 1]].reset_index(drop=True)
        out = combine(original_df, shuffled)
        for dawba_id in (1, 2, 3):
            row = out.loc[out.DAWBA_ID == dawba_id, SHOT_COL].iloc[0]
            assert all(str(dawba_id) == v.split("-")[1] for v in row)

    def test_every_participant_concatenated(self, combine, original_df, expanded_df):
        out = combine(original_df, expanded_df)
        assert out[SHOT_COL].map(len).tolist() == [5, 5, 5]

    def test_row_count_and_order_follow_original(
        self, combine, original_df, expanded_df
    ):
        out = combine(original_df, expanded_df)
        assert out["DAWBA_ID"].tolist() == original_df["DAWBA_ID"].tolist()

    def test_list_valued_entries_accepted(self, combine, original_df, expanded_df):
        """Guards the isinstance(x, list) normalization branch: whether the parquet
        round-trip yields lists or ndarrays is a pyarrow-version detail."""
        original_df[SHOT_COL] = original_df[SHOT_COL].map(list)
        expanded_df[SHOT_COL] = expanded_df[SHOT_COL].map(list)
        out = combine(original_df, expanded_df)
        assert list(out.loc[0, SHOT_COL]) == ["orig-1-0", "orig-1-1"] + [
            "exp-1-0",
            "exp-1-1",
            "exp-1-2",
        ]


# SCHEMA
class TestOutputSchema:
    def test_suffixed_columns_absent(self, combine, original_df, expanded_df):
        out = combine(original_df, expanded_df)
        assert f"{SHOT_COL}_original" not in out.columns
        assert f"{SHOT_COL}_expanded" not in out.columns

    def test_original_non_shot_columns_preserved(
        self, combine, original_df, expanded_df
    ):
        out = combine(original_df, expanded_df)
        assert set(original_df.columns) == set(out.columns)
        pd.testing.assert_series_equal(out["impact_score"], original_df["impact_score"])
        pd.testing.assert_series_equal(
            out["open_text_block"], original_df["open_text_block"]
        )

    def test_conflicting_column_takes_original_value(
        self, combine, original_df, expanded_df
    ):
        """original is the source of truth for everything but the shot column."""
        expanded_df["impact_score"] = [99, 99, 99]
        out = combine(original_df, expanded_df)
        assert out["impact_score"].tolist() == [4, 6, 8]

    def test_expanded_only_columns_dropped(self, combine, original_df, expanded_df):
        expanded_df["expanded_only"] = [1, 2, 3]
        out = combine(original_df, expanded_df)
        assert "expanded_only" not in out.columns

    def test_shot_column_moves_to_end(self, combine, original_df, expanded_df):
        """Documents current behaviour -- the merge appends rather than restoring
        the original position."""
        out = combine(original_df, expanded_df)
        assert out.columns[-1] == SHOT_COL


# EXPORT
class TestExport:
    def test_exports_once_with_save_path_and_impact_type(
        self, combine, captured, original_df, expanded_df, tmp_path
    ):
        combine(original_df, expanded_df, impact_type="parent")
        assert len(captured) == 1
        assert captured[0]["impact_type"] == "parent"
        assert captured[0]["save_path"] == tmp_path / "combined" / "self.parquet"


# FAILURE MODES
class TestFailureModes:
    def test_missing_original_file_raises(
        self, write_pair, tmp_path, captured, expanded_df
    ):
        save_path = write_pair(original=None, expanded=expanded_df)
        with pytest.raises(ValueError, match="Original output file does not exist"):
            combine_original_expanded(
                save_path=save_path, impact_type="self", log_path=tmp_path / "run.log"
            )

    def test_missing_expanded_file_raises(
        self, write_pair, tmp_path, captured, original_df
    ):
        save_path = write_pair(original=original_df, expanded=None)
        with pytest.raises(ValueError, match="Expanded output file does not exist"):
            combine_original_expanded(
                save_path=save_path, impact_type="self", log_path=tmp_path / "run.log"
            )

    def test_id_missing_from_expanded_raises_and_names_it(
        self, combine, original_df, expanded_df
    ):
        """A partially-completed expanded job must not silently shrink the cohort."""
        with pytest.raises(ValueError, match=r"missing from expanded output.*3"):
            combine(original_df, expanded_df.iloc[:2])

    def test_duplicate_id_in_original_raises(self, combine, original_df, expanded_df):
        dup = pd.concat([original_df, original_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(pd.errors.MergeError):
            combine(dup, expanded_df)

    def test_duplicate_id_in_expanded_raises(self, combine, original_df, expanded_df):
        dup = pd.concat([expanded_df, expanded_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(pd.errors.MergeError):
            combine(original_df, dup)
