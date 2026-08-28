# test_residual_wrapper.py
# Run w/ uv run pytest tests/test_residual_wrapper.py -v
# Written in collaboration w/ Claude Opus 5

# TODO: Write tests for parquet writer

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.residual_wrapper import ParquetStreamWriter, is_good_config

LLAMA = "Llama-3.3-70B-Instruct"
QWEN = "Qwen3.6-27B"
GEMINI = "gemini-2.5-pro"

LR_AGGS = ["block", "avg", "max"]
NEW_AGGS = ["binary-new", "scale-new"]
OG_AGGS = ["proba-limited-og", "binary-og"]

NEW_DESC = ["shot_zero", "shot_few"]


######## is_good_config — impactme LR branch ########
class TestImpactMeLRBranch:
    """Selected by prob_agg in {block, avg, max}: Llama only, temp 0.6, on the
    segment-style desc values."""

    @pytest.mark.parametrize("prob_agg", LR_AGGS)
    @pytest.mark.parametrize("desc_seg", ["turns", "pt_noshort"])
    def test_accepts_llama_at_temp_point_six(self, prob_agg, desc_seg):
        assert is_good_config(desc_seg, LLAMA, prob_agg, 0.6)

    @pytest.mark.parametrize(
        "desc_seg", ["shot_zero", "shot_few", "original", "expanded", "combined"]
    )
    def test_rejects_non_segment_desc(self, desc_seg):
        assert not is_good_config(desc_seg, LLAMA, "avg", 0.6)

    @pytest.mark.parametrize("model", [QWEN, GEMINI])
    def test_rejects_other_models(self, model):
        assert not is_good_config("turns", model, "avg", 0.6)

    @pytest.mark.parametrize("temp", [0.0, 0.2, 0.4, 1.0, 1.2])
    def test_rejects_other_temps(self, temp):
        """0.6 is the temperature the prelim extraction was run at, so nothing
        else can exist on disk for these aggregations."""
        assert not is_good_config("turns", LLAMA, "avg", temp)


######## is_good_config — new shot branch ########
class TestNewShotBranch:
    """Selected by prob_agg in {binary-new, scale-new}: shot descs, three models,
    each with its own allowed temperature grid."""

    @pytest.mark.parametrize("prob_agg", NEW_AGGS)
    @pytest.mark.parametrize("desc_seg", NEW_DESC)
    @pytest.mark.parametrize("llm_model", [LLAMA, QWEN])
    @pytest.mark.parametrize("temp", [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    def test_accepts_llama_qwen_across_its_grid(
        self, prob_agg, desc_seg, llm_model, temp
    ):
        assert is_good_config(desc_seg, llm_model, prob_agg, temp)

    @pytest.mark.parametrize("prob_agg", NEW_AGGS)
    @pytest.mark.parametrize("desc_seg", NEW_DESC)
    @pytest.mark.parametrize("temp", [0, 0.4, 0.8, 1.2, 1.6, 2.0])
    def test_accepts_gemini_across_its_wider_grid(self, prob_agg, desc_seg, temp):
        """Gemini runs on [0, 2], so its grid is the doubled version of the
        others"""
        assert is_good_config(desc_seg, GEMINI, prob_agg, temp)

    @pytest.mark.parametrize("prob_agg", NEW_AGGS)
    @pytest.mark.parametrize("desc_seg", NEW_DESC)
    @pytest.mark.parametrize("llm_model", [LLAMA, QWEN])
    @pytest.mark.parametrize("temp", [1.2, 1.6, 2.0])
    def test_rejects_gemini_temps_for_llama_qwen(
        self, prob_agg, desc_seg, llm_model, temp
    ):
        assert not is_good_config(desc_seg, llm_model, prob_agg, temp)

    @pytest.mark.parametrize("prob_agg", NEW_AGGS)
    @pytest.mark.parametrize("desc_seg", NEW_DESC)
    @pytest.mark.parametrize("temp", [0.2, 0.6, 1.4, 1.8])
    def test_rejects_off_grid_gemini_temps(self, prob_agg, desc_seg, temp):
        assert not is_good_config(desc_seg, GEMINI, prob_agg, temp)

    @pytest.mark.parametrize("desc_seg", ["original", "expanded", "turns"])
    @pytest.mark.parametrize("prob_agg", NEW_AGGS)
    @pytest.mark.parametrize("llm_model", [LLAMA, QWEN])
    @pytest.mark.parametrize("temp", [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    def test_rejects_non_shot_desc(self, desc_seg, prob_agg, llm_model, temp):
        assert not is_good_config(desc_seg, llm_model, prob_agg, temp)

    def test_unknown_model_rejected_before_temp_lookup(self):
        assert not is_good_config("shot_few", "some-future-model", "scale-new", 0.4)

    def test_integer_and_float_zero_both_accepted(self):
        """The grids are written with a bare 0; temps arriving as floats from a
        JSON param dict must still match."""
        assert is_good_config("shot_zero", LLAMA, "scale-new", 0)
        assert is_good_config("shot_zero", LLAMA, "scale-new", 0.0)


######## is_good_config — old shot branch ########
class TestOldShotBranch:
    """Selected by prob_agg in {proba-limited-og, binary-og}: Llama only,
    temp 0.6, on the pre-shot desc values."""

    @pytest.mark.parametrize("desc_seg", ["original", "combined"])
    @pytest.mark.parametrize("prob_agg", OG_AGGS)
    def test_accepts_llama_at_temp_point_six(self, desc_seg, prob_agg):
        assert is_good_config(desc_seg, LLAMA, prob_agg, 0.6)

    @pytest.mark.parametrize("model", [QWEN, GEMINI])
    @pytest.mark.parametrize("desc_seg", ["original", "combined"])
    @pytest.mark.parametrize("prob_agg", OG_AGGS)
    def test_rejects_newer_models(self, desc_seg, model, prob_agg):
        assert not is_good_config(desc_seg, model, prob_agg, 0.6)

    @pytest.mark.parametrize("desc_seg", ["original", "combined"])
    @pytest.mark.parametrize("prob_agg", OG_AGGS)
    @pytest.mark.parametrize("temp", [0.0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.8, 2.0])
    def test_rejects_other_temps(self, desc_seg, prob_agg, temp):
        assert not is_good_config(desc_seg, LLAMA, prob_agg, temp)

    @pytest.mark.parametrize("desc_seg", NEW_DESC + ["turns", "pt_noshort"])
    @pytest.mark.parametrize("prob_agg", OG_AGGS)
    def test_rejects_non_og_desc(self, desc_seg, prob_agg):
        assert not is_good_config(desc_seg, LLAMA, prob_agg, 0.6)

    def test_expanded_binary_og_excluded_as_redundant(self):
        """expanded + binary-og is covered by the new shot iteration, so it is
        skipped to avoid fitting the same configuration twice. Note this only
        fires for a config that has already passed every other check -- the elif
        is attached to the validity test above it."""
        assert not is_good_config("expanded", LLAMA, "binary-og", 0.6)

    def test_expanded_proba_limited_still_accepted(self):
        """Pins that the exclusion is narrow -- only the binary pairing is
        redundant."""
        assert is_good_config("expanded", LLAMA, "proba-limited-og", 0.6)

    def test_expanded_binary_og_rejected_for_other_reasons_too(self):
        """The redundancy skip must not become the only reason expanded configs
        fail; a wrong temp still rejects independently."""
        assert not is_good_config("expanded", LLAMA, "binary-og", 0.4)


######## is_good_config — branch selection ########
class TestBranchSelection:
    @pytest.mark.parametrize(
        "prob_agg, desc_seg", [("avg", "shot_few"), ("scale-new", "turns")]
    )
    def test_every_branch_keys_on_prob_agg(self, prob_agg, desc_seg):
        assert not is_good_config(desc_seg, LLAMA, prob_agg, 0.6)

    @pytest.mark.parametrize(
        "prob_agg", ["not-an-agg", "", "binary", "scale_new", "proba-limited"]
    )
    def test_unknown_aggregation_rejected(self, prob_agg):
        for desc_seg in ("turns", "shot_few", "original"):
            assert not is_good_config(desc_seg, LLAMA, prob_agg, 0.6)

    @pytest.mark.parametrize("desc_seg", ["typo_desc", "", "shot-zero", "orig"])
    def test_unknown_desc_seg_rejected_under_every_agg(self, desc_seg):
        for prob_agg in LR_AGGS + NEW_AGGS + OG_AGGS:
            assert not is_good_config(desc_seg, LLAMA, prob_agg, 0.6)


######## ParquetStreamWriter ########
@pytest.fixture
def path(tmp_path):
    return tmp_path / "nested" / "out.parquet"


class TestParquetStreamWriterBasics:
    def test_creates_parent_dirs(self, path):
        ParquetStreamWriter(path)
        assert path.parent.exists()

    def test_removes_existing_file_at_construction(self, path):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"stale")
        ParquetStreamWriter(path)
        assert not path.exists()

    def test_no_file_written_before_flush_threshold(self, path):
        w = ParquetStreamWriter(path, buffer_rows=100)
        w.write(pd.DataFrame({"a": [1, 2, 3]}))
        assert not path.exists()

    def test_close_flushes_the_remainder(self, path):
        w = ParquetStreamWriter(path, buffer_rows=100)
        w.write(pd.DataFrame({"a": [1, 2, 3]}))
        w.close()
        assert pd.read_parquet(path)["a"].tolist() == [1, 2, 3]

    def test_round_trips_all_written_rows_in_order(self, path):
        chunks = [pd.DataFrame({"a": [i], "b": [f"r{i}"]}) for i in range(10)]
        w = ParquetStreamWriter(path, buffer_rows=3)
        for c in chunks:
            w.write(c)
        w.close()
        pd.testing.assert_frame_equal(
            pd.read_parquet(path), pd.concat(chunks, ignore_index=True)
        )

    def test_threshold_batches_row_groups(self, path):
        """buffer_rows exists to avoid thousands of tiny row groups, so one flush
        should cover several writes."""
        w = ParquetStreamWriter(path, buffer_rows=4)
        for i in range(8):
            w.write(pd.DataFrame({"a": [i, i]}))  # 2 rows each -> flush every 2 writes
        w.close()
        assert pq.ParquetFile(path).num_row_groups == 4
        assert len(pd.read_parquet(path)) == 16

    def test_index_is_not_persisted(self, path):
        """preserve_index=False -- per-config indices are meaningless once chunks
        are concatenated."""
        w = ParquetStreamWriter(path, buffer_rows=1)
        w.write(pd.DataFrame({"a": [1, 2]}, index=[99, 100]))
        w.close()
        assert list(pd.read_parquet(path).index) == [0, 1]
