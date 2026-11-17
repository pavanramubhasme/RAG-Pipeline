"""The evaluation harness itself is code, so it is tested too (fake judge, real RAGAS metrics)."""

import math

from eval.ragas_judge import METRIC_NAMES
from eval.run_eval import (
    contains_answer,
    evaluate_end_to_end,
    evaluate_retrieval,
    is_relevant,
    load_golden,
    run_ablation,
)
from tests.conftest import SAMPLE_DIR
from tests.fakes import FakeRagasEmbedding, FakeRagasLLM, schema_valid_instance

GOLDEN = load_golden(SAMPLE_DIR.parents[1] / "eval" / "golden_set.jsonl")


def test_golden_set_is_well_formed():
    assert len(GOLDEN) >= 15
    for item in GOLDEN:
        assert item["question"].strip()
        if item["reference_source"]:
            assert (SAMPLE_DIR / item["reference_source"]).exists()
            assert item["reference"] and item["must_include"]
            # the labelled section exists as a heading in the reference document
            text = (SAMPLE_DIR / item["reference_source"]).read_text(encoding="utf-8")
            assert f"## {item['reference_section']}" in text


def test_retrieval_metrics(indexed):
    summary = evaluate_retrieval(indexed, GOLDEN)["summary"]
    assert summary["hit_rate@4"] >= 0.9
    assert 0 < summary["mrr"] <= 1
    assert summary["refusal_accuracy"] == 1.0
    assert 0 < summary["precision@4"] <= 1
    assert summary["recall@4"] >= 0.9


def test_every_question_has_exactly_one_answer_chunk_in_the_sample_index(indexed):
    rows = evaluate_retrieval(indexed, GOLDEN)["rows"]
    assert all(r["relevant_in_index"] == 1 for r in rows if r["answerable"])


def test_relevance_labels():
    assert contains_answer("up to 3 days per week", [["3", "three"]])
    assert not contains_answer("30 days of paid vacation", [["3", "three"]])  # "3" is not "30"
    assert contains_answer("requires manager approval", [["approv"]])  # prefix
    assert not contains_answer("anything", [])
    vacation = {"reference_section": "Vacation", "must_include": [["30", "thirty"]]}
    assert is_relevant("[Employee Handbook > Vacation] Full-time employees receive 30 days.", vacation)
    # Same fact, wrong section: not the answer to "how many vacation days?"
    assert not is_relevant("[Employee Handbook > Business travel] Submit within 30 days.", vacation)


def test_reranker_raises_precision(indexed):
    table = {row["config"]: row for row in run_ablation(indexed, GOLDEN)}
    assert table["hybrid + rerank (default)"]["precision@4"] > table["hybrid (RRF)"]["precision@4"]


def test_ablation_covers_all_configs_and_restores_settings(indexed):
    before = indexed.settings
    table = run_ablation(indexed, GOLDEN)
    assert [row["config"] for row in table][-1] == "hybrid + rerank (default)"
    assert table[0]["refusal_accuracy"] is None  # no reranker -> no refusal possible
    assert indexed.settings == before


def test_ragas_end_to_end_with_fake_judge(indexed):
    report = evaluate_end_to_end(indexed, GOLDEN[:4], (FakeRagasLLM(), FakeRagasEmbedding()), METRIC_NAMES, 2)
    summary = report["summary"]
    assert summary["judge_failures"] == 0
    for name in METRIC_NAMES:
        value = summary[f"ragas_{name}"]
        assert value is not None and 0.0 <= value <= 1.0, name
    scored = [r for r in report["rows"] if not r["refused"]]
    assert all(r["contexts"] for r in scored)


def test_schema_valid_fake_builds_every_ragas_response_model():
    from ragas.metrics.collections.faithfulness.util import NLIStatementOutput, StatementGeneratorOutput

    assert schema_valid_instance(StatementGeneratorOutput).statements
    verdicts = schema_valid_instance(NLIStatementOutput)
    assert not math.isnan(float(verdicts.statements[0].verdict))
