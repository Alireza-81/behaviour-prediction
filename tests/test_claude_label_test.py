import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("claude_label_test", Path(__file__).parents[1]/"scripts/label_with_claude.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def test_payload_excludes_ground_truth_and_model_identity():
    row = dict(prompt="question", response="answer", token_limit_reached="False",
               run="secret", step=10, human_harmful="1", category="secret")
    assert judge.payload(row) == dict(prompt="question", response="answer", token_limit_reached=False)


def test_prepare_can_select_only_remaining_run():
    rows = [dict(run="hate_speech"), dict(run="violent_threats"), dict(run="violent_threats")]
    assert judge.select_runs(rows, ["violent_threats"]) == rows[1:]
    with pytest.raises(ValueError, match="No response CSVs"):
        judge.select_runs(rows, ["missing"])


def test_request_compatible_with_installed_sdk_without_network():
    import inspect
    anthropic = pytest.importorskip("anthropic")
    with anthropic.Anthropic(api_key="not-a-real-key") as client:
        args = judge.request_arguments(dict(model="claude-sonnet-4-6", rubric="rubric"), "sample")
        inspect.signature(client.messages.create).bind(**args)
    assert "temperature" not in args
    assert "extra_body" not in args
    assert "top_p" not in args and "top_k" not in args


@pytest.mark.parametrize("value", [True, "1", 2])
def test_judge_labels_are_strict(value):
    with pytest.raises(ValueError):
        judge.validate_judgment(dict(harmful=value, behaviour_tags=[], needs_review=False, reason="test"))


def test_invalid_and_uncertain_predictions_do_not_inflate_coverage(tmp_path):
    rows = [dict(human_harmful=str(i%2)) for i in range(50)]
    results = {0: dict(judgment=None, status="invalid", cost_usd=.01),
               1: dict(judgment=dict(harmful=None, behaviour_tags=[], needs_review=True, reason="uncertain"), status="ok", cost_usd=.01)}
    report = judge.write_report(tmp_path, dict(sample=rows), results)
    assert report["completed"] == 2 and report["decided"] == 0
    assert report["agreement"] is None and report["correct_of_all_50"] == 0


@pytest.mark.parametrize("tag", ["harassment", "self_harm_facilitation"])
def test_revised_rubric_tags_are_accepted(tag):
    value = dict(harmful=1, behaviour_tags=[tag], needs_review=False, reason="Visible harmful content")
    assert judge.validate_judgment(value) == value


def test_reused_sample_refreshes_labels_without_changing_response(tmp_path):
    import csv
    row = dict(run="run", step="10", probe_id="p", prompt="question", response="answer",
               group="benign", category="test", human_harmful="1", review_notes="")
    path = tmp_path / "reviews.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(dict(row, human_harmful="0"))
    sample = judge.refresh_sample([dict(row, source_file=str(path))])
    assert sample[0]["human_harmful"] == "0"
    assert sample[0]["response"] == row["response"]
    with pytest.raises(ValueError):
        judge.refresh_sample([dict(row, source_file=str(path), response="changed")])


def test_full_export_handles_unlabeled_rows_without_claiming_agreement(tmp_path):
    row = dict(run="model", human_harmful="", response="answer")
    result = dict(judgment=dict(harmful=1, behaviour_tags=["harassment"], needs_review=False,
                               reason="Abusive response"), status="ok", cost_usd=.01)
    report = judge.write_report(tmp_path, dict(sample=[row], all_responses=True), {0: result})
    assert report["total"] == report["completed"] == report["decided"] == 1
    assert report["human_comparisons"] == 0 and report["agreement"] is None
    assert (tmp_path / "labeled/model.csv").exists()


def test_compact_judgment_needs_no_explanation():
    value = dict(harmful=1, behaviour_tags=["harassment"], needs_review=False)
    assert judge.validate_judgment(value) == value
    with pytest.raises(ValueError):
        judge.validate_judgment(dict(harmful=None, behaviour_tags=[], needs_review=False))


def test_request_uses_saved_model_and_token_limit():
    args = judge.request_arguments(dict(model="claude-haiku-4-5", rubric="compact", max_tokens=128), "sample")
    assert args["model"] == "claude-haiku-4-5"
    assert args["max_tokens"] == 128


def test_json_fences_recover_without_changing_labels():
    import json
    value = dict(harmful=1, behaviour_tags=['harassment'], needs_review=False)
    assert judge.parse_judgment('```json\n'+json.dumps(value)+'\n```') == value
    with pytest.raises(ValueError):
        judge.parse_judgment('Explanation\n'+json.dumps(value))
    with pytest.raises(ValueError):
        judge.parse_judgment('```json\n'+json.dumps(value))
