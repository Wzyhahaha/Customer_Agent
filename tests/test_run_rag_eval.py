from pathlib import Path

from scripts import run_rag_eval


def test_pipeline_output_path_uses_pipeline_name():
    output_path = run_rag_eval.pipeline_output_path(Path("reports"), "baseline")

    assert output_path == Path("reports") / "eval_results_baseline.json"


def test_main_runs_both_pipelines_and_report(monkeypatch):
    calls = []

    def fake_run_pipeline(*, pipeline, input_path, output_dir):
        calls.append(("pipeline", pipeline, input_path, output_dir))
        return output_dir / f"eval_results_{pipeline}.json"

    def fake_generate_report(**kwargs):
        calls.append(("report", kwargs))

    monkeypatch.setattr(run_rag_eval, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(run_rag_eval, "generate_report", fake_generate_report)

    exit_code = run_rag_eval.main(["--pipeline", "all", "--input", "data/eval/test_queries.jsonl"])

    assert exit_code == 0
    assert calls[0][0:2] == ("pipeline", "baseline")
    assert calls[1][0:2] == ("pipeline", "enhanced")
    assert calls[2][0] == "report"
    assert calls[2][1]["output_path"].endswith("reports\\eval_report.md")
