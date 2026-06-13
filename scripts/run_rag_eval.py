from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import eval as eval_module
from rag.eval_report import generate_report


DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "test_queries.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"


def pipeline_output_path(output_dir: Path, pipeline: str) -> Path:
    return output_dir / f"eval_results_{pipeline}.json"


def _with_summary(results: dict) -> dict:
    merged = dict(results)
    merged.update(eval_module._results_summary(results))
    return merged


def run_pipeline(*, pipeline: str, input_path: Path, output_dir: Path) -> Path:
    output_path = pipeline_output_path(output_dir, pipeline)
    print(f"\n运行 {pipeline} 评测")
    print(f"数据集: {input_path}")
    print(f"输出: {output_path}")

    results = eval_module.evaluate(str(input_path), pipeline=pipeline)
    results["pipeline"] = pipeline
    eval_module._save_results(_with_summary(results), str(output_path))
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation and generate reports.")
    parser.add_argument(
        "--pipeline",
        choices=["baseline", "enhanced", "all"],
        default="all",
        help="要评测的检索流程，默认 all。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"评测集 JSONL 路径，默认：{DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"结果输出目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="只输出 JSON 指标，不生成 Markdown 报告。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    output_dir = args.output_dir

    if not input_path.exists():
        print(f"评测集不存在：{input_path}", file=sys.stderr)
        return 1

    pipelines = ["baseline", "enhanced"] if args.pipeline == "all" else [args.pipeline]
    outputs: dict[str, Path] = {}
    for pipeline in pipelines:
        outputs[pipeline] = run_pipeline(
            pipeline=pipeline,
            input_path=input_path,
            output_dir=output_dir,
        )

    if not args.no_report:
        baseline_path = outputs.get("baseline") or pipeline_output_path(output_dir, "baseline")
        enhanced_path = outputs.get("enhanced") or pipeline_output_path(output_dir, "enhanced")
        report_path = output_dir / "eval_report.md"
        generate_report(
            baseline_path=str(baseline_path),
            enhanced_path=str(enhanced_path),
            output_path=str(report_path),
            dataset_path=str(input_path),
        )

    print("\n评测完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
