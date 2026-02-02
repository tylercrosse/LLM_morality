#!/usr/bin/env python3
import argparse
from pathlib import Path

from plotting_adapter import PlottingAdapter
from analyzers import (
    create_reciprocity_comparison,
    compute_moral_regret,
    analyze_cross_game_generalization,
    analyze_prompt_robustness,
)


ANALYSIS_FUNCS = {
    "reciprocity": create_reciprocity_comparison,
    "regret": None,
    "generalization": analyze_cross_game_generalization,
    "prompts": analyze_prompt_robustness,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./publication_figures")
    parser.add_argument(
        "--analyses",
        nargs="+",
        default=["reciprocity", "regret", "generalization", "prompts"],
        help="Choose from: reciprocity, regret, generalization, prompts",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = PlottingAdapter(results_dir=args.results_dir)

    analyses = args.analyses
    total = len(analyses)
    step = 1

    for analysis in analyses:
        if analysis == "regret":
            print(f"[{step}/{total}] Computing moral regret...")
            de_path = compute_moral_regret(adapter, output_dir, label="Deontological")
            ut_path = compute_moral_regret(adapter, output_dir, label="Utilitarian")
            print(f"  ✓ Saved: {de_path.name}")
            print(f"  ✓ Saved: {ut_path.name}")
        elif analysis in ANALYSIS_FUNCS:
            print(f"[{step}/{total}] Computing {analysis}...")
            path = ANALYSIS_FUNCS[analysis](adapter, output_dir)
            if path is not None:
                print(f"  ✓ Saved: {path.name}")
        else:
            print(f"[{step}/{total}] Skipping unknown analysis: {analysis}")
        step += 1


if __name__ == "__main__":
    main()
