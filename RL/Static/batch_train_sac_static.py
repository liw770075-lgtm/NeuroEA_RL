"""Train one independent paper-state static SAC model per problem."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import re
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from RL.Static.train_sac_static import add_arguments, train
from RL.shared.env.problem_utils import ProblemTask


def parse_names(text):
    names = []
    for item in (part.strip() for part in str(text).split(",")):
        match = re.fullmatch(r"(.+_F)(\d+)-(?:(?:.+_F)?)(\d+)", item, re.IGNORECASE)
        if match:
            prefix, first, last = match.groups()
            first, last = int(first), int(last)
            step = 1 if last >= first else -1
            names.extend(f"{prefix}{index}" for index in range(first, last + step, step))
        elif item:
            names.append(item)
    return names


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__), "output-root")
    parser.set_defaults(problem_names="SOP_F1-SOP_F10", episodes=5000)
    args = parser.parse_args()
    summaries = []
    for index, problem_name in enumerate(parse_names(args.problem_names)):
        seed = args.seed + index * args.seed_stride
        print(f"\n===== Static paper state: {problem_name} =====")
        task = ProblemTask(
            problem_name,
            (args.population_size, 1, args.dimension, args.max_fe),
        )
        summary = train(args, [task], Path(args.output_root) / problem_name, seed)
        summary.update(problem_name=problem_name, seed=seed)
        summaries.append(summary)

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem_name", "seed", "episodes", "train_best_fitness",
        "best_last_10_mean", "reward_last_10_mean", "best_checkpoint",
    ]
    with (root / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item.get(key) for key in fields} for item in summaries)
    with (root / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "summaries": summaries}, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
