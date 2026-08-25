"""Train independent paper-state static SAC models with multiple seeds."""

from __future__ import annotations

from copy import copy
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


def parse_problem_names(text):
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


def parse_seeds(text):
    seeds = []
    for item in (part.strip() for part in str(text).split(",")):
        if not item:
            continue
        match = re.fullmatch(r"(-?\d+)-(-?\d+)", item)
        if match:
            first, last = (int(value) for value in match.groups())
            step = 1 if last >= first else -1
            seeds.extend(range(first, last + step, step))
        else:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicate values.")
    return seeds


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__), "output-root")
    parser.set_defaults(
        problem_names="SOP_F1",
        output_root="RL/runs/Static/paper_state_sopf1_multi_seed",
    )
    parser.add_argument(
        "--seeds",
        default="0,42,100,2025,3407",
        help="Comma-separated seeds or an inclusive integer range, e.g. 0,42,100 or 0-4.",
    )
    args = parser.parse_args()
    problems = parse_problem_names(args.problem_names)
    seeds = parse_seeds(args.seeds)
    summaries = []

    for problem_name in problems:
        task = ProblemTask(
            problem_name,
            (args.population_size, 1, args.dimension, args.max_fe),
        )
        for seed in seeds:
            run_args = copy(args)
            run_args.seed = seed
            log_dir = Path(args.output_root) / problem_name / f"seed_{seed}"
            print(f"\n===== Static paper state: {problem_name}, seed={seed} =====")
            summary = train(run_args, [task], log_dir, seed)
            summary.update(problem_name=problem_name, seed=seed, log_dir=str(log_dir))
            summaries.append(summary)
            print(
                f"[{problem_name}] seed={seed} "
                f"best_training_fitness={summary['train_best_fitness']:.6e}",
                flush=True,
            )

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem_name",
        "seed",
        "episodes",
        "train_best_fitness",
        "best_last_10_mean",
        "reward_last_10_mean",
        "after_eval_best_mean",
        "after_eval_reward_mean",
        "best_checkpoint",
        "log_dir",
    ]
    rows = []
    for summary in summaries:
        rows.append(
            {
                **{key: summary.get(key) for key in fields},
                "after_eval_best_mean": summary["after_eval"]["best_mean"],
                "after_eval_reward_mean": summary["after_eval"]["reward_mean"],
            }
        )
    with (root / "aggregate_seed_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "aggregate_seed_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"args": vars(args), "problems": problems, "seeds": seeds, "summaries": summaries},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print("\n===== Best training fitness by seed =====")
    for summary in summaries:
        print(
            f"{summary['problem_name']} seed={summary['seed']}: "
            f"{summary['train_best_fitness']:.6e}"
        )
    print(f"\nAggregate CSV: {(root / 'aggregate_seed_results.csv').resolve()}")


if __name__ == "__main__":
    main()
