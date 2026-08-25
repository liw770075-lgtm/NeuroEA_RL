"""Train sequential static TD3 with the paper state s_i = [e_0, h_i]."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from RL.Static.train_sac_static import add_arguments, parse_names, train
from RL.shared.env.problem_utils import ProblemTask


def parse_hidden_dims(text):
    dims = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive integers")
    return dims


def parse_args():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(log_dir="RL/runs/Static/paper_state_td3_sop_f1")
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(128, 128))
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = [
        ProblemTask(name, (args.population_size, 1, args.dimension, args.max_fe))
        for name in parse_names(args.problem_names)
    ]
    summary = train(args, tasks, args.log_dir, args.seed, algorithm="td3")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
