"""Train stable per-generation dynamic SAC."""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from RL.Dynamic.train import main


if __name__ == "__main__":
    main("sac", "dynamic")
