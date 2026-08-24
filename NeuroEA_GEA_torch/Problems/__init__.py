import importlib
import inspect
import pkgutil

from NeuroEA_GEA_torch import Problems
from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


def get_problem_instance(problem_name, N=100, M=1, D=10, max_fe=10000, **kwargs):
    pkg_path = Problems.__path__
    pkg_name = Problems.__name__

    for _, module_full_name, _ in pkgutil.walk_packages(pkg_path, pkg_name + "."):
        try:
            module = importlib.import_module(module_full_name)
            classes = inspect.getmembers(module, inspect.isclass)
            for name, cls in classes:
                if name.lower() != problem_name.lower():
                    continue
                if issubclass(cls, Problem) and cls is not Problem:
                    return cls(N=N, M=M, D=D, max_fe=max_fe, **kwargs)
        except Exception:
            continue

    raise ValueError(f"Unable to find problem class: {problem_name}.")
