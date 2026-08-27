# NeuroEA-RL

Official implementation of:

**Trainable Evolutionary Algorithms: A Reinforcement Learning Perspective on Algorithm Configuration**

Wenbiao Li, Shangshang Yang, Ye Tian, and Zimo Sheng

The 8th International Conference on Data-driven Optimization of Complex Systems (DOCS 2026)

## Overview

NeuroEA-RL provides the official implementation of the reinforcement-learning-based
configuration framework for NeuroEA. The framework studies two complementary
configuration paradigms:

- **Static configuration (S-SAC):** generates the 30 NeuroEA parameters before optimization and keeps them fixed during the run.
- **Dynamic configuration (D-SAC):** updates the complete NeuroEA parameter vector online according to the current search state.

NeuroEA-RL is a research project for learning the parameter configuration of NeuroEA with reinforcement learning. It provides a PyTorch implementation of NeuroEA and supports Soft Actor-Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradient (TD3) agents for learning configuration policies:

- **Dynamic configuration:** SAC produces a complete parameter vector for every NeuroEA generation.
- **Static configuration:** SAC or TD3 selects the parameters sequentially, after which NeuroEA runs to completion with the resulting fixed configuration.

The current implementation focuses on single-objective optimization and includes `SOP_F1`–`SOP_F10` and `BBOB_F1`–`BBOB_F10` benchmark problems.

## Main Training Entry Points

| Configuration | Agent | Entry point | Agent action | Observation | When parameters are applied |
| --- | --- | --- | --- | --- | --- |
| Dynamic | SAC | `RL/Dynamic/train_sac_dynamic.py` | A complete normalized parameter vector at every step | Search summary, current control parameters, ELA features, and optional task context | Before every NeuroEA generation |
| Static | SAC | `RL/Static/train_sac_static.py` | One normalized parameter at every step | Initial ELA vector `e_0` and current parameter-index one-hot vector `h_i` (39 dimensions) | After all parameters have been selected |
| Static | TD3 | `RL/Static/train_td3_static.py` | One normalized parameter at every step | The same paper-state observation, `s_i = [e_0, h_i]`, used by static SAC | After all parameters have been selected |

Actions in both environments are normalized to `[-1, 1]`. The environment maps them back to the actual lower and upper bounds of each NeuroEA parameter.

### Dynamic configuration

The dynamic environment treats one NeuroEA generation as one reinforcement-learning step:

1. Construct an observation from the current population and algorithm state.
2. Use SAC to produce a complete NeuroEA parameter vector.
3. Run one NeuroEA generation with the new parameters.
4. Compute a reward from the change in the best-fitness gap.
5. Repeat until the maximum number of function evaluations is reached.

The observation uses normalized search-summary and ELA features to reduce scale differences across optimization problems. `train_sac_dynamic.py` is a dedicated **SAC + dynamic** entry point. The same directory also contains entry points for first-action SAC and dynamic TD3 experiments.

### Static configuration

The static environment divides one complete configuration into a sequence of reinforcement-learning steps. SAC or TD3 selects one scalar parameter at each step. The observation follows the paper definition `s_i = [e_0, h_i]`: the 9-dimensional initial ELA vector remains fixed throughout the episode, while the 30-dimensional one-hot vector identifies the parameter currently being selected.

After all parameters have been selected, NeuroEA runs to completion with the fixed parameter vector. The final reward is the logarithmic improvement from the initial best-fitness gap to the final best-fitness gap. This reward is then assigned back to the parameter-selection transitions. Optional reward shaping relative to the default configuration is also supported.

### ELA feature extraction

The ELA vector uses the actual output order of the public [DesignX](https://github.com/MetaEvo/DesignX) `env/ela_feature.py` pipeline. This code-level order differs from the order of the declared selection list in DesignX. The values are calculated by the standard [pflacco](https://pflacco.readthedocs.io/en/stable/pflacco.classical_ela_features.html) implementation pinned to version `1.2.2`.

| Index | Feature name | Interpretation |
| ---: | --- | --- |
| 0 | `ela_meta.lin_simple.intercept` | Intercept of the simple linear meta-model |
| 1 | `ela_meta.lin_w_interact.adj_r2` | Adjusted R-squared of the linear model with pairwise interactions |
| 2 | `ela_meta.quad_simple.adj_r2` | Adjusted R-squared of the quadratic model without interactions |
| 3 | `ic.h_max` | Maximum information content |
| 4 | `ic.eps_ratio` | Half partial-information sensitivity |
| 5 | `ic.m0` | Initial partial information |
| 6 | `ela_distr.number_of_peaks` | Estimated number of peaks in the objective-value density |
| 7 | `nbc.nn_nb.mean_ratio` | Mean nearest-neighbor to nearest-better-neighbor distance ratio |
| 8 | `nbc.dist_ratio.coeff_var` | Coefficient of variation of nearest-better distance ratios |

Before extraction, objective values are min-max normalized using the DesignX denominator `max(Y) - min(Y) + 1e-15`. Information-content and tie-breaking randomness use the fixed default seed `42`. Undefined scalar results follow the public DesignX sanitization rule: `NaN -> 0` and `Inf -> 1`. Finite negative values are retained because they are valid outputs for features such as adjusted R-squared. When all objective values are equal, the DesignX constant-landscape branch sets `ela_distr.number_of_peaks` to `1` and continues extracting the remaining features. The default post-extraction divisor is `1.0`, so no second scaling is applied.

The sampling protocols are deliberately explicit:

- Static RL computes `e_0` from the initial NeuroEA population, so the ELA sample count equals `--population-size`.
- Dynamic RL recomputes ELA from the current population at every generation.
- The standalone audit command below defaults to the DesignX offline sampling convention of `100 * D` points.

At `D=20`, the linear interaction model contains `210` predictors. A population of `100` therefore makes its adjusted R-squared statistically underdetermined, while the DesignX offline sample of `2000` points does not. The implementation emits a `RuntimeWarning` for this condition. Paper results must state whether ELA is population-based (`N=100`) or independently sampled (`N=100*D`); these protocols are not interchangeable.

```powershell
python .\RL\shared\ELA\ELA.py `
  --problem-names BBOB_F1 `
  --dimension 10 `
  --sample-factor 100 `
  --seed 42
```

ELA extraction remains fail-fast for missing dependencies, invalid shapes, non-finite samples, insufficient samples, and calculation errors. A constant objective vector is treated as the explicit DesignX convergence case rather than as an extraction failure; the environment never silently replaces a failed ELA vector with an all-zero vector.

## Repository Structure

```text
NeuroEA_RL/
├── EAs/                         # GA/CMA-ES parameter-configuration baselines
├── NeuroEA_GEA_torch/           # PyTorch implementation of NeuroEA
│   ├── Algorithms/              # NeuroEA, GA, operators, and blocks
│   ├── Problems/                # SOP and BBOB benchmark problems
│   └── utils/                   # RNG, sorting, and fitness utilities
└── RL/
    ├── Dynamic/
    │   ├── train_sac_dynamic.py # Main dynamic SAC entry point
    │   ├── train.py             # Shared dynamic training implementation
    │   ├── test_sac_dynamic.py  # Dynamic SAC evaluation entry point
    │   └── common.py            # Dynamic observations and environment helpers
    ├── Static/
    │   ├── train_sac_static.py  # Main static SAC entry point
    │   ├── train_td3_static.py  # Main static TD3 entry point
    │   ├── batch_train_sac_static.py
    │   ├── multi_seed_train_sac_static.py
    │   └── action_history_env.py # Legacy 69-dimensional environment (not used by main scripts)
    └── shared/                  # SAC, TD3, replay buffers, trainers, ELA, and environments
```

## Installation

Python 3.11 and an isolated virtual environment are recommended. The following commands use Windows PowerShell:

```powershell
git clone https://github.com/liw770075-lgtm/NeuroEA_RL.git
cd NeuroEA_RL

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPU training, install a PyTorch build compatible with the CUDA version on your system, and run the scripts with `--device cuda` or `--device auto`. The default configuration uses CPU and `float64` tensors for the NeuroEA environment.

Verify that both main entry points can be imported successfully:

```powershell
python .\RL\Dynamic\train_sac_dynamic.py --help
python .\RL\Static\train_sac_static.py --help
```

> ELA extraction requires the pinned `pflacco==1.2.2` dependency and its compatible NumPy, SciPy, and scikit-learn versions from `requirements.txt`. Dependency or extraction failures stop training instead of silently substituting an all-zero vector.

## Quick Start

Run all commands from the repository root.

### Dynamic SAC smoke test

Use a small number of episodes and function evaluations to verify the complete pipeline:

```powershell
python .\RL\Dynamic\train_sac_dynamic.py `
  --problem-names SOP_F1 `
  --episodes 1 `
  --population-size 100 `
  --dimension 10 `
  --max-fe 200 `
  --eval-rounds 1 `
  --output-root RL/runs/dynamic/smoke
```

### Default dynamic SAC training

```powershell
python .\RL\Dynamic\train_sac_dynamic.py `
  --problem-names SOP_F1 `
  --episodes 5000 `
  --population-size 100 `
  --dimension 10 `
  --max-fe 10000 `
  --device cpu `
  --output-root RL/runs/dynamic/sac_sopf1
```

The dynamic entry point accepts comma-separated problem names and inclusive problem ranges. A separate model is trained for every problem. The following 1000-episode command is a reduced-budget orchestration example, not the paper configuration:

```powershell
python .\RL\Dynamic\train_sac_dynamic.py `
  --problem-names "SOP_F1-SOP_F10,BBOB_F1-BBOB_F10" `
  --episodes 1000 `
  --output-root RL/runs/dynamic/all_problems
```

### Static SAC smoke test

```powershell
python .\RL\Static\train_sac_static.py `
  --problem-names SOP_F1 `
  --episodes 1 `
  --population-size 100 `
  --dimension 10 `
  --max-fe 200 `
  --eval-rounds 1 `
  --log-dir RL/runs/Static/smoke
```

### Default static SAC training

```powershell
python .\RL\Static\train_sac_static.py `
  --problem-names SOP_F1 `
  --episodes 5000 `
  --population-size 100 `
  --dimension 10 `
  --max-fe 10000 `
  --device cpu `
  --log-dir RL/runs/Static/paper_state_sopf1
```

### Static TD3 training

Static TD3 uses the same paper-state environment, `s_i = [e_0, h_i]`, and sequential reward assignment as static SAC, but replaces the stochastic SAC policy with a deterministic TD3 policy and target-policy smoothing:

```powershell
python .\RL\Static\train_td3_static.py `
  --problem-names SOP_F1 `
  --episodes 5000 `
  --population-size 100 `
  --dimension 10 `
  --max-fe 10000 `
  --device cpu `
  --exploration-noise 0.1 `
  --policy-noise 0.2 `
  --noise-clip 0.5 `
  --policy-delay 2 `
  --log-dir RL/runs/Static/paper_state_td3_sopf1
```

The static entry point accepts comma-separated problem names. A single SAC model can be trained across multiple tasks using either cyclic or random task selection. The strict paper-state implementation intentionally excludes task context:

```powershell
python .\RL\Static\train_sac_static.py `
  --problem-names "SOP_F1,SOP_F2,SOP_F3" `
  --task-mode cycle `
  --episodes 5000 `
  --log-dir RL/runs/Static/multitask_sop
```

### Example: independent static models with a reduced budget

```powershell
python .\RL\Static\batch_train_sac_static.py `
  --problem-names SOP_F1-SOP_F10 `
  --episodes 1000 `
  --output-root RL/runs/Static/sop_f1_f10
```

### Example: multi-seed static experiments with a reduced budget

```powershell
python .\RL\Static\multi_seed_train_sac_static.py `
  --problem-names SOP_F1 `
  --seeds "0,42,100,2025,3407" `
  --episodes 1000 `
  --output-root RL/runs/Static/sopf1_multi_seed
```

`--seeds` also accepts inclusive integer ranges, for example `--seeds 0-4`.

## Reproducing the Paper

The paper experiments use the following training budget. These values, rather than the smoke-test settings above, should be used when reproducing reported results:

| Setting | Paper value |
| --- | ---: |
| Training episodes | `5000` |
| Population size | `100` |
| Maximum function evaluations | `10000` |
| Decision-space dimensions | `10` and `20` |

Run each paper problem and random seed at both dimensions. For example, the commands below reproduce the training budget for `SOP_F1`:

```powershell
foreach ($d in 10, 20) {
  python .\RL\Dynamic\train_sac_dynamic.py `
    --problem-names SOP_F1 `
    --episodes 5000 `
    --population-size 100 `
    --dimension $d `
    --max-fe 10000 `
    --device cpu `
    --output-root "RL/runs/paper/dynamic/dim$d"
}
```

```powershell
foreach ($d in 10, 20) {
  python .\RL\Static\train_sac_static.py `
    --problem-names SOP_F1 `
    --episodes 5000 `
    --population-size 100 `
    --dimension $d `
    --max-fe 10000 `
    --device cpu `
    --log-dir "RL/runs/paper/static/dim$d"
}
```

For the complete paper results, repeat these commands with the exact problem list and random seeds reported in the experiment protocol. Do not use the smoke-test settings for result reproduction.

## Main Arguments

### Shared arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--problem-names` | `SOP_F1` | Problem names, separated by commas when multiple tasks are used |
| `--episodes` | `5000` | Number of training episodes; for dynamic multi-problem training, this is the number per problem |
| `--dimension` | `10` | Decision-space dimension |
| `--population-size` | `100` | Population size |
| `--max-fe` | `10000` | Maximum function evaluations in each NeuroEA run |
| `--seed` | `0` | Random seed |
| `--device` | `cpu` | PyTorch device: `cpu`, `cuda`, or `auto` |
| `--dtype` | `float64` | Tensor dtype used by the NeuroEA environment |
| `--eval-every` | `10` | Evaluation interval in episodes |
| `--eval-rounds` | `5` | Repeated runs in each evaluation |
| `--save-every` | `1000` | Checkpoint interval in episodes |
| `--start-steps` | `128` | Initial environment steps that use random actions |
| `--updates-per-step` | `1` | Gradient updates performed at each training opportunity |
| `--include-task-context` | disabled | Add task identity and normalized task configuration to dynamic observations; strict static paper-state training rejects this option |

### Dynamic SAC arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--objectives` | `1` | Number of objectives passed to the single-objective optimization problem |
| `--batch-size` | `64` | SAC batch size |
| `--hidden-dims` | `128,128` | Actor and critic hidden-layer dimensions |
| `--actor-lr` | `3e-4` | Actor learning rate |
| `--critic-lr` | `3e-4` | Critic learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--tau` | `0.005` | Target-network soft-update coefficient |
| `--summary-clip` | `5.0` | Clipping range for normalized search-summary features |
| `--objective-log-scale` | `10.0` | Signed-log scaling factor for objective-summary values |
| `--output-root` | `RL/runs/dynamic/stable` | Root directory for dynamic training outputs |

### Static SAC arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--task-mode` | `cycle` | Multi-task selection mode: `cycle` or `random` |
| `--batch-size` | `960` | SAC batch size |
| `--reward-clip` | `None` | Optional clipping bound for the final environment reward |
| `--default-penalty-weight` | `0.0` | Penalty for moving away from the default parameter value |
| `--default-close-radius` | `0.6` | Radius used to identify actions close to the default parameter |
| `--default-close-signed-weight` | `0.0` | Signed shaping weight for actions close to the default parameter |
| `--final-only-after-episode` | `None` | Episode threshold after which only the final reward is assigned |
| `--log-dir` | `RL/runs/Static/paper_state_sac_sop_f1` | Output directory for a static training run |

### Static TD3 arguments

Static TD3 accepts all shared and static-environment arguments above, in addition to:

| Argument | Default | Description |
| --- | ---: | --- |
| `--hidden-dims` | `128,128` | Actor and critic hidden-layer dimensions |
| `--actor-lr` | `3e-4` | Actor learning rate |
| `--critic-lr` | `3e-4` | Critic learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--tau` | `0.005` | Target-network soft-update coefficient |
| `--exploration-noise` | `0.1` | Standard deviation of action noise used during exploration |
| `--policy-noise` | `0.2` | Standard deviation of target-policy smoothing noise |
| `--noise-clip` | `0.5` | Clipping bound for target-policy noise |
| `--policy-delay` | `2` | Number of critic updates per delayed actor update |

Run a script with `--help` for the complete and current argument list.

## Output Files

A training run normally produces the following files:

```text
<run-directory>/
├── before_eval.json       # Evaluation before training
├── after_eval.json        # Evaluation after training
├── run_config.json        # Arguments, problem names, and random seed
├── summary.json           # Summary metrics
├── reward_curve.csv       # Reward and best-fitness curve data
├── reward_curve.png       # Training curves, when matplotlib is available
├── train_history.csv      # Episode-level training history
├── eval_history.csv       # Evaluation history
└── checkpoints/
    ├── latest.pt
    ├── best_fitness.pt
    ├── best_reward.pt
    └── episode_<N>.pt
```

Dynamic multi-problem and static batch training also produce aggregate CSV and JSON files in the output root. Multi-seed static training produces `aggregate_seed_results.csv` and `aggregate_seed_results.json`.

## Evaluating Trained Models

Inspect the dynamic SAC evaluation interface with:

```powershell
python .\RL\Dynamic\test_sac_dynamic.py --help
```

Inspect the static SAC evaluation interface with:

```powershell
python .\RL\Static\test__sac_static.py --help
```

Pass the generated checkpoint and its matching problem configuration to the evaluation script. The evaluation `dimension`, `population-size`, `max-fe`, observation design, and neural-network architecture must be consistent with training.

## Reproducibility Recommendations

- Retain `run_config.json` with every result to record the complete configuration and random seed.
- Keep the problem, dimension, population size, evaluation budget, and evaluation seeds fixed when comparing methods.
- Use multiple random seeds for final experiments, and report means, standard deviations, and per-seed results.
- Install the pinned ELA stack from `requirements.txt` and record `ela_implementation` from each run's `run_config.json`.
- Do not commit large checkpoints and generated training logs directly to Git. Consider GitHub Releases, Git LFS, or external storage when model files need to be published.

## License

This project is released under the [MIT License](LICENSE).
