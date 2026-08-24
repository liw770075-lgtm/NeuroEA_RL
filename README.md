# NeuroEA-RL

NeuroEA-RL is a research project for learning the parameter configuration of NeuroEA with reinforcement learning. It provides a PyTorch implementation of NeuroEA and supports Soft Actor-Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradient (TD3) agents for learning configuration policies:

- **Dynamic configuration:** SAC produces a complete parameter vector for every NeuroEA generation.
- **Static configuration:** SAC selects the parameters sequentially, after which NeuroEA runs to completion with the resulting fixed configuration.

The current implementation focuses on single-objective optimization and includes `SOP_F1`–`SOP_F10` and `BBOB_F1`–`BBOB_F10` benchmark problems.

## Main Training Entry Points

| Configuration | Agent | Entry point | Agent action | Observation | When parameters are applied |
| --- | --- | --- | --- | --- | --- |
| Dynamic | SAC | `RL/Dynamic/train_sac_dynamic.py` | A complete normalized parameter vector at every step | Search summary, current control parameters, ELA features, and optional task context | Before every NeuroEA generation |
| Static | SAC | `RL/Static/train_sac_static.py` | One normalized parameter at every step | ELA features, previously selected parameters, a selection mask, and optional task context | After all parameters have been selected |
| Static | TD3 | `RL/Static/train_td3_static.py` | One normalized parameter at every step | The same action-history observation used by static SAC | After all parameters have been selected |

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

The static environment divides one complete configuration into a sequence of reinforcement-learning steps. SAC or TD3 selects one scalar parameter at each step, while the observation records the previous choices and the selected-parameter mask.

After all parameters have been selected, NeuroEA runs to completion with the fixed parameter vector. The final reward is the logarithmic improvement from the initial best-fitness gap to the final best-fitness gap. This reward is then assigned back to the parameter-selection transitions. Optional reward shaping relative to the default configuration is also supported.

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
    │   └── action_history_env.py
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

> `scipy` and `scikit-learn` are required to calculate ELA features. If these dependencies cannot be imported, the environment falls back to an all-zero ELA vector. Training may continue, but the experimental setting will no longer be equivalent.

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

### Dynamic SAC training

```powershell
python .\RL\Dynamic\train_sac_dynamic.py `
  --problem-names SOP_F1 `
  --episodes 10000 `
  --population-size 100 `
  --dimension 10 `
  --max-fe 10000 `
  --device cpu `
  --output-root RL/runs/dynamic/sac_sopf1
```

The dynamic entry point accepts comma-separated problem names and inclusive problem ranges. A separate model is trained for every problem:

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

### Static SAC training

```powershell
python .\RL\Static\train_sac_static.py `
  --problem-names SOP_F1 `
  --episodes 10000 `
  --population-size 100 `
  --dimension 10 `
  --max-fe 10000 `
  --device cpu `
  --log-dir RL/runs/Static/action_history_sopf1
```

### Static TD3 training

Static TD3 uses the same action-history environment and sequential reward assignment as static SAC, but replaces the stochastic SAC policy with a deterministic TD3 policy and target-policy smoothing:

```powershell
python .\RL\Static\train_td3_static.py `
  --problem-names SOP_F1 `
  --episodes 10000 `
  --population-size 100 `
  --dimension 10 `
  --max-fe 10000 `
  --device cpu `
  --exploration-noise 0.1 `
  --policy-noise 0.2 `
  --noise-clip 0.5 `
  --policy-delay 2 `
  --log-dir RL/runs/Static/action_history_td3_sopf1
```

The static entry point accepts comma-separated problem names. A single SAC model can be trained across multiple tasks using either cyclic or random task selection:

```powershell
python .\RL\Static\train_sac_static.py `
  --problem-names "SOP_F1,SOP_F2,SOP_F3" `
  --task-mode cycle `
  --include-task-context `
  --episodes 10000 `
  --log-dir RL/runs/Static/multitask_sop
```

### Train an independent static model for each problem

```powershell
python .\RL\Static\batch_train_sac_static.py `
  --problem-names SOP_F1-SOP_F10 `
  --episodes 1000 `
  --output-root RL/runs/Static/sop_f1_f10
```

### Multi-seed static experiments

```powershell
python .\RL\Static\multi_seed_train_sac_static.py `
  --problem-names SOP_F1 `
  --seeds "0,42,100,2025,3407" `
  --episodes 1000 `
  --output-root RL/runs/Static/sopf1_multi_seed
```

`--seeds` also accepts inclusive integer ranges, for example `--seeds 0-4`.

## Main Arguments

### Shared arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--problem-names` | `SOP_F1` | Problem names, separated by commas when multiple tasks are used |
| `--episodes` | `10000` | Number of training episodes; for dynamic multi-problem training, this is the number per problem |
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
| `--include-task-context` | disabled | Add task identity and normalized task configuration to the observation |

### Dynamic SAC arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--objectives` | `2` | Number of objectives passed to the optimization problem |
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
| `--log-dir` | `RL/runs/Static/action_history_sac_sop_f1` | Output directory for a static training run |

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
- Verify that `scipy` and `scikit-learn` can be imported before running ELA-based experiments.
- Do not commit large checkpoints and generated training logs directly to Git. Consider GitHub Releases, Git LFS, or external storage when model files need to be published.

## License

This repository does not currently include an open-source license. Add an appropriate `LICENSE` file before public distribution or reuse by third parties.
