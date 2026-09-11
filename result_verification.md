# Verification of the reported Car Racing and Door Opening results

Verified on 2026-09-09 by rerunning archived checkpoints with the original
`utils.py` evaluator in `/Users/zfy/miniconda3/envs/IPR/bin/python`.
No matching archived per-seed performance summaries were found. The diagnostic
CSVs omit rewards; `count.py` preserves the aggregate Car Racing plot arrays.
All measurements below are new inference runs, not recovered historical output.
No model was trained and no LLM request was made. Source logs were not edited.

## Car Racing: exact match is the January round-4 checkpoint

- Model: `/Users/zfy/Projects/meta/llm-log-analysis/logs/IL_racecar/model_racecar_iter004.py`
- Weights: `/Users/zfy/Projects/meta/llm-log-analysis/logs/IL_racecar/iteration_04/weights_final.pth`
- Both files are byte-identical to the counterparts under `logs/IL_01172209`.
- Evaluation: mean actions (`sample=False`), environment seeds 1000–1009,
  at most 1,000 steps; PyTorch seed 0 per episode; CPU.
- Return: **881.988341797095 ± 29.331464175040** (population standard deviation).
  This reproduces **881.99 ± 29.33**, hence **882.0 ± 29.33** in the table.

`IL_11130120` and `IL_11130143` are consecutive parts of an earlier sequence:
rounds 0–2 and 3–5. Prompt-embedded code and the recorded video filenames link
`IL_11130120/model_iter003.py` to `IL_11130143/iteration_03/weights_final.pth`.
None of their six final checkpoints matches the target under the same evaluation:

| November round | Mean return | Population standard deviation |
|---|---:|---:|
| 0 | 762.580792 | 55.047364 |
| 1 | 869.282945 | 33.434779 |
| 2 | 889.810474 | 24.845078 |
| 3 | 892.390880 | 22.082669 |
| 4 | 890.912606 | 22.225455 |
| 5 | 800.369837 | 51.287642 |

Sampled actions were also checked for November rounds 3 and 4, yielding
599.747648 ± 32.437748 and 597.392182 ± 32.382685 respectively. These do not match.
The November initial result does reproduce the plot's 762.58 ± 55.05 baseline.
The manuscript table's separate 766.5 ± 100.4 initial row remains to be mapped.

### Refinement counts

The matched January fourth revision follows four diagnostic rollout CSVs and four
pairs of analysis/revision responses in `prompt_iter000.json` through
`prompt_iter003.json`: eight recorded LLM responses, or nine calls including the
initial generation counted by the original protocol. The correct cost for this
verified IL result is therefore **four refinement resets and nine LLM calls**,
excluding final evaluation. The manuscript's IL row is corrected accordingly.

This verification does not identify which refined model initialized each retained
RL result. Those rows retain their historical counts pending a separate checkpoint
mapping; they should not automatically inherit the verified IL row's budget.

## Door Opening: all four displayed results reproduce

The initial code embedded in `logs/IL_11202146/prompt_iter000.json` matches
`models/model_door_iter000.py` by its Python syntax tree. Subsequent models are
`logs/IL_door/model_door_iter001.py` through `model_door_iter003.py`, each with
`logs/IL_door/iteration_0N/weights_final.pth`. The requested round-3 model and
checkpoint are byte-identical to those under `IL_11202209` and `IL_11202259`.

Protocol: `evaluate_policy_door`, **sampled Gaussian actions**, 500 steps,
10 episodes, environment and PyTorch sampling seeds 0–9. All four evaluations
use the same ten distinct initial handle/target configurations.

| Round | Minimum distance | Final distance |
|---|---:|---:|
| 0 | 0.547530 ± 0.000000 | 0.547530 ± 0.000000 |
| 1 | 0.544925 ± 0.007813 | 0.557508 ± 0.012802 |
| 2 | 0.539759 ± 0.022594 | 0.542321 ± 0.023022 |
| 3 | 0.110052 ± 0.059459 | 0.133546 ± 0.049751 |

The four minimum-distance rows reproduce the original figure at its printed
precision. For round 3, mean actions instead give 0.196163 ± 0.029511; the action
selection mode is necessary to reproduce the claim. A second candidate,
`IL_11192141` round 3, was checked with mean actions before the requested model's
sampled-action match was found; its result is retained in the verification data.
The remaining alternate candidates were unnecessary after the exact match.

### Distance definition

For each step **after** `env.step`, the evaluator computes
`np.linalg.norm(base_env.data.geom("handle").xpos - base_env._target_pos)`.
If `_target_pos` is unavailable it falls back to `obs[36:39]`. The reported number
is the minimum of these Euclidean distances within each episode, followed by the
mean and `np.std(..., ddof=0)` across ten episodes. It is not an endpoint metric or
a success rate. The observer wrapper added by the verification script records
these same quantities without changing observations, actions, or termination.

## Reproduction and evidence

The existing IPR environment has PyTorch 2.5.1, NumPy 2.2.6, Gymnasium 1.1.1,
Meta-World 3.0.0, MuJoCo 3.3.7, and box2d-py 2.3.5. The runner uses CPU and one
PyTorch thread per worker. Source/checkpoint hashes, full per-seed returns,
per-step Door distances, initial configurations, settings, and package versions
are in `data/verification/2026-09-09/`. `verification_manifest.json` indexes them
and records the four-round prompt/rollout evidence. It also verifies agreement
with the printed metrics and the January run aliases.

Example commands, from the manuscript repository (use a fresh output filename):

```sh
PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy /Users/zfy/miniconda3/envs/IPR/bin/python scripts/verify_reported_results.py racecar \
  --model /Users/zfy/Projects/meta/llm-log-analysis/logs/IL_racecar/model_racecar_iter004.py \
  --weights /Users/zfy/Projects/meta/llm-log-analysis/logs/IL_racecar/iteration_04/weights_final.pth \
  --output /tmp/car-verification.json

PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy /Users/zfy/miniconda3/envs/IPR/bin/python scripts/verify_reported_results.py door \
  --model /Users/zfy/Projects/meta/llm-log-analysis/logs/IL_door/model_door_iter003.py \
  --weights /Users/zfy/Projects/meta/llm-log-analysis/logs/IL_door/iteration_03/weights_final.pth \
  --sample --output /tmp/door-verification.json
```

The confirmed metric and corrected IL costs are incorporated into the manuscript.
Remaining evidence gaps concern the other comparison rows, the RL initialization
and budget mapping, historical training commands/data, and independent training
replicates. No ablation or performance claim after correcting the old logger is
introduced by this verification.
