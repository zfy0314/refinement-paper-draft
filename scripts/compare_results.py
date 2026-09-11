"""Compute requested evaluation comparisons and a six-test Holm correction.

Car Racing uses rounded manuscript summaries, not paired episode outcomes.
Door Opening uses episode minima paired by configuration and sampling seed.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sections/04_experiments.tex"
N = 10


def welch(rows, first, second, label):
    mean_a, sd_a = rows[first]
    mean_b, sd_b = rows[second]
    # The manuscript reports population SD; t tests require sample variance.
    var_a, var_b = np.square([sd_a, sd_b]) * N / (N - 1)
    se = np.sqrt((var_a + var_b) / N)
    df = (var_a + var_b) ** 2 / ((var_a ** 2 + var_b ** 2) / (N - 1))
    result = stats.ttest_ind_from_stats(
        mean_a, np.sqrt(var_a), N, mean_b, np.sqrt(var_b), N,
        equal_var=False, alternative="two-sided")
    delta = mean_a - mean_b
    radius = stats.t.ppf(0.975, df) * se
    return dict(label=label, method="Welch approximation from rounded summaries",
                first=first, second=second, n_per_policy=N,
                mean_difference=delta, t=float(result.statistic), df=float(df),
                p=float(result.pvalue), ci95=[delta - radius, delta + radius])


def main():
    rows = {}
    for line in SOURCE.read_text().splitlines():
        match = re.search(r"^([^&]+)&.*\$([\d.]+)\\pm([\d.]+)\$", line)
        if match:
            rows[match[1].strip()] = tuple(map(float, match.group(2, 3)))
    comparisons = [
        welch(rows, "Refined structure + IL", "Initial structure + IL", "Car: refined vs. initial, IL"),
        welch(rows, "Refined structure + RL, low compute", "Initial structure + RL, low compute", "Car: refined vs. initial, low RL"),
        welch(rows, "Refined structure + IL", "MLP + IL", "Car: refined vs. MLP, IL"),
        welch(rows, "Refined structure + RL, low compute", "MLP + RL, low compute", "Car: refined vs. MLP, low RL"),
        welch(rows, "Refined structure + RL, high compute", "MLP + RL, high compute", "Car: refined vs. MLP, high RL"),
    ]
    paths = [ROOT / f"data/verification/2026-09-09/door_IL_door_round{i}_sampled.json" for i in (0, 3)]
    episodes = [json.loads(p.read_text())["episodes"] for p in paths]
    maps = [{e["seed"]: e for e in group} for group in episodes]
    assert len(maps[0]) == len(maps[1]) == N
    assert maps[0].keys() == maps[1].keys()
    seeds = sorted(maps[0])
    for seed in seeds:
        a, b = maps[0][seed], maps[1][seed]
        for key in ("initial_handle", "target", "initial_observation", "torch_seed"):
            np.testing.assert_array_equal(a[key], b[key])
        for episode in (a, b):
            assert episode["min_distance"] == min(episode["distances"])
    before, after = [np.array([group[s]["min_distance"] for s in seeds]) for group in maps]
    result = stats.ttest_rel(before, after)
    comparisons.insert(2, dict(
        label="Door: iteration 0 vs. 3", method="Paired t test of episode-minimum distance",
        n_pairs=N, seeds=seeds, mean_difference=float((before - after).mean()),
        t=float(result.statistic), df=int(result.df), p=float(result.pvalue),
        ci95=list(result.confidence_interval()),
        before=before.tolist(), after=after.tolist()))
    order = np.argsort([row["p"] for row in comparisons])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = max(adjusted, min(1.0, comparisons[index]["p"] * (len(order) - rank)))
        comparisons[index]["p_holm"] = adjusted
    destination = ROOT / "data/statistics"
    destination.mkdir(exist_ok=True)
    payload = dict(
        scope="Evaluation of fitted policies; not independent training or refinement runs",
        car_assumptions="n=10 per policy; population SD; independent-sample approximation; cross-policy covariance unavailable",
        holm_family="All six requested non-MLES comparisons; two-sided tests",
        car_summary_rows=rows,
        door_sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        comparisons=comparisons)
    (destination / "requested_comparisons.json").write_text(json.dumps(payload, indent=2) + "\n")
    for row in comparisons:
        print(f"{row['label']}: delta={row['mean_difference']:.6g}, p={row['p']:.9g}, Holm={row['p_holm']:.9g}")


if __name__ == "__main__":
    main()
