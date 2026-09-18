"""Summarise an ablation sweep: mean, spread and a permutation p against control.

    python tools/ablation_report.py results/ablation

Three metrics, because they do not all carry the same information. The
soft-landing rate is what you care about but it swings wildly between seeds.
Touchdown speed and timeout rate are far steadier and say *how* a variant fails,
which is usually the thing that identifies the missing reward term.
"""
import collections
import glob
import itertools
import json
import statistics
import sys

ORDER = ["abl_full", "abl_no_time", "abl_no_gate", "abl_no_fuel",
         "abl_no_fuel_no_time", "abl_flat_speed", "abl_no_speed_term",
         "abl_banded_shaping", "abl_no_shaping", "abl_original_reward"]

LABEL = {
    "abl_full": "FULL REWARD (control)",
    "abl_no_time": "minus time penalty",
    "abl_no_gate": "minus landing speed gate",
    "abl_no_fuel": "minus fuel penalty",
    "abl_no_fuel_no_time": "minus fuel AND time",
    "abl_flat_speed": "minus approach gating",
    "abl_no_speed_term": "minus speed term in phi",
    "abl_banded_shaping": "banded shaping instead",
    "abl_no_shaping": "no shaping at all",
    "abl_original_reward": "THE ORIGINAL REWARD",
}


def permutation_p(a, b):
    """Two-sided p for a difference in means, by exact relabelling.

    No distributional assumption, which is what a handful of seeds deserves.
    Note the floor: with n_a and n_b seeds the smallest reachable p is
    1 / C(n_a + n_b, n_a), so p = 0.036 at 2-vs-6 means "as separated as this
    sample size can show", not "weak".
    """
    observed = abs(statistics.mean(a) - statistics.mean(b))
    pool = list(a) + list(b)
    hits = 0
    splits = list(itertools.combinations(range(len(pool)), len(a)))
    for idx in splits:
        left = [pool[i] for i in idx]
        right = [pool[i] for i in range(len(pool)) if i not in idx]
        if abs(statistics.mean(left) - statistics.mean(right)) >= observed - 1e-12:
            hits += 1
    return hits / len(splits)


def impact_speeds(runs):
    ## None when a policy never reached the pad at all. Dropped rather than
    ## zero-filled: "never touched down" is not "touched down gently".
    return [r["mean_impact_speed"] for r in runs if r["mean_impact_speed"] is not None]


def timeout_rates(runs):
    return [r["outcomes"].get("timeout", 0) / sum(r["outcomes"].values()) for r in runs]


def main(results_dir):
    runs = collections.defaultdict(list)
    for path in sorted(glob.glob(results_dir + "/abl_*.json")):
        record = json.load(open(path))
        runs[record["name"]].append(record)
    if "abl_full" not in runs:
        raise SystemExit("no abl_full control runs found in " + results_dir)

    control = runs["abl_full"]
    baselines = {
        "soft": [r["soft_landing_rate"] for r in control],
        "impact": impact_speeds(control),
        "timeout": timeout_rates(control),
    }

    def cell(values):
        if not values:
            return "     n/a"
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        return f"{statistics.mean(values):.2f}+-{sd:.2f}"

    print(f"{'variant':<26} {'n':>2} | {'soft land':>12} {'p':>6} | "
          f"{'touchdown':>12} {'p':>6} | {'timeout':>12} {'p':>6}")
    print("-" * 104)
    for name in ORDER:
        group = runs.get(name)
        if not group:
            continue
        series = {
            "soft": [r["soft_landing_rate"] for r in group],
            "impact": impact_speeds(group),
            "timeout": timeout_rates(group),
        }
        if name == "abl_full":
            ps = {k: "" for k in series}
        else:
            ps = {k: (f"{permutation_p(series[k], baselines[k]):.3f}" if series[k] else "   n/a")
                  for k in series}
        print(f"{LABEL[name]:<26} {len(group):>2} | "
              f"{cell(series['soft']):>12} {ps['soft']:>6} | "
              f"{cell(series['impact']):>12} {ps['impact']:>6} | "
              f"{cell(series['timeout']):>12} {ps['timeout']:>6}")

    print("\noutcome mix, pooled over seeds:")
    for name in ORDER:
        group = runs.get(name)
        if not group:
            continue
        pooled = collections.Counter()
        for record in group:
            pooled.update(record["outcomes"])
        total = sum(pooled.values())
        mix = "  ".join(f"{k}={v / total:.2f}" for k, v in pooled.most_common(4))
        print(f"  {LABEL[name]:<26} {mix}")

    print("\nsoft-landing rate every 500k steps (mean over seeds):")
    for name in ORDER:
        group = [r for r in runs.get(name, []) if r.get("curve")]
        if not group:
            continue
        points = min(len(r["curve"]) for r in group)
        curve = [statistics.mean(r["curve"][i]["soft_landing_rate"] for r in group)
                 for i in range(points)]
        print(f"  {LABEL[name]:<26} " + " ".join(f"{p:5.2f}" for p in curve))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/ablation")
