#!/usr/bin/env python3
"""Aggregate TVT campaign output into seed-level and paired tables.

The independent experimental unit is a seed, never a decision row. A campaign
produces thousands of step rows per episode; treating those as independent
samples would be pseudo-replication, so this tool only ever aggregates upward:

    raw step  ->  episode summary  ->  seed-level summary  ->  paired table

Training and evaluation output live in separate subtrees and are never mixed:
training episodes run with epsilon exploration and an active optimizer, so they
cannot serve as a performance measurement.

Concurrent service scenarios in a campaign compare identical ns-3 physical
traffic under different *declared* requirements ("heterogeneous concurrent QoS
requirements over a shared, RAT-switchable bearer"). They are not concurrent
measured flows: ns-3 measures a single ToD KPI cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Metrics carried from episode_summary.csv. Measured = ToD KPI cohort;
# declared = scored against QOS_THRESHOLDS; diagnostic = learning/system.
METRIC_KINDS: Dict[str, str] = {
    "cumulative_reward": "measured-composite",
    "average_throughput_bps": "measured",
    "average_delay_ms": "measured",
    "p95_delay_ms": "measured",
    "deadline_loss_rate": "measured",
    "outage_duration_ms": "measured",
    "qos_satisfaction_ratio": "declared-requirement",
    "switch_count": "measured-control-cost",
    "pingpong_count": "measured-control-cost",
    "mean_upper_grad_norm": "diagnostic",
    "mean_lower_grad_norm": "diagnostic",
    "mean_total_grad_norm": "diagnostic",
    "mean_train_time_ms": "diagnostic",
    "mean_recv_wait_ms": "diagnostic",
    "pending_expired_count": "diagnostic",
    "pending_overflow_count": "diagnostic",
    "unmatched_outcome_count": "diagnostic",
}

# Metrics whose numeric scale is defined by the reward function itself. Two
# configs trained under different reward_version values put these on different
# scales, so pairing them is a category error rather than a comparison. Every
# other metric is a measured task-level outcome and stays comparable across
# reward families -- that is the whole point of the task-level protocol.
REWARD_FAMILY_METRICS = {"cumulative_reward"}

# The four preregistered primary comparisons. Anything else this tool can emit
# is exploratory: the automatic all-combinations enumeration must never be used
# as the paper's primary table.
PRIMARY_PAIRS: List[Tuple[str, str]] = [
    ("goal-free-ddqn__clip5",
     "e2e-transformer__ctx-multi__flow-legacy__clip5"),                  # RQ1
    ("e2e-transformer__ctx-none__flow-legacy__clip5",
     "e2e-transformer__ctx-multi__flow-legacy__clip5"),                  # RQ2
    ("e2e-transformer__ctx-multi__flow-legacy__clip5",
     "e2e-transformer__ctx-multi__flow-video-safety__clip5"),            # RQ3
    ("goal-ddqn__planner-rule__svc-default__clip1",
     "e2e-transformer__ctx-multi__flow-legacy__clip5"),                  # RQ4
]

EPISODE_DIR = re.compile(r"^ep(\d+)$")
SEED_DIR = re.compile(r"^seed_(\d+)$")
TRAIN_SEED_DIR = re.compile(r"^train_seed_(\d+)$")
EVAL_SEED_DIR = re.compile(r"^eval_seed_(\d+)$")


class AggregationError(Exception):
    pass


# --------------------------------------------------------------------- input
def _read_single_row(path: Path) -> Tuple[Optional[Dict[str, str]], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (rows[0] if rows else None), len(rows)


def _read_config(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AggregationError(f"unreadable config.json: {path} ({exc})") from exc


def _number(value: str, field: str, source: Path, allow_nonfinite: bool):
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return value
    if not math.isfinite(parsed):
        message = f"non-finite {field}={value!r} in {source}"
        if not allow_nonfinite:
            raise AggregationError(message)
        return None
    return parsed


def collect_runs(campaign: Path, allow_nonfinite: bool) -> Tuple[List[dict], List[str]]:
    """Walk the campaign tree and return one record per completed run."""
    problems: List[str] = []
    runs: List[dict] = []

    for phase in ("train", "eval"):
        phase_root = campaign / phase
        if not phase_root.is_dir():
            continue
        for config_dir in sorted(p for p in phase_root.iterdir() if p.is_dir()):
            config_id = config_dir.name
            for seed_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
                if phase == "train":
                    match = SEED_DIR.match(seed_dir.name)
                    if not match:
                        problems.append(f"unexpected directory: {seed_dir}")
                        continue
                    leaves = [
                        (int(EPISODE_DIR.match(p.name).group(1)), None, p)
                        for p in sorted(seed_dir.iterdir())
                        if p.is_dir() and EPISODE_DIR.match(p.name)
                    ]
                    train_seed = int(match.group(1))
                else:
                    match = TRAIN_SEED_DIR.match(seed_dir.name)
                    if not match:
                        problems.append(f"unexpected directory: {seed_dir}")
                        continue
                    train_seed = int(match.group(1))
                    leaves = []
                    for p in sorted(seed_dir.iterdir()):
                        inner = EVAL_SEED_DIR.match(p.name) if p.is_dir() else None
                        if inner:
                            leaves.append((1, int(inner.group(1)), p))
                        elif p.is_dir():
                            problems.append(f"unexpected directory: {p}")

                for episode, eval_seed, leaf in leaves:
                    summary_path = leaf / "episode_summary.csv"
                    config_path = leaf / "config.json"
                    if not summary_path.is_file():
                        problems.append(f"missing episode_summary.csv: {leaf}")
                        continue
                    if not config_path.is_file():
                        problems.append(f"missing config.json: {leaf}")
                        continue
                    row, row_count = _read_single_row(summary_path)
                    if row is None:
                        problems.append(f"empty episode_summary.csv: {leaf}")
                        continue
                    if row_count > 1:
                        problems.append(
                            f"duplicate summary rows ({row_count}) in {summary_path}")
                    config = _read_config(config_path)
                    provenance = config.get("provenance", {})
                    record = {
                        "phase": phase,
                        "config_id": config_id,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "episode": episode,
                        "run_dir": str(leaf),
                        "architecture": row.get("architecture")
                        or provenance.get("architecture"),
                        "config_architecture": provenance.get("architecture"),
                        "config_hash": provenance.get("config_hash"),
                        "config_config_id": provenance.get("config_id"),
                        "reward_version": row.get("reward_version")
                        or provenance.get("reward_version"),
                        "ns3_rng_run": provenance.get("ns3_rng_run"),
                        "seed": provenance.get("seed"),
                    }
                    for metric in METRIC_KINDS:
                        record[metric] = _number(
                            row.get(metric), metric, summary_path, allow_nonfinite)
                    runs.append(record)
    return runs, problems


# ---------------------------------------------------------------- validation
def validate(campaign: Path, runs: List[dict], problems: List[str]) -> dict:
    manifest_path = campaign / "campaign_manifest.json"
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            problems.append(f"unreadable manifest: {exc}")

    checks: Dict[str, Any] = {
        "missing_episodes": [],
        "missing_seeds": [],
        "duplicate_runs": [],
        "architecture_mismatch": [],
        "config_hash_mismatch": [],
        "config_id_mismatch": [],
    }

    seen: Dict[tuple, List[str]] = defaultdict(list)
    for run in runs:
        key = (run["phase"], run["config_id"], run["train_seed"],
               run["eval_seed"], run["episode"])
        seen[key].append(run["run_dir"])
    for key, directories in seen.items():
        if len(directories) > 1:
            checks["duplicate_runs"].append(
                {"key": list(key), "run_dirs": directories})

    # episode continuity per training seed
    by_seed: Dict[tuple, List[int]] = defaultdict(list)
    for run in runs:
        if run["phase"] == "train":
            by_seed[(run["config_id"], run["train_seed"])].append(run["episode"])
    expected = manifest.get("episodes_per_seed") if manifest else None
    for (config_id, seed), episodes in sorted(by_seed.items()):
        episodes = sorted(episodes)
        highest = expected if expected else max(episodes)
        gaps = [n for n in range(1, highest + 1) if n not in episodes]
        if gaps:
            checks["missing_episodes"].append(
                {"config_id": config_id, "seed": seed, "expected": highest,
                 "missing": gaps})

    if manifest:
        declared_train = set(manifest.get("train_seeds") or [])
        found_train = {
            run["train_seed"] for run in runs if run["phase"] == "train"}
        if declared_train - found_train:
            checks["missing_seeds"].append(
                {"phase": "train", "missing": sorted(declared_train - found_train)})
        declared_eval = set(manifest.get("eval_seeds") or [])
        found_eval = {
            run["eval_seed"] for run in runs
            if run["phase"] == "eval" and run["eval_seed"] is not None}
        if declared_eval - found_eval:
            checks["missing_seeds"].append(
                {"phase": "eval", "missing": sorted(declared_eval - found_eval)})

    # a config directory must agree with the recorded provenance
    hashes: Dict[str, set] = defaultdict(set)
    for run in runs:
        if run["config_architecture"] and run["architecture"] \
                and run["config_architecture"] != run["architecture"]:
            checks["architecture_mismatch"].append(
                {"run_dir": run["run_dir"], "summary": run["architecture"],
                 "config": run["config_architecture"]})
        if run["config_config_id"] and run["config_config_id"] != run["config_id"]:
            checks["config_id_mismatch"].append(
                {"run_dir": run["run_dir"], "directory": run["config_id"],
                 "config": run["config_config_id"]})
        if run["config_hash"]:
            hashes[run["config_id"]].add(run["config_hash"])
    for config_id, values in sorted(hashes.items()):
        if len(values) > 1:
            checks["config_hash_mismatch"].append(
                {"config_id": config_id, "hashes": sorted(values)})
    if manifest and manifest.get("config_hash"):
        declared = manifest["config_hash"]
        for config_id, values in sorted(hashes.items()):
            if config_id == manifest.get("config_id") and values != {declared}:
                checks["config_hash_mismatch"].append(
                    {"config_id": config_id, "manifest": declared,
                     "runs": sorted(values)})

    checks["structural_problems"] = problems
    checks["manifest_present"] = manifest is not None
    return checks


# ------------------------------------------------------------------- tables
def seed_level(runs: List[dict], phase: str) -> List[dict]:
    """One row per (config, seed unit). Eval seeds are their own unit."""
    rows: List[dict] = []
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for run in runs:
        if run["phase"] != phase:
            continue
        if phase == "eval":
            key = (run["config_id"], run["train_seed"], run["eval_seed"])
        else:
            key = (run["config_id"], run["train_seed"], None)
        grouped[key].append(run)

    for (config_id, train_seed, eval_seed), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1],
                                           item[0][2] or 0)
    ):
        last = max(group, key=lambda run: run["episode"])
        row = {
            "phase": phase,
            "config_id": config_id,
            "architecture": last["architecture"],
            "reward_version": last["reward_version"],
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "seed_unit": (
                f"train{train_seed:03d}" if eval_seed is None
                else f"train{train_seed:03d}_eval{eval_seed:03d}"),
            "episodes": len(group),
            "last_episode": last["episode"],
            "config_hash": last["config_hash"],
        }
        for metric in METRIC_KINDS:
            values = [
                run[metric] for run in group
                if isinstance(run[metric], (int, float))
            ]
            if phase == "eval":
                # one evaluation episode per unit; keep it verbatim
                row[metric] = values[-1] if values else None
            else:
                # training diagnostics only: last episode plus a mean
                row[metric] = last[metric] if isinstance(
                    last[metric], (int, float)) else None
                row[f"{metric}__mean_over_episodes"] = (
                    sum(values) / len(values) if values else None)
        rows.append(row)
    return rows


def paired_table(seed_rows: List[dict], pairs: Optional[List[Tuple[str, str]]],
                 allow_cross_reward: bool = False):
    """Wide table ready for paired bootstrap / Wilcoxon / effect size."""
    by_config: Dict[str, Dict[str, dict]] = defaultdict(dict)
    families: Dict[str, set] = defaultdict(set)
    for row in seed_rows:
        by_config[row["config_id"]][row["seed_unit"]] = row
        families[row["config_id"]].add(row.get("reward_version") or None)
    configs = sorted(by_config)
    candidates = pairs if pairs else list(combinations(configs, 2))

    out: List[dict] = []
    notes: List[str] = []
    for left, right in candidates:
        if left not in by_config or right not in by_config:
            notes.append(f"pair skipped, config absent: {left} vs {right}")
            continue
        mixed = [name for name in (left, right) if len(families[name]) > 1]
        if mixed:
            notes.append(
                f"pair skipped, more than one reward_version inside "
                f"{mixed}: {left} vs {right}")
            continue
        family_a = next(iter(families[left]))
        family_b = next(iter(families[right]))
        cross_family = family_a != family_b
        if cross_family:
            dropped = sorted(REWARD_FAMILY_METRICS)
            notes.append(
                f"cross reward-family pair {left} ({family_a}) vs {right} "
                f"({family_b}): reward-scale metrics {dropped} "
                + ("kept because --allow-cross-reward-metric was given; the "
                   "resulting differences are NOT a valid comparison"
                   if allow_cross_reward else
                   "dropped; measured task-level metrics remain comparable"))
        left_units = set(by_config[left])
        right_units = set(by_config[right])
        common = sorted(left_units & right_units)
        if not common:
            notes.append(f"pair skipped, no shared seed unit: {left} vs {right}")
            continue
        if left_units ^ right_units:
            notes.append(
                f"unpaired seed units dropped for {left} vs {right}: "
                f"{sorted(left_units ^ right_units)}")
        for unit in common:
            a, b = by_config[left][unit], by_config[right][unit]
            for metric, kind in METRIC_KINDS.items():
                if (cross_family and not allow_cross_reward
                        and metric in REWARD_FAMILY_METRICS):
                    continue
                value_a, value_b = a.get(metric), b.get(metric)
                if not isinstance(value_a, (int, float)) or not isinstance(
                        value_b, (int, float)):
                    continue
                out.append({
                    "seed_unit": unit,
                    "train_seed": a["train_seed"],
                    "eval_seed": a["eval_seed"],
                    "metric": metric,
                    "metric_kind": kind,
                    "architecture_A": left,
                    "architecture_B": right,
                    "reward_version_A": family_a,
                    "reward_version_B": family_b,
                    "cross_reward_family": cross_family,
                    "value_A": value_a,
                    "value_B": value_b,
                    "difference": value_b - value_a,
                })
    return out, notes


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate a TVT campaign into seed-level tables")
    parser.add_argument("campaign_dir")
    parser.add_argument(
        "--phase", choices=("eval", "train"), default="eval",
        help="phase used for the paired table; training episodes carry "
             "epsilon exploration and are never a performance measurement")
    parser.add_argument(
        "--pair", action="append", default=None,
        metavar="CONFIG_A,CONFIG_B", help="restrict paired output to this pair")
    parser.add_argument(
        "--primary-pairs", action="store_true",
        help="use exactly the four preregistered primary comparisons "
             "(RQ1 D-F1, RQ2 E-F1, RQ3 F1-F2, RQ4 C-F1); without an explicit "
             "pair selection the output is exploratory, not a primary table")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allow-cross-reward-metric", action="store_true",
        help="keep reward-scale metrics when the two configs were trained "
             "under different reward_version values; off by default because "
             "those values live on different scales")
    parser.add_argument(
        "--allow-nonfinite", action="store_true",
        help="report instead of failing on NaN/Inf inputs")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="continue when completeness checks fail")
    args = parser.parse_args(argv)

    campaign = Path(args.campaign_dir)
    if not campaign.is_dir():
        print(f"[aggregate][ERROR] not a directory: {campaign}", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir) if args.output_dir else campaign / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        runs, problems = collect_runs(campaign, args.allow_nonfinite)
    except AggregationError as exc:
        print(f"[aggregate][ERROR] {exc}", file=sys.stderr)
        return 3

    checks = validate(campaign, runs, problems)
    train_rows = seed_level(runs, "train")
    eval_rows = seed_level(runs, "eval")
    phase_rows = eval_rows if args.phase == "eval" else train_rows

    if args.primary_pairs and args.pair:
        print("[aggregate][ERROR] --primary-pairs and --pair are exclusive",
              file=sys.stderr)
        return 2
    pairs = None
    if args.primary_pairs:
        pairs = list(PRIMARY_PAIRS)
    if args.pair:
        pairs = []
        for item in args.pair:
            left, _, right = item.partition(",")
            if not left or not right:
                print(f"[aggregate][ERROR] bad --pair value: {item!r}",
                      file=sys.stderr)
                return 2
            pairs.append((left, right))

    paired, pair_notes = paired_table(phase_rows, pairs,
                                      args.allow_cross_reward_metric)

    _write_csv(output_dir / "seed_summary_train.csv", train_rows)
    _write_csv(output_dir / "seed_summary_eval.csv", eval_rows)
    _write_csv(output_dir / "paired_differences.csv", paired)

    blocking = [
        name for name in ("missing_episodes", "missing_seeds",
                          "duplicate_runs", "architecture_mismatch",
                          "config_hash_mismatch", "config_id_mismatch")
        if checks[name]
    ]
    if problems:
        blocking.append("structural_problems")

    report = {
        "campaign_dir": str(campaign),
        "paired_phase": args.phase,
        "runs_found": len(runs),
        "train_seed_units": len(train_rows),
        "eval_seed_units": len(eval_rows),
        "paired_rows": len(paired),
        "reward_family_metrics": sorted(REWARD_FAMILY_METRICS),
        "allow_cross_reward_metric": args.allow_cross_reward_metric,
        "cross_reward_family_rows": sum(
            1 for row in paired if row.get("cross_reward_family")),
        "primary_table": bool(args.primary_pairs or args.pair),
        "pairs_used": ("preregistered primary pairs" if args.primary_pairs
                       else "explicit --pair selection" if args.pair
                       else "automatic all-combinations (EXPLORATORY ONLY)"),
        "primary_pairs": [list(pair) for pair in PRIMARY_PAIRS],
        "reward_versions": sorted(
            {row["reward_version"] for row in phase_rows
             if row.get("reward_version")}),
        "configs": sorted({row["config_id"] for row in phase_rows}),
        "checks": checks,
        "pair_notes": pair_notes,
        "blocking_checks": blocking,
        "independent_unit": (
            "one seed unit per row; decision/step rows are never treated as "
            "independent samples"),
        "phase_isolation": (
            "train/ and eval/ subtrees are read separately and never merged"),
        "scenario_semantics": (
            "concurrent scenarios differ only in declared QoS requirements "
            "over a shared, RAT-switchable bearer; ns-3 physical traffic is "
            "identical and a single ToD KPI cohort is measured"),
        "metric_kinds": METRIC_KINDS,
    }
    (output_dir / "aggregate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not (args.primary_pairs or args.pair):
        print("[aggregate][WARNING] no pair selection given: the paired table "
              "is the automatic all-combinations enumeration and is "
              "EXPLORATORY ONLY. Use --primary-pairs for the primary table.",
              file=sys.stderr)
    print(f"[aggregate] runs={len(runs)} "
          f"train_units={len(train_rows)} eval_units={len(eval_rows)} "
          f"paired_rows={len(paired)}")
    print(f"[aggregate] output={output_dir}")
    for name in blocking:
        print(f"[aggregate][CHECK-FAIL] {name}: "
              f"{json.dumps(checks[name])[:300]}", file=sys.stderr)
    if not phase_rows:
        print(f"[aggregate][ERROR] no runs for phase={args.phase!r}; "
              f"train_units={len(train_rows)} eval_units={len(eval_rows)}",
              file=sys.stderr)
        return 4
    if blocking and not args.allow_incomplete:
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
