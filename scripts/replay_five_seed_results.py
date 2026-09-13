"""Replay the revised paper from saved router outputs and endpoint outcomes.

Requires Python 3.10+ and NumPy, without GPUs, model weights, or network access.
Run from a downloaded/extracted release:
  python scripts/replay_five_seed_results.py --artifacts . --output-dir replayed

All reported values are fractions (multiply by 100 for paper percentages).
Recipe selection reads validation scores only. Test outcomes are evaluated after
selection. This numerical separation does not erase the disclosed study history.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = [2026, 2027, 2028, 2029, 2030]
TASKS = ["cub", "grefcoco", "nlvr2", "construction"]
METHODS = ["error", "four_difference", "four_ratio", "gain", "route"]
METRICS = ["validation_score25", "test_fixed_score25", "test_ranked_score25",
           "validation_metric", "test_metric", "validation_call", "test_call"]
RANK_BUDGETS = [0, .05, .10, .15, .20, .25, .50, .75, 1.]
FIXED_BUDGETS = [i / 100 for i in range(101)]
TOL = 1e-12


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_close(actual, expected, context, tolerance=TOL):
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"{context}: replayed={actual!r}, recorded={expected!r}")


def scores_from_logits(raw, target, ranking):
    if target == "four_state":
        if ranking == "log_ratio":
            return raw[:, 1] - raw[:, 2]
        if ranking == "probability_difference":
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            prob = exp / exp.sum(axis=1, keepdims=True)
            return prob[:, 1] - prob[:, 2]
    elif target in ("error", "route"):
        # Stable sigmoid with the original float32 output precision.
        x = raw[:, 0]
        return np.where(x >= 0, 1 / (1 + np.exp(-np.abs(x))),
                        np.exp(-np.abs(x)) / (1 + np.exp(-np.abs(x))))
    elif target == "gain":
        return raw[:, 0]
    raise ValueError((target, ranking))


def saved_scores(run, raw, target, ranking, split):
    # Saved with the original PyTorch CPU scoring operations, preserving exact
    # float32 threshold ties. NumPy independently checks the scoring formula.
    with np.load(run / "replay_scores.npz") as data:
        scores = data[split + "_" + ranking]
    calculated = scores_from_logits(raw, target, ranking)
    if scores.shape != calculated.shape or not np.allclose(scores, calculated, rtol=0, atol=8e-7):
        raise ValueError(f"{run.name}/{split}/{ranking}: raw outputs do not reproduce saved scores")
    return scores


def prepare_curve(scores, small, large, small_correct, large_correct):
    """Sufficient statistics for routing the first k items in stable score order."""
    if len(scores) != len(small) or small.shape != large.shape:
        raise ValueError("Router output and endpoint outcome dimensions disagree")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite saved router output")
    order = np.argsort(-scores, kind="stable")
    if not (small.ndim == 1 or (small.ndim == 3 and small.shape[1:] == (4, 3))):
        raise ValueError(f"Unexpected endpoint outcomes shape: {small.shape}")
    # Correctness labels are separate from F1 sufficient statistics: a failed
    # prediction can have an all-zero TP/FP/FN triple yet be marked incorrect.
    s, l = small_correct, large_correct
    if s.shape != l.shape or len(s) != len(scores) or s.ndim != 2:
        raise ValueError("Endpoint correctness label dimensions disagree")
    rescue, harm = (s == 0) & (l == 1), (s == 1) & (l == 0)
    delta = large - small
    cumulative = np.concatenate([np.zeros_like(delta[:1]), np.cumsum(delta[order], axis=0)])
    events = np.stack([rescue.sum(1), harm.sum(1), (s == l).sum(1)], axis=1)
    event_cumulative = np.concatenate([np.zeros((1, 3), dtype=np.int64),
                                      np.cumsum(events[order], axis=0)])
    return {"scores": scores[order], "base": small.sum(axis=0),
            "delta": cumulative, "events": event_cumulative,
            "total_rescue": int(rescue.sum()), "n": len(scores)}


def point_at_count(prepared, count, budget):
    total = prepared["base"] + prepared["delta"][count]
    if np.ndim(total) == 0:
        metric = float(total / prepared["n"])
    else:
        tp, fp, fn = total.T
        den = 2 * tp + fp + fn
        metric = float(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0).mean())
    rescue, harm, wasted = prepared["events"][count]
    return {"budget": budget, "actual_fraction": count / prepared["n"], "metric": metric,
            "rescue": int(rescue), "harm": int(harm),
            "missed_rescue": prepared["total_rescue"] - int(rescue), "wasted_events": int(wasted)}


def ranked_curve(prepared):
    return [point_at_count(prepared, int(np.floor(prepared["n"] * b)), b) for b in RANK_BUDGETS]


def score25(curve):
    return float(sum((a["metric"] + b["metric"]) * .5 * (b["budget"] - a["budget"])
                     for a, b in zip(curve[:5], curve[1:6])) / .25)


def validation_thresholds(prepared):
    ordered, n = prepared["scores"], prepared["n"]
    thresholds = []
    for budget in FIXED_BUDGETS:
        count = int(np.floor(n * budget))
        if count == 0:
            threshold, operator = None, "never"
        elif count == n:
            threshold, operator = None, "always"
        else:
            threshold = float(ordered[count - 1])
            count_ge = int(np.searchsorted(-ordered, -threshold, side="right"))
            operator = "ge" if count_ge <= count else "gt"
        thresholds.append({"budget": budget, "threshold": threshold, "operator": operator})
    return thresholds


def fixed_curve(prepared, thresholds):
    curve = []
    for item in thresholds:
        op = item["operator"]
        if op == "never":
            count = 0
        elif op == "always":
            count = prepared["n"]
        elif op in ("ge", "gt"):
            count = int(np.searchsorted(-prepared["scores"], -item["threshold"],
                                        side="right" if op == "ge" else "left"))
        else:
            raise ValueError(op)
        point = point_at_count(prepared, count, item["budget"])
        point.pop("wasted_events")
        curve.append(dict(point, **{"threshold": item["threshold"], "operator": op}))
    return curve


def verify_curve(actual, recorded, context):
    if len(actual) != len(recorded):
        raise ValueError(f"{context}: curve length mismatch")
    for i, (a, b) in enumerate(zip(actual, recorded)):
        for field in ("budget", "metric", "actual_fraction", "rescue", "harm", "missed_rescue"):
            check_close(a[field], b[field], f"{context}[{i}].{field}")
        if "wasted_events" in b:
            check_close(a["wasted_events"], b["wasted_events"], f"{context}[{i}].wasted_events")
        if "operator" in b:
            if a["operator"] != b["operator"]:
                raise ValueError(f"{context}[{i}]: threshold tie operator changed")
            if a["threshold"] is None or b["threshold"] is None:
                if a["threshold"] != b["threshold"]:
                    raise ValueError(f"{context}[{i}]: threshold endpoint changed")
            else:
                check_close(a["threshold"], b["threshold"], f"{context}[{i}].threshold")


def selected_index(curve):
    # Exactly the evaluator's rule: metric, fewer calls, then smaller budget.
    # No approximate tie is inserted into experimental threshold selection.
    return min(range(1, 100), key=lambda i: (-curve[i]["metric"],
                                          curve[i]["actual_fraction"], curve[i]["budget"]))


def aggregate_objectives(primary, metrics):
    output = []
    for method in METHODS:
        row = {"method": method}
        for metric in metrics:
            values = []
            for seed in SEEDS:
                task_means = []
                for task in TASKS:
                    group = [r[metric] for r in primary if r["method"] == method
                             and r["seed"] == seed and r["task"] == task]
                    if len(group) != 4:
                        raise ValueError((method, seed, task, "expected four model pairs", len(group)))
                    task_means.append(float(np.mean(group)))
                values.append(float(np.mean(task_means)))
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values, ddof=1))
        row["validation_mean_minus_std"] = row["validation_score25_mean"] - row["validation_score25_std"]
        output.append(row)
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("replayed"))
    args = parser.parse_args()
    root = args.artifacts / "experiment_data"
    paper = read_json(args.artifacts / "paper_results" / "data" / "results.json")
    lock = read_json(root / "method_lock_five_seeds.json")
    runs = sorted((root / "runs").glob("repeat_*"))
    if len(runs) != 400 or lock["repeat_seeds"] != SEEDS:
        raise ValueError("The complete release requires 400 runs and seeds 2026–2030")
    outcomes, validation_rows, replay, configurations, confidence_scores = {}, [], {}, {}, {}
    for run in runs:
        cfg = read_json(run / "config.json")
        configurations[run.name] = cfg
        pair = f"shared_12_{cfg['expert']}_{cfg['endpoint']}"
        if pair not in outcomes:
            outcomes[pair] = {}
            confidence_scores[pair] = {}
            for split in ("val", "test"):
                with np.load(root / "pairs" / pair / (split + ".npz")) as data:
                    outcomes[pair][split] = (data["small"], data["large"],
                                             data["small_correct"], data["large_correct"])
                    confidence_scores[pair][split] = np.where(data["confidence_valid"], -data["confidence"], 1.)
        with np.load(run / "paired_logits.npz") as data:
            raw = data["validation"]
        rankings = ["log_ratio", "probability_difference"] if cfg["target"] == "four_state" else [cfg["ranking"]]
        done = read_json(run / "completion.json")
        if done["status"] != "validation_complete" or done["test_loaded"]:
            raise ValueError(f"{run.name}: invalid training-completion evidence")
        replay[run.name] = {}
        for ranking in rankings:
            scores = saved_scores(run, raw, cfg["target"], ranking, "validation")
            prepared = prepare_curve(scores, *outcomes[pair]["val"])
            thresholds = validation_thresholds(prepared)
            curve = fixed_curve(prepared, thresholds)
            area = score25(ranked_curve(prepared))
            best = selected_index(curve)
            replay[run.name][ranking] = {"validation_score25": area, "validation_curve": curve,
                                        "validation_selected": curve[best], "best": best}
            if ranking == cfg["ranking"]:
                check_close(area, done["validation_selection_score"], f"{run.name}: training validation Score25")
                validation_rows.append({"task": cfg["task"], "expert": cfg["expert"],
                                        "endpoint": cfg["endpoint"], "method": cfg["method"],
                                        "seed": cfg["seed"], "validation_score25": area})
    if len({(r["expert"], r["endpoint"], r["method"], r["seed"]) for r in validation_rows}) != 400:
        raise ValueError("Duplicate or missing five-seed run")
    validation_summary = aggregate_objectives(validation_rows, ["validation_score25"])
    selected = min(validation_summary, key=lambda r: (-r["validation_mean_minus_std"],
                                                     r["method"] != "four_ratio", r["method"]))["method"]
    if selected != lock["selected_method"]:
        raise ValueError(f"Validation-only recipe selection disagrees: {selected}")
    for summary in validation_summary:
        expected = next(r for r in lock["comparison"] if r["method"] == summary["method"])
        for field, recorded in [("validation_score25_mean", "validation_mean"),
                                ("validation_score25_std", "validation_std"),
                                ("validation_mean_minus_std", "selection_value")]:
            check_close(summary[field], expected[recorded], f"method selection {summary['method']} {field}")
    print(f"Validation-only recipe selection verified: {selected}", flush=True)

    per_seed, curve_points = [], 0
    for run_index, run in enumerate(runs):
        cfg = configurations[run.name]
        pair = f"shared_12_{cfg['expert']}_{cfg['endpoint']}"
        ev = read_json(run / "test_evaluation.json")
        if ev["config"] != cfg:
            raise ValueError(f"{run.name}: evaluation configuration differs")
        with np.load(run / "paired_logits.npz") as data:
            raw = data["test"]
        for ranking, result in replay[run.name].items():
            expected = ev["comparisons"][ranking]
            scores = saved_scores(run, raw, cfg["target"], ranking, "test")
            prepared = prepare_curve(scores, *outcomes[pair]["test"])
            curve = fixed_curve(prepared, result["validation_curve"])
            exact = ranked_curve(prepared)
            result["test_curve"] = curve
            result["test_selected"] = curve[result["best"]]
            result["test_ranked_score25"] = score25(exact)
            result["test_fixed_score25"] = sum((curve[i]["metric"] + curve[i + 5]["metric"]) * .5 * .05
                                                for i in range(0, 25, 5)) / .25
            for field, points in [("validation_curve", result["validation_curve"]),
                                  ("test_curve", curve), ("test_exact_budget_diagnostic", exact)]:
                verify_curve(points, expected[field], f"{run.name}/{ranking}/{field}")
                curve_points += len(points)
            for field in ("validation_selected", "test_selected"):
                verify_curve([result[field]], [expected[field]], f"{run.name}/{ranking}/{field}")
            for field in ("validation_score25", "test_fixed_score25", "test_ranked_score25"):
                check_close(result[field], expected[field], f"{run.name}/{ranking}/{field}")
            check_close(point_at_count(prepared, 0, 0)["metric"], ev["small_test"], run.name + "/specialist")
            check_close(point_at_count(prepared, prepared["n"], 1)["metric"], ev["large_test"], run.name + "/VLM")
            row = {k: cfg[k] for k in ("task", "expert", "endpoint", "method", "seed")}
            row.update({"ranking": ranking, "same_checkpoint_control": ranking != cfg["ranking"]})
            for metric in METRICS[:3]:
                row[metric] = result[metric]
            for split in ("validation", "test"):
                row[split + "_metric"] = result[split + "_selected"]["metric"]
                row[split + "_call"] = result[split + "_selected"]["actual_fraction"]
            row.update({key: result["validation_selected"][key] for key in ("threshold", "operator")})
            row["nominal_budget"] = result["validation_selected"]["budget"]
            per_seed.append(row)
        if (run_index + 1) % 50 == 0:
            print(f"Verified {run_index + 1}/400 runs", flush=True)

    primary = [r for r in per_seed if not r["same_checkpoint_control"]]
    objectives = aggregate_objectives(primary, METRICS)
    with (root / "objective_mean_std.csv").open(encoding="utf-8", newline="") as stream:
        recorded_objectives = {r["method"]: r for r in csv.DictReader(stream)}
    for row in objectives:
        for key, value in row.items():
            if key != "method":
                check_close(value, float(recorded_objectives[row["method"]][key]),
                            f"published objective summary {row['method']}/{key}")
    with (root / "per_seed.csv").open(encoding="utf-8", newline="") as stream:
        recorded_runs = {(r["expert"], r["endpoint"], r["method"], int(r["seed"]), r["ranking"]): r
                         for r in csv.DictReader(stream)}
    if len(recorded_runs) != len(per_seed):
        raise ValueError("Published per-seed summary has a different number of comparisons")
    for row in per_seed:
        key = tuple(row[k] for k in ("expert", "endpoint", "method", "seed", "ranking"))
        for field in METRICS + ["threshold", "nominal_budget"]:
            check_close(row[field], float(recorded_runs[key][field]), f"published per-seed summary {key}/{field}")
        if row["operator"] != recorded_runs[key]["operator"]:
            raise ValueError(f"Published per-seed summary operator differs: {key}")
    with (root / "pair_mean_std.csv").open(encoding="utf-8", newline="") as stream:
        recorded_pairs = list(csv.DictReader(stream))
    if len(recorded_pairs) != 80:
        raise ValueError("The full objective ablation requires 80 model-pair/objective rows")
    for expected in recorded_pairs:
        rr = [r for r in primary if all(r[k] == expected[k] for k in ("task", "expert", "endpoint", "method"))]
        if sorted(r["seed"] for r in rr) != SEEDS:
            raise ValueError(f"Incomplete pair/objective summary: {expected}")
        for metric in METRICS:
            values = [r[metric] for r in rr]
            check_close(float(np.mean(values)), float(expected[metric + "_mean"]),
                        f"published full ablation {expected['expert']}/{expected['endpoint']}/{expected['method']}/{metric}/mean")
            check_close(float(np.std(values, ddof=1)), float(expected[metric + "_std"]),
                        f"published full ablation {expected['expert']}/{expected['endpoint']}/{expected['method']}/{metric}/SD")
    main_rows, confidence_curve_points = [], 0
    counts = {"higher_metric_than_confidence": 0, "fewer_calls_than_confidence": 0,
              "higher_metric_and_fewer_calls": 0, "above_both_endpoints": 0}
    for group in paper["groups"]:
        task, expert, endpoint = group["key"]
        rr = [r for r in primary if (r["task"], r["expert"], r["endpoint"], r["method"])
              == (task, expert, endpoint, selected)]
        if sorted(r["seed"] for r in rr) != SEEDS:
            raise ValueError(f"Missing paper model pair: {group['key']}")
        row = {"task": task, "expert": expert, "endpoint": endpoint, "method": selected}
        for metric in METRICS:
            row[metric + "_mean"] = float(np.mean([r[metric] for r in rr]))
            row[metric + "_std"] = float(np.std([r[metric] for r in rr], ddof=1))
        learned, confidence = group["methods"]["learned"], group["methods"]["confidence"]
        pair = f"shared_12_{expert}_{endpoint}"
        prepared_val = prepare_curve(confidence_scores[pair]["val"], *outcomes[pair]["val"])
        conf_thresholds = validation_thresholds(prepared_val)
        conf_curves = {"validation": fixed_curve(prepared_val, conf_thresholds),
                       "test": fixed_curve(prepare_curve(confidence_scores[pair]["test"],
                                                        *outcomes[pair]["test"]), conf_thresholds)}
        conf_best = selected_index(conf_curves["validation"])
        if conf_best != confidence["selected_budget_index"]:
            raise ValueError(f"Confidence validation-selected threshold differs: {expert}/{endpoint}")
        conf_area = score25([conf_curves["validation"][i] for i in (0, 5, 10, 15, 20, 25)])
        check_close(conf_area, confidence["validation_area_0_25"], f"confidence {expert}/{endpoint}/validation area")
        for split in ("validation", "test"):
            reference_curve = [dict(point, rescue=point["rescued"], harm=point["harmed"])
                               for point in confidence[split + "_curve"]]
            verify_curve(conf_curves[split], reference_curve, f"confidence {expert}/{endpoint}/{split}")
            reference_point = confidence[split + "_point"]
            verify_curve([conf_curves[split][conf_best]],
                         [dict(reference_point, rescue=reference_point["rescued"], harm=reference_point["harmed"])],
                         f"confidence {expert}/{endpoint}/{split}/selected")
            confidence_curve_points += len(reference_curve)
            for field, label in [("metric", "metric"), ("actual_fraction", "call")]:
                check_close(row[split + "_" + label + "_mean"], learned[split + "_point"][field],
                            f"paper {expert}/{endpoint}/{split}/{field} mean")
                check_close(row[split + "_" + label + "_std"], learned[split + "_point"][field + "_std"],
                            f"paper {expert}/{endpoint}/{split}/{field} SD")
                row["confidence_" + split + "_" + label] = conf_curves[split][conf_best][field]
            for i, point in enumerate(learned[split + "_curve"]):
                for field in ("metric", "actual_fraction"):
                    values = [replay[f"repeat_{expert}_{endpoint}_{selected}_s{s}"]
                              [configurations[f"repeat_{expert}_{endpoint}_{selected}_s{s}"]["ranking"]]
                              [split + "_curve"][i][field] for s in SEEDS]
                    check_close(float(np.mean(values)), point[field], f"paper mean curve {expert}/{endpoint}/{split}/{i}/{field}")
                    check_close(float(np.std(values, ddof=1)), point[field + "_std"],
                                f"paper curve SD {expert}/{endpoint}/{split}/{i}/{field}")
        row["specialist_test_metric"] = group["small"]["test"]
        row["VLM_test_metric"] = group["large"]["test"]
        better = row["test_metric_mean"] > row["confidence_test_metric"] + TOL
        fewer = row["test_call_mean"] < row["confidence_test_call"] - TOL
        counts["higher_metric_than_confidence"] += int(better)
        counts["fewer_calls_than_confidence"] += int(fewer)
        counts["higher_metric_and_fewer_calls"] += int(better and fewer)
        counts["above_both_endpoints"] += int(row["test_metric_mean"] > max(row["specialist_test_metric"], row["VLM_test_metric"]) + TOL)
        main_rows.append(row)

    controls = []
    for run in runs:
        cfg = configurations[run.name]
        if cfg["target"] != "four_state":
            continue
        row = {k: cfg[k] for k in ("task", "expert", "endpoint", "method", "seed")}
        for metric in METRICS[:3]:
            difference = replay[run.name]["probability_difference"][metric]
            ratio = replay[run.name]["log_ratio"][metric]
            row[metric + "_probability_difference"] = difference
            row[metric + "_log_ratio"] = ratio
            row[metric + "_difference_minus_ratio"] = difference - ratio
        controls.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "replayed_main_five_seeds.csv", main_rows)
    write_csv(args.output_dir / "replayed_objectives_five_seeds.csv", objectives)
    write_csv(args.output_dir / "replayed_per_seed.csv", per_seed)
    write_csv(args.output_dir / "replayed_same_checkpoint_controls.csv", controls)
    verification = {"status": "passed", "selected_method": selected, "seeds": SEEDS,
                    "primary_runs": len(primary), "same_checkpoint_controls": len(controls),
                    "verified_curve_points": curve_points, "main_pairs": len(main_rows),
                    "verified_confidence_curve_points": confidence_curve_points,
                    "sample_standard_deviation_ddof": 1, "metric_and_call_tolerance": TOL,
                    "verified_pair_objective_summary_rows": len(recorded_pairs),
                    "raw_logit_score_check_absolute_tolerance": 8e-7,
                    "counts_use_strict_improvement_above_tolerance": counts,
                    "selection": "validation task-balanced Score25 mean minus sample SD, before test replay",
                    "scope": "Saved router outputs, frozen endpoint outcomes, explicit per-event correctness labels and specialist confidence/validity. Both learned and confidence curves and validation-selected points are independently replayed; no model retraining.",
                    "study_history": lock["rule_amendment"]}
    (args.output_dir / "verification_five_seeds.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
