#!/usr/bin/env python
"""P0: head-damage-matched Pareto audit of scalar vs class aggregation.

P0 is an exploratory, non-deployable analysis built on frozen D2b artifacts.
It does not perform federated training or refit client weights.  The complete
gamma/tau grid is evaluated on the deterministic train calibration split,
budget choices and direct matches are frozen, and only then are the same
candidates evaluated on official test.  Per-gamma metric caches make the
expensive exact class-state evaluation resumable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_d2b_scalar_ceiling import (
    _dump_dir,
    _method_metrics,
    build_train_subset_loader,
    client_vectors,
    exact_class_conditional_logits,
    exact_scalar_logits,
    tail_coverage_groups,
)
from scripts.analyze_d3_boundary import apply_logit_adjustment
from utils.d23_common import (
    D23_SCHEMA_VERSION,
    build_trainer,
    class_split,
    load_dump,
    sha256_file,
    validate_dump,
    write_csv,
    write_json,
)


METHODS = ("scalar", "class_conditional")
OBJECTIVES = ("head_tail_harmonic", "tail_accuracy")


def parse_float_grid(text: str, *, name: str) -> list[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    unique = sorted(set(values))
    if any(not math.isfinite(value) for value in unique):
        raise ValueError(f"{name} contains a non-finite value")
    return unique


def gamma_key(value: float) -> str:
    return f"{float(value):.6f}".replace("-", "m").replace(".", "p")


def recover_alternative_weights(
    base_weights: torch.Tensor,
    selected_weights: torch.Tensor,
    selected_gamma: float,
) -> torch.Tensor:
    """Recover D2b's gamma=1 endpoint from its frozen selected mixture."""
    gamma = float(selected_gamma)
    if gamma <= 1e-12:
        raise ValueError(
            "Cannot recover the D2b alternative endpoint from selected gamma=0. "
            "Rerun D2b with alternative weights stored explicitly."
        )
    base = torch.as_tensor(base_weights, dtype=torch.float64)
    selected = torch.as_tensor(selected_weights, dtype=torch.float64)
    if selected.ndim == 2:
        base = base[:, None]
    alternative = base + (selected - base) / gamma
    reconstructed = (1.0 - gamma) * base + gamma * alternative
    if not torch.allclose(reconstructed, selected, atol=1e-8, rtol=1e-7):
        raise RuntimeError("P0 failed to reconstruct the frozen D2b weights")
    sums = alternative.sum(dim=0) if alternative.ndim == 2 else alternative.sum()
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise RuntimeError("Recovered D2b endpoint is not a normalized distribution")
    if float(alternative.min().item()) < -1e-6:
        raise RuntimeError("Recovered D2b endpoint contains negative aggregation weights")
    return alternative.clamp_min(0.0)


def weights_at_gamma(
    base_weights: torch.Tensor, alternative_weights: torch.Tensor, gamma: float
) -> torch.Tensor:
    base = torch.as_tensor(base_weights, dtype=torch.float64)
    alternative = torch.as_tensor(alternative_weights, dtype=torch.float64)
    if alternative.ndim == 2:
        base = base[:, None]
    return (1.0 - float(gamma)) * base + float(gamma) * alternative


def _candidate_metrics(
    round_id: int,
    method: str,
    gamma: float,
    logits: torch.Tensor,
    labels: torch.Tensor,
    priors: torch.Tensor,
    taus: Sequence[float],
    head: Sequence[int],
    tail: Sequence[int],
    covered: Sequence[int],
    uncovered: Sequence[int],
    split: str,
) -> list[dict]:
    rows = []
    for tau in taus:
        adjusted = apply_logit_adjustment(logits, priors, tau)
        rows.append({
            "split": split,
            "communication_round": int(round_id),
            "method": method,
            "gamma": float(gamma),
            "tau": float(tau),
            **_method_metrics(
                round_id,
                method,
                adjusted,
                labels,
                head,
                tail,
                covered,
                uncovered,
            ),
        })
    return rows


def _read_cache(path: Path, expected: Mapping) -> list[dict] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if payload.get(key) != value:
            return None
    rows = payload.get("rows")
    return list(rows) if isinstance(rows, list) else None


def _write_cache(path: Path, metadata: Mapping, rows: Sequence[Mapping]) -> None:
    write_json(path, {**metadata, "rows": list(rows)})


def _best_row(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("Cannot select from an empty candidate set")
    return dict(max(
        rows,
        key=lambda row: (
            float(row["head_tail_harmonic"]),
            float(row["balanced_accuracy"]),
            float(row["tail_accuracy"]),
            float(row["head_accuracy"]),
            -float(row["gamma"]),
            -float(row["tau"]),
        ),
    ))


def select_budget_choices(
    rows: Sequence[Mapping],
    reference_head: float,
    budgets: Sequence[float],
) -> list[dict]:
    choices = []
    for budget in budgets:
        threshold = float(reference_head) - float(budget)
        eligible = [
            row for row in rows if float(row["head_accuracy"]) >= threshold - 1e-9
        ]
        if not eligible:
            choices.append({
                "head_budget": float(budget),
                "head_threshold": threshold,
                "eligible": False,
            })
            continue
        best = _best_row(eligible)
        choices.append({
            "head_budget": float(budget),
            "head_threshold": threshold,
            "eligible": True,
            **best,
        })
    return choices


def match_class_to_scalar(
    class_rows: Sequence[Mapping],
    scalar_rows: Sequence[Mapping],
    tolerance: float,
) -> list[dict]:
    """Freeze the best scalar comparator for every class candidate by head accuracy."""
    matches = []
    for class_row in class_rows:
        eligible = [
            row
            for row in scalar_rows
            if abs(float(row["head_accuracy"]) - float(class_row["head_accuracy"]))
            <= float(tolerance) + 1e-9
        ]
        base = {
            "communication_round": int(class_row["communication_round"]),
            "class_gamma": float(class_row["gamma"]),
            "class_tau": float(class_row["tau"]),
            "calibration_class_head_accuracy": float(class_row["head_accuracy"]),
            "head_match_tolerance": float(tolerance),
            "matched": bool(eligible),
        }
        if eligible:
            scalar = _best_row(eligible)
            base.update({
                "scalar_gamma": float(scalar["gamma"]),
                "scalar_tau": float(scalar["tau"]),
                "calibration_scalar_head_accuracy": float(scalar["head_accuracy"]),
                "calibration_head_gap": (
                    float(class_row["head_accuracy"])
                    - float(scalar["head_accuracy"])
                ),
            })
        matches.append(base)
    return matches


def pareto_frontier(rows: Sequence[Mapping], objective: str) -> list[dict]:
    """Return points not dominated when maximizing head accuracy and objective."""
    result = []
    for index, row in enumerate(rows):
        head_value = float(row["head_accuracy"])
        objective_value = float(row[objective])
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            other_head = float(other["head_accuracy"])
            other_objective = float(other[objective])
            if (
                other_head >= head_value - 1e-12
                and other_objective >= objective_value - 1e-12
                and (
                    other_head > head_value + 1e-12
                    or other_objective > objective_value + 1e-12
                )
            ):
                dominated = True
                break
        if not dominated:
            result.append(dict(row))
    return sorted(
        result,
        key=lambda row: (float(row["head_accuracy"]), float(row[objective])),
    )


def envelope_auc(
    rows: Sequence[Mapping], objective: str, x_low: float, x_high: float, steps: int = 400
) -> float:
    """Area of best objective attainable at each minimum-head threshold."""
    if not rows or x_high <= x_low:
        return math.nan
    xs = torch.linspace(float(x_low), float(x_high), int(steps) + 1)
    ys = []
    for value in xs.tolist():
        eligible = [
            float(row[objective])
            for row in rows
            if float(row["head_accuracy"]) >= value - 1e-9
        ]
        ys.append(max(eligible) if eligible else math.nan)
    if any(not math.isfinite(value) for value in ys):
        return math.nan
    return float(torch.trapezoid(torch.tensor(ys), xs).item())


def _lookup(rows: Sequence[Mapping], method: str, gamma: float, tau: float) -> dict:
    matches = [
        row
        for row in rows
        if row["method"] == method
        and abs(float(row["gamma"]) - float(gamma)) <= 1e-9
        and abs(float(row["tau"]) - float(tau)) <= 1e-9
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {method} candidate at gamma={gamma}, tau={tau}; "
            f"found {len(matches)}"
        )
    return dict(matches[0])


def budget_test_report(
    frozen_choices: Sequence[Mapping],
    test_rows: Sequence[Mapping],
    *,
    min_h_gain: float,
    min_tail_gain: float,
    min_balanced_gain: float,
    max_uncovered_drop: float,
) -> list[dict]:
    by_key = {
        (
            int(row["communication_round"]),
            float(row["head_budget"]),
            str(row["method"]),
        ): row
        for row in frozen_choices
        if bool(row.get("eligible", False))
    }
    keys = sorted((round_id, budget) for round_id, budget, _ in by_key)
    keys = sorted(set(keys))
    report = []
    for round_id, budget in keys:
        scalar_choice = by_key.get((round_id, budget, "scalar"))
        class_choice = by_key.get((round_id, budget, "class_conditional"))
        if scalar_choice is None or class_choice is None:
            continue
        scalar = _lookup(
            test_rows,
            "scalar",
            float(scalar_choice["gamma"]),
            float(scalar_choice["tau"]),
        )
        classwise = _lookup(
            test_rows,
            "class_conditional",
            float(class_choice["gamma"]),
            float(class_choice["tau"]),
        )
        reference_tau = float(class_choice["reference_tau"])
        reference = _lookup(test_rows, "scalar", 0.0, reference_tau)
        scalar_head_damage = (
            float(reference["head_accuracy"]) - float(scalar["head_accuracy"])
        )
        class_head_damage = (
            float(reference["head_accuracy"]) - float(classwise["head_accuracy"])
        )
        test_budget_respected = (
            scalar_head_damage <= float(budget) + 1e-9
            and class_head_damage <= float(budget) + 1e-9
        )
        contrasts = {
            "head_gain": float(classwise["head_accuracy"]) - float(scalar["head_accuracy"]),
            "tail_gain": float(classwise["tail_accuracy"]) - float(scalar["tail_accuracy"]),
            "balanced_gain": float(classwise["balanced_accuracy"]) - float(scalar["balanced_accuracy"]),
            "overall_gain": float(classwise["overall_accuracy"]) - float(scalar["overall_accuracy"]),
            "harmonic_gain": float(classwise["head_tail_harmonic"]) - float(scalar["head_tail_harmonic"]),
            "covered_tail_gain": float(classwise["covered_tail_accuracy"]) - float(scalar["covered_tail_accuracy"]),
            "uncovered_tail_gain": float(classwise["uncovered_tail_accuracy"]) - float(scalar["uncovered_tail_accuracy"]),
        }
        passed = (
            test_budget_respected
            and contrasts["harmonic_gain"] >= float(min_h_gain)
            and contrasts["tail_gain"] >= float(min_tail_gain)
            and contrasts["balanced_gain"] >= float(min_balanced_gain)
            and contrasts["uncovered_tail_gain"] >= -float(max_uncovered_drop)
        )
        report.append({
            "communication_round": round_id,
            "head_budget": budget,
            "scalar_gamma": float(scalar_choice["gamma"]),
            "scalar_tau": float(scalar_choice["tau"]),
            "class_gamma": float(class_choice["gamma"]),
            "class_tau": float(class_choice["tau"]),
            "test_reference_head_accuracy": float(reference["head_accuracy"]),
            "scalar_head_damage_vs_fedavg_la": scalar_head_damage,
            "class_head_damage_vs_fedavg_la": class_head_damage,
            "test_budget_respected": test_budget_respected,
            **{f"scalar_{key}": value for key, value in scalar.items() if key.endswith("accuracy") or key == "head_tail_harmonic"},
            **{f"class_{key}": value for key, value in classwise.items() if key.endswith("accuracy") or key == "head_tail_harmonic"},
            **contrasts,
            "pass": passed,
        })
    return report


def direct_match_test_report(
    frozen_matches: Sequence[Mapping], test_rows: Sequence[Mapping]
) -> list[dict]:
    rows = []
    for match in frozen_matches:
        if not bool(match.get("matched", False)):
            rows.append(dict(match))
            continue
        classwise = _lookup(
            test_rows,
            "class_conditional",
            float(match["class_gamma"]),
            float(match["class_tau"]),
        )
        scalar = _lookup(
            test_rows,
            "scalar",
            float(match["scalar_gamma"]),
            float(match["scalar_tau"]),
        )
        rows.append({
            **match,
            "test_class_head_accuracy": float(classwise["head_accuracy"]),
            "test_scalar_head_accuracy": float(scalar["head_accuracy"]),
            "test_head_gap": float(classwise["head_accuracy"]) - float(scalar["head_accuracy"]),
            "test_tail_gain": float(classwise["tail_accuracy"]) - float(scalar["tail_accuracy"]),
            "test_balanced_gain": float(classwise["balanced_accuracy"]) - float(scalar["balanced_accuracy"]),
            "test_overall_gain": float(classwise["overall_accuracy"]) - float(scalar["overall_accuracy"]),
            "test_harmonic_gain": float(classwise["head_tail_harmonic"]) - float(scalar["head_tail_harmonic"]),
            "test_covered_tail_gain": float(classwise["covered_tail_accuracy"]) - float(scalar["covered_tail_accuracy"]),
            "test_uncovered_tail_gain": float(classwise["uncovered_tail_accuracy"]) - float(scalar["uncovered_tail_accuracy"]),
        })
    return rows


def _svg_polyline(points: Sequence[tuple[float, float]], color: str, panel: Mapping) -> str:
    if not points:
        return ""
    def sx(value):
        return panel["x"] + (value - panel["xmin"]) / max(panel["xmax"] - panel["xmin"], 1e-9) * panel["w"]
    def sy(value):
        return panel["y"] + panel["h"] - (value - panel["ymin"]) / max(panel["ymax"] - panel["ymin"], 1e-9) * panel["h"]
    coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    circles = "".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.7" fill="{color}"/>'
        for x, y in points
    )
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>{circles}'


def write_pareto_svg(path: Path, round_id: int, test_rows: Sequence[Mapping]) -> None:
    width, height = 1000, 430
    panels = []
    for panel_index, objective in enumerate(OBJECTIVES):
        all_x = [float(row["head_accuracy"]) for row in test_rows]
        all_y = [float(row[objective]) for row in test_rows]
        x_pad = max(0.5, (max(all_x) - min(all_x)) * 0.08)
        y_pad = max(0.5, (max(all_y) - min(all_y)) * 0.08)
        panels.append({
            "x": 70 + panel_index * 490,
            "y": 50,
            "w": 400,
            "h": 310,
            "xmin": min(all_x) - x_pad,
            "xmax": max(all_x) + x_pad,
            "ymin": min(all_y) - y_pad,
            "ymax": max(all_y) + y_pad,
            "objective": objective,
        })
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="500" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">P0 official-test Pareto, round {round_id}</text>',
    ]
    colors = {"scalar": "#2864dc", "class_conditional": "#e4771b"}
    for panel in panels:
        parts.extend([
            f'<rect x="{panel["x"]}" y="{panel["y"]}" width="{panel["w"]}" height="{panel["h"]}" fill="none" stroke="#555"/>',
            f'<text x="{panel["x"] + panel["w"] / 2}" y="400" text-anchor="middle" font-family="sans-serif" font-size="13">head accuracy (%)</text>',
            f'<text x="{panel["x"] + panel["w"] / 2}" y="42" text-anchor="middle" font-family="sans-serif" font-size="14">{panel["objective"]}</text>',
            f'<text x="{panel["x"]}" y="380" font-family="sans-serif" font-size="11">{panel["xmin"]:.1f}</text>',
            f'<text x="{panel["x"] + panel["w"]}" y="380" text-anchor="end" font-family="sans-serif" font-size="11">{panel["xmax"]:.1f}</text>',
            f'<text x="{panel["x"] - 8}" y="{panel["y"] + panel["h"]}" text-anchor="end" font-family="sans-serif" font-size="11">{panel["ymin"]:.1f}</text>',
            f'<text x="{panel["x"] - 8}" y="{panel["y"] + 10}" text-anchor="end" font-family="sans-serif" font-size="11">{panel["ymax"]:.1f}</text>',
        ])
        for method in METHODS:
            method_rows = [row for row in test_rows if row["method"] == method]
            frontier = pareto_frontier(method_rows, panel["objective"])
            points = [
                (float(row["head_accuracy"]), float(row[panel["objective"]]))
                for row in frontier
            ]
            parts.append(_svg_polyline(points, colors[method], panel))
    parts.extend([
        '<line x1="730" y1="416" x2="755" y2="416" stroke="#2864dc" stroke-width="3"/><text x="762" y="420" font-family="sans-serif" font-size="12">scalar</text>',
        '<line x1="830" y1="416" x2="855" y2="416" stroke="#e4771b" stroke-width="3"/><text x="862" y="420" font-family="sans-serif" font-size="12">class-conditional</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--d2b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", default="20,50,80")
    parser.add_argument("--gammas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--taus", default="0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5")
    parser.add_argument("--head-budgets", default="0,0.5,1.0,2.0")
    parser.add_argument("--head-match-tolerance", type=float, default=0.25)
    parser.add_argument("--min-h-gain", type=float, default=1.0)
    parser.add_argument("--min-tail-gain", type=float, default=2.0)
    parser.add_argument("--min-balanced-gain", type=float, default=-0.2)
    parser.add_argument("--max-uncovered-drop", type=float, default=1.0)
    parser.add_argument("--min-budgets-per-round", type=int, default=2)
    parser.add_argument("--min-positive-rounds", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rounds = [int(value) for value in args.rounds.split(",") if value.strip()]
    gammas = parse_float_grid(args.gammas, name="gammas")
    taus = parse_float_grid(args.taus, name="taus")
    budgets = parse_float_grid(args.head_budgets, name="head budgets")
    if gammas[0] < 0.0 or gammas[-1] > 1.0:
        raise ValueError("P0 gammas must stay in [0, 1]")
    if any(value < 0.0 for value in taus + budgets):
        raise ValueError("P0 taus and head budgets must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for round_id in rounds:
        dump_dir = _dump_dir(args.dump_root, round_id)
        payload, metadata = load_dump(dump_dir)
        validate_dump(payload, metadata)
        artifact_path = args.d2b_dir / f"d2b_frozen_round_{round_id:03d}.pt"
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"P0 requires frozen D2b artifact: {artifact_path}. Run D2b first."
            )
        loaded.append((dump_dir, payload, metadata, artifact_path))

    cfg, trainer = build_trainer(
        loaded[0][2], args.output_dir / "eval_runtime", args.eval_batch_size
    )
    calibration_rows, audit_rows = [], []
    round_runtime = []

    # Phase one: evaluate every candidate on train calibration only.
    for dump_dir, payload, metadata, artifact_path in loaded:
        round_id = int(metadata["communication_round"])
        print(f"P0 round {round_id}: calibration grid", flush=True)
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        spec, before, deltas, fedavg_weights = client_vectors(payload)
        head, tail = class_split(payload["global_class_counts"])
        if [int(value) for value in artifact["tail_class_ids"]] != [int(value) for value in tail]:
            raise RuntimeError(
                f"P0 round {round_id} tail-class order differs from frozen D2b"
            )
        covered, uncovered = tail_coverage_groups(payload, tail)
        selected_scalar_gamma = float(artifact["scalar_gamma"])
        selected_class_gamma = float(artifact["class_gamma"])
        alternative_scalar = recover_alternative_weights(
            fedavg_weights, artifact["scalar_weights"], selected_scalar_gamma
        )
        alternative_class = recover_alternative_weights(
            fedavg_weights, artifact["class_weights"], selected_class_gamma
        )
        calibration_indices = torch.as_tensor(artifact["calibration_indices"]).long()
        calibration_loader = build_train_subset_loader(cfg, trainer, calibration_indices)
        source = list(getattr(trainer.dm.dataset, "train_x", []) or [])
        calibration_labels = torch.tensor(
            [int(source[int(index)].label) for index in calibration_indices.tolist()],
            dtype=torch.long,
        )
        priors = torch.as_tensor(artifact["priors"]).float()
        fedavg_calibration, observed = exact_scalar_logits(
            trainer, spec, before, deltas, fedavg_weights, calibration_loader
        )
        if not torch.equal(calibration_labels, observed):
            raise RuntimeError("P0 calibration loader order differs from frozen D2b indices")
        audit_rows.append({
            "communication_round": round_id,
            "dump_sha256": sha256_file(dump_dir / "round_state.pt"),
            "d2b_artifact_sha256": sha256_file(artifact_path),
            "selected_scalar_gamma": selected_scalar_gamma,
            "selected_class_gamma": selected_class_gamma,
            "recovered_scalar_min_weight": float(alternative_scalar.min().item()),
            "recovered_class_min_weight": float(alternative_class.min().item()),
            "covered_tail_class_count": len(covered),
            "uncovered_tail_class_count": len(uncovered),
        })
        for method in METHODS:
            endpoint = alternative_scalar if method == "scalar" else alternative_class
            for gamma in gammas:
                cache_path = (
                    cache_dir
                    / f"round_{round_id:03d}"
                    / f"calibration_{method}_gamma_{gamma_key(gamma)}.json"
                )
                cache_meta = {
                    "schema_version": "p0_head_pareto_cache_v1",
                    "split": "calibration",
                    "communication_round": round_id,
                    "method": method,
                    "gamma": float(gamma),
                    "taus": taus,
                    "dump_sha256": sha256_file(dump_dir / "round_state.pt"),
                    "artifact_sha256": sha256_file(artifact_path),
                }
                rows = _read_cache(cache_path, cache_meta)
                if rows is None:
                    weights = weights_at_gamma(fedavg_weights, endpoint, gamma)
                    if method == "scalar":
                        logits, labels = (
                            (fedavg_calibration, calibration_labels)
                            if abs(gamma) <= 1e-12
                            else exact_scalar_logits(
                                trainer, spec, before, deltas, weights, calibration_loader
                            )
                        )
                    else:
                        labels = calibration_labels
                        logits = (
                            fedavg_calibration
                            if abs(gamma) <= 1e-12
                            else exact_class_conditional_logits(
                                trainer,
                                spec,
                                before,
                                deltas,
                                weights,
                                tail,
                                calibration_loader,
                                fedavg_calibration,
                                calibration_labels,
                                split_name=f"P0 calibration gamma={gamma:g}",
                            )
                        )
                    if not torch.equal(labels, calibration_labels):
                        raise RuntimeError("P0 calibration labels changed across candidates")
                    rows = _candidate_metrics(
                        round_id,
                        method,
                        gamma,
                        logits,
                        calibration_labels,
                        priors,
                        taus,
                        head,
                        tail,
                        covered,
                        uncovered,
                        "calibration",
                    )
                    _write_cache(cache_path, cache_meta, rows)
                calibration_rows.extend(rows)
        round_runtime.append({
            "round_id": round_id,
            "artifact": artifact,
            "payload": payload,
            "metadata": metadata,
            "dump_dir": dump_dir,
            "alternative_scalar": alternative_scalar,
            "alternative_class": alternative_class,
            "head": head,
            "tail": tail,
            "covered": covered,
            "uncovered": uncovered,
        })

    write_csv(args.output_dir / "p0_calibration_candidate_grid.csv", calibration_rows)
    write_csv(args.output_dir / "p0_reconstruction_audit.csv", audit_rows)

    frozen_budget_rows, frozen_match_rows = [], []
    calibration_frontier_rows, calibration_auc_rows = [], []
    for item in round_runtime:
        round_id = item["round_id"]
        artifact = item["artifact"]
        round_rows = [
            row for row in calibration_rows
            if int(row["communication_round"]) == round_id
        ]
        fedavg_tau = float(artifact["selected_taus"]["fedavg"])
        fedavg_reference = _lookup(round_rows, "scalar", 0.0, fedavg_tau)
        reference_head = float(fedavg_reference["head_accuracy"])
        for method in METHODS:
            method_rows = [row for row in round_rows if row["method"] == method]
            choices = select_budget_choices(method_rows, reference_head, budgets)
            frozen_budget_rows.extend({
                "communication_round": round_id,
                "method": method,
                "reference_method": "fedavg_la",
                "reference_tau": fedavg_tau,
                "reference_head_accuracy": reference_head,
                **row,
            } for row in choices)
            for objective in OBJECTIVES:
                frontier = pareto_frontier(method_rows, objective)
                calibration_frontier_rows.extend({
                    "objective": objective,
                    "pareto": True,
                    **row,
                } for row in frontier)
        class_rows = [row for row in round_rows if row["method"] == "class_conditional"]
        scalar_rows = [row for row in round_rows if row["method"] == "scalar"]
        frozen_match_rows.extend(
            match_class_to_scalar(class_rows, scalar_rows, args.head_match_tolerance)
        )
        x_low = max(
            min(float(row["head_accuracy"]) for row in scalar_rows),
            min(float(row["head_accuracy"]) for row in class_rows),
        )
        x_high = min(
            max(float(row["head_accuracy"]) for row in scalar_rows),
            max(float(row["head_accuracy"]) for row in class_rows),
        )
        for objective in OBJECTIVES:
            scalar_auc = envelope_auc(scalar_rows, objective, x_low, x_high)
            class_auc = envelope_auc(class_rows, objective, x_low, x_high)
            head_span = x_high - x_low
            calibration_auc_rows.append({
                "split": "calibration",
                "communication_round": round_id,
                "objective": objective,
                "common_head_low": x_low,
                "common_head_high": x_high,
                "common_head_span": head_span,
                "scalar_envelope_auc": scalar_auc,
                "class_envelope_auc": class_auc,
                "class_minus_scalar_auc": class_auc - scalar_auc,
                "scalar_normalized_envelope_auc": scalar_auc / head_span if head_span > 0 else math.nan,
                "class_normalized_envelope_auc": class_auc / head_span if head_span > 0 else math.nan,
            })

    write_csv(args.output_dir / "p0_frozen_budget_choices.csv", frozen_budget_rows)
    write_csv(args.output_dir / "p0_frozen_direct_matches.csv", frozen_match_rows)
    write_csv(args.output_dir / "p0_calibration_pareto_frontier.csv", calibration_frontier_rows)
    write_csv(args.output_dir / "p0_calibration_pareto_auc.csv", calibration_auc_rows)
    frozen_manifest = {
        "schema_version": D23_SCHEMA_VERSION,
        "diagnostic": "P0_head_damage_matched_pareto",
        "seed": 42,
        "rounds": rounds,
        "gammas": gammas,
        "taus": taus,
        "head_budgets": budgets,
        "head_match_tolerance": args.head_match_tolerance,
        "selection_source": "disjoint deterministic global-train calibration split frozen by D2b",
        "candidate_weights_source": "frozen D2b scalar/class endpoints",
        "candidate_selection_used_official_test": False,
        "exploratory_after_seed42_test_access": True,
        "choices_frozen_before_new_test_grid_inference": True,
        "budget_choices_sha256": sha256_file(args.output_dir / "p0_frozen_budget_choices.csv"),
        "direct_matches_sha256": sha256_file(args.output_dir / "p0_frozen_direct_matches.csv"),
        "new_test_grid_inference_accessed": False,
    }
    write_json(args.output_dir / "p0_manifest.json", frozen_manifest)
    print("P0 calibration choices frozen; start exact official-test grid", flush=True)

    # Phase two: evaluate the already frozen grid on official test.
    test_rows = []
    for item in round_runtime:
        round_id = item["round_id"]
        payload = item["payload"]
        artifact = item["artifact"]
        dump_dir = item["dump_dir"]
        spec, before, deltas, fedavg_weights = client_vectors(payload)
        priors = torch.as_tensor(artifact["priors"]).float()
        fedavg_test, test_labels = exact_scalar_logits(
            trainer, spec, before, deltas, fedavg_weights, trainer.test_loader
        )
        for method in METHODS:
            endpoint = (
                item["alternative_scalar"]
                if method == "scalar"
                else item["alternative_class"]
            )
            for gamma in gammas:
                print(
                    f"P0 round {round_id}: test {method} gamma={gamma:g}", flush=True
                )
                cache_path = (
                    cache_dir
                    / f"round_{round_id:03d}"
                    / f"test_{method}_gamma_{gamma_key(gamma)}.json"
                )
                cache_meta = {
                    "schema_version": "p0_head_pareto_cache_v1",
                    "split": "test",
                    "communication_round": round_id,
                    "method": method,
                    "gamma": float(gamma),
                    "taus": taus,
                    "dump_sha256": sha256_file(dump_dir / "round_state.pt"),
                    "artifact_sha256": sha256_file(
                        args.d2b_dir / f"d2b_frozen_round_{round_id:03d}.pt"
                    ),
                    "frozen_budget_choices_sha256": frozen_manifest["budget_choices_sha256"],
                }
                rows = _read_cache(cache_path, cache_meta)
                if rows is None:
                    weights = weights_at_gamma(fedavg_weights, endpoint, gamma)
                    if method == "scalar":
                        logits, labels = (
                            (fedavg_test, test_labels)
                            if abs(gamma) <= 1e-12
                            else exact_scalar_logits(
                                trainer, spec, before, deltas, weights, trainer.test_loader
                            )
                        )
                    else:
                        labels = test_labels
                        logits = (
                            fedavg_test
                            if abs(gamma) <= 1e-12
                            else exact_class_conditional_logits(
                                trainer,
                                spec,
                                before,
                                deltas,
                                weights,
                                item["tail"],
                                trainer.test_loader,
                                fedavg_test,
                                test_labels,
                                split_name=f"P0 test gamma={gamma:g}",
                            )
                        )
                    if not torch.equal(labels, test_labels):
                        raise RuntimeError("P0 test labels changed across candidates")
                    rows = _candidate_metrics(
                        round_id,
                        method,
                        gamma,
                        logits,
                        test_labels,
                        priors,
                        taus,
                        item["head"],
                        item["tail"],
                        item["covered"],
                        item["uncovered"],
                        "test",
                    )
                    _write_cache(cache_path, cache_meta, rows)
                test_rows.extend(rows)

    write_csv(args.output_dir / "p0_test_candidate_grid.csv", test_rows)
    budget_report = budget_test_report(
        frozen_budget_rows,
        test_rows,
        min_h_gain=args.min_h_gain,
        min_tail_gain=args.min_tail_gain,
        min_balanced_gain=args.min_balanced_gain,
        max_uncovered_drop=args.max_uncovered_drop,
    )
    direct_report = direct_match_test_report(frozen_match_rows, test_rows)
    write_csv(args.output_dir / "p0_budget_report.csv", budget_report)
    write_csv(args.output_dir / "p0_direct_match_report.csv", direct_report)

    test_frontier_rows, test_auc_rows = [], []
    for round_id in rounds:
        round_rows = [
            row for row in test_rows if int(row["communication_round"]) == round_id
        ]
        scalar_rows = [row for row in round_rows if row["method"] == "scalar"]
        class_rows = [row for row in round_rows if row["method"] == "class_conditional"]
        x_low = max(
            min(float(row["head_accuracy"]) for row in scalar_rows),
            min(float(row["head_accuracy"]) for row in class_rows),
        )
        x_high = min(
            max(float(row["head_accuracy"]) for row in scalar_rows),
            max(float(row["head_accuracy"]) for row in class_rows),
        )
        for method, method_rows in (("scalar", scalar_rows), ("class_conditional", class_rows)):
            for objective in OBJECTIVES:
                test_frontier_rows.extend({
                    "objective": objective,
                    "pareto": True,
                    **row,
                } for row in pareto_frontier(method_rows, objective))
        for objective in OBJECTIVES:
            scalar_auc = envelope_auc(scalar_rows, objective, x_low, x_high)
            class_auc = envelope_auc(class_rows, objective, x_low, x_high)
            head_span = x_high - x_low
            test_auc_rows.append({
                "split": "test",
                "communication_round": round_id,
                "objective": objective,
                "common_head_low": x_low,
                "common_head_high": x_high,
                "common_head_span": head_span,
                "scalar_envelope_auc": scalar_auc,
                "class_envelope_auc": class_auc,
                "class_minus_scalar_auc": class_auc - scalar_auc,
                "scalar_normalized_envelope_auc": scalar_auc / head_span if head_span > 0 else math.nan,
                "class_normalized_envelope_auc": class_auc / head_span if head_span > 0 else math.nan,
            })
        write_pareto_svg(
            args.output_dir / f"p0_pareto_round_{round_id:03d}.svg",
            round_id,
            round_rows,
        )
    write_csv(args.output_dir / "p0_test_pareto_frontier.csv", test_frontier_rows)
    write_csv(args.output_dir / "p0_test_pareto_auc.csv", test_auc_rows)

    pass_counts = {
        round_id: sum(
            bool(row["pass"])
            for row in budget_report
            if int(row["communication_round"]) == round_id
        )
        for round_id in rounds
    }
    positive_rounds = sum(
        count >= int(args.min_budgets_per_round) for count in pass_counts.values()
    )
    passed = positive_rounds >= int(args.min_positive_rounds)
    verdict = {
        **frozen_manifest,
        "new_test_grid_inference_accessed": True,
        "pass_thresholds": {
            "min_h_gain_pp": args.min_h_gain,
            "min_tail_gain_pp": args.min_tail_gain,
            "min_balanced_gain_pp": args.min_balanced_gain,
            "max_uncovered_tail_drop_pp": args.max_uncovered_drop,
            "min_budgets_per_round": args.min_budgets_per_round,
            "min_positive_rounds": args.min_positive_rounds,
        },
        "passing_budget_count_by_round": pass_counts,
        "positive_round_count": positive_rounds,
        "pareto_pass": passed,
        "verdict": (
            "P0_CLASS_CONDITIONAL_PARETO_SPACE_EXISTS"
            if passed
            else "P0_CLASS_CONDITIONAL_PARETO_SPACE_NOT_SUPPORTED"
        ),
        "next_action": (
            "Proceed to exploratory low-rank interaction V1a, then freeze on a fresh seed."
            if passed
            else "Stop per-class adapter, CLIP-router, and low-rank CCAR development on this path."
        ),
        "method_ready": False,
        "note": (
            "Seed-42 official test had already been inspected before P0 was designed; "
            "a passing result is exploratory and requires a frozen fresh-seed confirmation."
        ),
    }
    write_json(args.output_dir / "p0_verdict.json", verdict)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
