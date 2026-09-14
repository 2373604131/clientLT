"""Offline attribution on exact A/B anchors, using train-only held-out probes."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.eri_closure.analysis import TrainOnlyFunctionalEvaluator
from tools.eri_closure.attribution import integrated_client_effects, first_order_client_effects, rows_from_effects
from utils.cusp_minimal import make_flat_spec, flatten_state
from utils.cliplora_a_refresh import state_hash, train_only
from utils.cliplora_bridge_audit import write_json, write_csv


def attribute_run(run, data_root, device="cuda", normal_rounds=(10,20,40,60,80,90,100), segments=1, event_paths=None):
    from federated_main import setup_cfg
    from trainers.cliplora import build_cliplora_model

    run = Path(run)
    meta = json.loads((run / "bridge_metadata.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((run/"protocol/probe_manifest.csv").read_bytes()).hexdigest()==meta["probe_manifest_sha256"]
    args = SimpleNamespace(**meta["resolved_args"])
    args.root, args.output_dir = str(data_root), str(run / "analysis/model_build")
    cfg = setup_cfg(args)
    model = build_cliplora_model(cfg, meta["classnames"]).to(device).eval()
    all_keys = sorted(k for k in model.state_dict() if k.endswith(("_lora_A", "_lora_B")))
    frozen = sorted(set(model.state_dict()) - set(all_keys))
    assert state_hash(model.state_dict(), frozen) == meta["frozen_model_sha256"], "Frozen backbone/config differs from training"
    assert segments >= 1
    if event_paths is None:
        files = sorted((run / "bridge_dumps").glob("round_*/*/state.pt"))
        chosen = [p for p in files if p.parent.name != "normal_B" or int(p.parents[1].name.split("_")[1]) in normal_rounds]
        expected = {(r,"normal_B") for r in normal_rounds} | {(r,"extra_B" if meta["method"]=="c1" else "refresh_A") for r in range(10,100,10)}
        assert {(int(p.parents[1].name.split("_")[1]),p.parent.name) for p in chosen} == expected, "Missing requested phase dumps"
    else:
        chosen = list(map(Path,event_paths))
    client_all, budget_all, validity_all, event_all, first_all = [], [], [], [], []
    evaluator = None
    for number, path in enumerate(chosen, 1):
        print(f"[{number}/{len(chosen)}] attribute {path.parent.parent.name}/{path.parent.name}", flush=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        identity = {k:payload[k] for k in ("seed","topology","method","round","phase")}
        identity.update({k:payload[k] for k in ("event_id","branch","candidate_round") if k in payload})
        spec = make_flat_spec(payload["anchor_lora_state"], payload["active_keys"])
        assert sorted(payload["anchor_lora_state"]) == all_keys
        model.load_state_dict(payload["anchor_lora_state"], strict=False)
        train_only(model, payload["active_factor"])
        if evaluator is None:
            from tools.eri_closure.protocol import load_protocol
            from utils.functional_coverage_validation import _TrainOnlyCifar100, _locate_cifar100
            _, probe_rows = load_protocol(run/"protocol")
            probe_store = _TrainOnlyCifar100(_locate_cifar100(Path(data_root)))
            digest = hashlib.sha256()
            for row in probe_rows:
                digest.update(probe_store.images[int(row["raw_train_index"])].tobytes())
            assert digest.hexdigest()==meta["probe_images_sha256"], "Probe images differ from the training-time manifest"
            evaluator = TrainOnlyFunctionalEvaluator(cfg=cfg, trainer=SimpleNamespace(model=model),
                payload={"flatten_spec":spec.as_dict()}, protocol_dir=run/"protocol",
                data_root=data_root, device=device, batch_size=10)
        evaluator.spec = spec
        before = flatten_state(payload["anchor_lora_state"], spec)
        after = flatten_state(payload["actual_after_lora_state"], spec)
        # Use exact local-minus-anchor in double for normal parameter averaging;
        # refresh deltas are the actual uploaded float32 deltas.
        if payload["phase"] == "normal_B":
            deltas = torch.stack([flatten_state(s,spec)-before for s in payload["local_factor_states"]])
        else:
            deltas = torch.stack([flatten_state(s,spec) for s in payload["local_factor_deltas"]])
        q = payload["server_weights"]
        aggregate = (q[:,None]*deltas).sum(0)
        effects = torch.zeros((len(evaluator.class_ids),len(q)),dtype=torch.float64)
        for part in range(segments):
            values, _ = integrated_client_effects(before+aggregate*(part/segments), deltas/segments,
                        q, evaluator.class_ids, evaluator.gradient, quadrature_points=8)
            effects += values
        first, _ = first_order_client_effects(before,deltas,q,evaluator.class_ids,evaluator.gradient)
        clients, budgets = rows_from_effects(effects,evaluator.class_ids,payload["selected_client_ids"],
                            payload["client_class_counts"],communication_round=payload["round"],
                            method=payload["method"],aggregation_weights=q)
        first_clients, first_budgets = rows_from_effects(first,evaluator.class_ids,payload["selected_client_ids"],
                            payload["client_class_counts"],communication_round=payload["round"],
                            method=payload["method"],aggregation_weights=q)
        checks = []
        for ci,c in enumerate(evaluator.class_ids):
            f0 = evaluator.metric(before,c)
            fpath = evaluator.metric(before+aggregate,c)
            factual = evaluator.metric(after,c)
            absolute = abs(float(effects[ci].sum())-(fpath-f0))
            tolerance = 1e-4+0.01*float(effects[ci].abs().sum())
            endpoint_error = abs(factual-fpath)
            valid = bool(payload["reconstruction_passed"] and absolute<=tolerance and endpoint_error<=1e-4)
            checks.append({**identity,"class_id":c,"before_F":f0,"path_after_F":fpath,
                "actual_after_F":factual,"direct_change":factual-f0,
                "attributed_change":float(effects[ci].sum()),"absolute_completeness_error":absolute,
                "completeness_tolerance":tolerance,"candidate_vs_trained_endpoint_error":endpoint_error,
                "valid":valid,"quadrature_nodes":8,"quadrature_segments":segments})
            budgets[ci].update(direct_change=factual-f0, valid=valid,
                positive_budget_near_zero=budgets[ci]["positive_refresh"]<=1e-8)
        for rows in (clients,budgets,first_clients,first_budgets):
            for row in rows:
                row.update(identity)
        event = {**identity, **{k:payload[k] for k in ("effective_delta_norm","delta_a_norm","delta_b_norm",
                  "new_direction_fraction","a_drift_from_initial","mean_a_subspace_from_initial",
                  "reconstruction_max_abs_error","frozen_factor_max_abs_error")},
                 "valid":all(r["valid"] for r in checks)}
        destination = (run / "analysis/events" / payload['event_id'] if event_paths is not None
                       else run / "analysis/events" / path.parents[1].name / payload["phase"])
        for name, rows in (("client_effects",clients),("class_budgets",budgets),("validity",checks),
                           ("first_order_client_effects",first_clients),("first_order_budgets",first_budgets)):
            write_csv(destination/f"{name}.csv",rows)
        write_json(destination/"event.json",event)
        client_all.extend(clients); budget_all.extend(budgets); validity_all.extend(checks)
        first_all.extend(first_budgets); event_all.append(event)
        print(f"  valid={event['valid']} max closure={max(r['absolute_completeness_error'] for r in checks):.3g}",flush=True)
    for name,rows in (("phase_client_effects",client_all),("phase_class_budgets",budget_all),
                     ("attribution_validity",validity_all),("phase_update_norms",event_all),
                     ("first_order_budgets",first_all)):
        write_csv(run/"analysis"/f"{name}.csv",rows)
    result = {"normal_rounds":list(normal_rounds),"refresh_rounds":list(range(10,100,10)),
              "quadrature_nodes":8,"quadrature_segments":segments,"event_count":len(event_all),
              "valid":all(r["valid"] for r in event_all),"invalid_class_events":sum(not r["valid"] for r in validity_all)}
    write_json(run/"analysis/attribution_summary.json",result)
    if not result["valid"]:
        raise RuntimeError("Attribution validity failed; inspect attribution_validity.csv and rerun with --quadrature-segments 2. Training is already complete.")
    return result
