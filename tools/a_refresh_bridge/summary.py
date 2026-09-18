"""Four-cell audit with explicitly labeled standard or historical matched splits."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import yaml

from utils.cliplora_bridge_audit import write_csv, write_json

TOPOLOGIES = ("client-longtail", "noniid-labeldir-fine")
METHODS = ("c1", "c2")
FIELDS = ("overall_acc", "non_tail_acc", "bottom20_tail_acc")
REFRESH = list(range(10,100,10))


def read(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(row,key):
    value = row.get(key, "")
    return float(value) if value not in ("",None) else float("nan")


def spearman(x,y):
    def ranks(values):
        values = np.asarray(values,dtype=float)
        order = np.argsort(values,kind="stable")
        out = np.empty(len(values),dtype=float)
        for value in np.unique(values):
            positions = np.flatnonzero(values[order]==value)
            out[order[positions]] = positions.mean()+1
        return out
    x,y = ranks(x),ranks(y)
    return float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None


def gap(values):
    clt = "client-longtail"
    directory = next(t for t,m in values if t != clt)
    gb = values[(directory,"c1")]-values[(clt,"c1")]
    ga = values[(directory,"c2")]-values[(clt,"c2")]
    return {"G_B":gb,"G_A":ga,"delta_G":ga-gb,
            "CLT_C2_minus_C1":values[(clt,"c2")]-values[(clt,"c1")],
            "Dir_C2_minus_C1":values[(directory,"c2")]-values[(directory,"c1")]}


def summarize(root,seed=42,dirichlet_partition="noniid-labeldir-fine"):
    root = Path(root)
    out = root / "analysis" / f"seed{seed}"
    if dirichlet_partition == 'noniid-labeldir-fine':
        out = out / dirichlet_partition
    topologies = ('client-longtail',dirichlet_partition)
    matched = dirichlet_partition=='matched-dirichlet'
    paths = {(t,m):root/f"seed{seed}"/t/m for t in topologies for m in METHODS}
    meta, curves, stages, counts, manifests, events = {},{},{},{},{},{}
    checks, observations = {},{}
    performance, event_metrics = [],[]
    reference = (topologies[0],"c1")
    excluded_args = {"root","output_dir","client_schedule_file","partition","a_refresh_variant",
                     "capt_reset_global_before_client"}  # Unused by ClipLora; added after legacy CE runs.
    for cell,path in paths.items():
        t,m = cell
        meta[cell] = js(path/"bridge_metadata.json")
        checks[f"{t}/{m}/plain_CE_bridge"] = (
            meta[cell]['resolved_args'].get('lac_method','off')=='off'
            and not meta[cell]['resolved_args'].get('lac_partition_manifest',''))
        raw = read(path/"round_metrics.csv")
        checks[f"{t}/{m}/100_rounds"] = Counter(int(r["epoch"]) for r in raw)==Counter(range(-1,100))
        curves[cell] = {int(r["epoch"])+1:r for r in raw}
        stages[cell] = {(int(r["round"]),r["stage"]):r for r in read(path/"a_refresh_stage_metrics.csv")}
        checks[f"{t}/{m}/stages"] = set(stages[cell])=={(r,s) for r in REFRESH for s in ("normal_pre","after_B_aggregation","after_refresh")}
        checks[f"{t}/{m}/stage_metrics"] = all(abs(number(stages[cell][(r,"after_refresh")],f)-number(curves[cell][r],f))<1e-8 for r in REFRESH for f in FIELDS)
        manifests[cell] = read(path/"partition_manifest.csv")
        matrix = np.zeros((30,100),dtype=int)
        for row in manifests[cell]:
            matrix[int(row["client_id"]),int(row["class_id"])] += 1
        counts[cell] = matrix
        ids = [int(r["raw_sample_id"]) for r in manifests[cell]]
        checks[f"{t}/{m}/unique_samples"] = len(ids)==len(set(ids))==10847
        probes = {int(r["raw_train_index"]) for r in read(path/"protocol/probe_manifest.csv")}
        checks[f"{t}/{m}/probe_disjoint"] = not (set(ids)&probes)
        budget = read(path/"a_refresh_budget.csv")
        steps_per_epoch = int(np.ceil(matrix.sum(1)/32).sum())
        checks[f"{t}/{m}/normal_budget"] = sum(int(r["optimizer_steps"]) for r in budget if r["phase"]=="normal_B")==100*3*steps_per_epoch
        checks[f"{t}/{m}/refresh_budget"] = sum(int(r["optimizer_steps"]) for r in budget if r["phase"]=="refresh")==9*steps_per_epoch
        checks[f"{t}/{m}/unique_budget"] = len(budget)==len({(r["round"],r["client_id"],r["phase"]) for r in budget})==3270
        progress = js(path/"a_refresh_progress.json")
        checks[f"{t}/{m}/completed_progress"] = progress["completed_round"]==100 and progress["completed_refresh_rounds"]==REFRESH
        checks[f"{t}/{m}/probe_budget"] = progress["probe_sample_presentations"]==292869
        events[cell] = {(int(v["round"]),v["phase"]):v for p in (path/"bridge_dumps").glob("round_*/*/event.json") for v in [js(p)]}
        refresh_phase = "extra_B" if m=="c1" else "refresh_A"
        checks[f"{t}/{m}/109_events"] = set(events[cell])=={(r,"normal_B") for r in range(1,101)}|{(r,refresh_phase) for r in REFRESH}
        checks[f"{t}/{m}/reconstruction"] = all(e["reconstruction_passed"] for e in events[cell].values())
        last_hash = meta[cell]["initial_lora_sha256"]
        continuity = True
        for r in range(1,101):
            for phase in (["normal_B",refresh_phase] if r in REFRESH else ["normal_B"]):
                e = events[cell][(r,phase)]
                continuity &= e["before_lora_sha256"]==last_hash
                last_hash = e["after_lora_sha256"]
                q = np.array(e["server_weights"])
                order = e["selected_client_ids"]
                continuity &= order == meta[cell]["schedule"][r-1]
                continuity &= np.allclose(q,matrix.sum(1)[order]/matrix.sum(),rtol=0,atol=1e-12)
        checks[f"{t}/{m}/state_schedule_weight_chain"] = bool(continuity)
        normal_calls = [(r["round"],r["client_id"],r["optimizer_steps"]) for r in budget if r["phase"]=="normal_B"]
        observations[f"{t}/{m}/normal_calls"] = normal_calls
        for f in FIELDS:
            values = {r:number(row,f) for r,row in curves[cell].items()}
            peak = max(range(1,101),key=values.get)
            performance.append({"seed":seed,"topology":t,"method":m,"metric":f,
                "initial":values[0],"final":values[100],"last20":mean(values[r] for r in range(81,101)),
                "last10":mean(values[r] for r in range(91,101)),"final_gain":values[100]-values[0],
                "last20_gain":mean(values[r] for r in range(81,101))-values[0],
                "peak":values[peak],"peak_round":peak,
                "best_to_last20_drop":values[peak]-mean(values[r] for r in range(81,101))})
        for r in REFRESH:
            row = {"seed":seed,"topology":t,"method":m,"round":r}
            for f in FIELDS:
                pre = number(stages[cell][(r,"normal_pre")],f)
                middle = number(stages[cell][(r,"after_B_aggregation")],f)
                after = number(stages[cell][(r,"after_refresh")],f)
                end = number(stages[cell][(r+10,"after_B_aggregation")],f) if r<90 else number(curves[cell][100],f)
                row.update({f+"_pre":pre,f+"_middle":middle,f+"_after":after,
                    f+"_normal_delta":middle-pre,f+"_refresh_delta":after-middle,
                    f+"_next_B_delta":number(curves[cell][r+1],f)-after,
                    f+"_following_B_interval_delta":end-after})
            event_metrics.append(row)
    for key in ("initial_lora_sha256","frozen_model_sha256","schedule_sha256","pool_sha256",
                "test_sha256","probe_images_sha256","probe_manifest_sha256"):
        checks[key+"_equal"] = len({v[key] for v in meta.values()})==1
    def normalize_args(value):
        # Legacy bridge metadata predates the inactive LA-controller options.
        return {k:v for k,v in value["resolved_args"].items()
                if k not in excluded_args and not k.startswith('lac_')}
    checks["training_args_matched"] = all(normalize_args(v)==normalize_args(meta[reference]) for v in meta.values())
    def normalize_cfg(value):
        config = yaml.safe_load(value["resolved_config"])
        for key in ("OUTPUT_DIR","RESUME"):
            config.pop(key,None)
        for key in ("ROOT","imagenetROOT","PARTITION","PARTITION_MANIFEST"):
            config["DATASET"].pop(key,None)
        return config
    checks["resolved_configs_matched"] = all(normalize_cfg(v)==normalize_cfg(meta[reference]) for v in meta.values())
    observations["training_code_matched"] = all(v["training_code_hashes"]==meta[reference]["training_code_hashes"] for v in meta.values())
    observations["normal_client_steps_matched_across_topologies"] = len({json.dumps(v) for k,v in observations.items() if k.endswith("normal_calls")})==1
    for t in topologies:
        checks[t+"/C1_C2_membership"] = manifests[(t,"c1")]==manifests[(t,"c2")]
        checks[t+"/C1_C2_normal_steps"] = observations[f'{t}/c1/normal_calls']==observations[f'{t}/c2/normal_calls']
        observations[t+"/first9_max_metric_difference"] = max(abs(number(curves[(t,"c1")][r],f)-number(curves[(t,"c2")][r],f)) for r in range(10) for f in FIELDS)
    for cell,matrix in counts.items():
        equal_sizes = bool(np.array_equal(matrix.sum(1),counts[reference].sum(1)))
        observations[str(cell)+"/nk_equal_to_CLT"] = equal_sizes
        if matched:
            checks[str(cell)+"/nk"] = equal_sizes
        checks[str(cell)+"/nc"] = bool(np.array_equal(matrix.sum(0),counts[reference].sum(0)))
        checks[str(cell)+"/global_ids"] = sorted(int(r["raw_sample_id"]) for r in manifests[cell])==sorted(int(r["raw_sample_id"]) for r in manifests[reference])
    checks["coupling_changed"] = not np.array_equal(counts[reference],counts[(topologies[1],"c1")])
    observations = {k:v for k,v in observations.items() if not k.endswith("normal_calls")}
    write_json(out/"protocol_audit.json",{"valid":all(checks.values()),"checks":checks,"observations":observations})
    assert all(checks.values()), f"Protocol mismatch; inspect {out/'protocol_audit.json'}"
    write_csv(out/"factorial_performance.csv",performance)
    write_csv(out/"phase_accuracy_changes.csv",event_metrics)
    interactions = [{"seed":seed,"round":r,"metric":f,**gap({cell:number(curves[cell][r],f) for cell in paths})}
                    for r in range(101) for f in FIELDS]
    write_csv(out/"coupling_gap_by_round.csv",interactions)
    endpoints = [{"seed":seed,"endpoint":endpoint,"metric":f,**gap({cell:next(row[endpoint] for row in performance if (row["topology"],row["method"])==cell and row["metric"]==f) for cell in paths})}
                 for endpoint in ("final","last20","last10") for f in FIELDS]
    write_csv(out/"coupling_gap_endpoints.csv",endpoints)
    immediate = [{"seed":seed,"round":r,"metric":f,**gap({(row["topology"],row["method"]):row[f+"_refresh_delta"] for row in event_metrics if row["round"]==r})}
                 for r in REFRESH for f in FIELDS]
    write_csv(out/"refresh_event_interactions.csv",immediate)
    decomposition = []
    for f in FIELDS:
        final = next(r["delta_G"] for r in interactions if r["round"]==100 and r["metric"]==f)
        initial = next(r["delta_G"] for r in interactions if r["round"]==0 and r["metric"]==f)
        refresh_sum = sum(r["delta_G"] for r in immediate if r["metric"]==f)
        decomposition.append({"metric":f,"final_minus_initial_delta_G":final-initial,
                              "refresh_interaction_sum":refresh_sum,"remaining_normal_interaction":final-initial-refresh_sum})
    write_csv(out/"interaction_trajectory_decomposition.csv",decomposition)

    structure, correlations = [],[]
    for cell,path in paths.items():
        matrix = counts[cell]
        q = matrix.sum(1)/matrix.sum()
        per_stage = {(int(r["round"]),r["stage"],int(r["class_id"])):float(r["accuracy"])
                     for r in read(path/"a_refresh_stage_per_class.csv")}
        per_last = {r:{int(v["class_id"]):float(v["per_class_acc"]) for v in read(path/f"per_class_accuracy_epoch_{r-1}.csv")} for r in range(81,101)}
        for c in range(80,100):
            support = matrix[:,c]>0
            access = q[support].sum()
            structure.append({"seed":seed,"topology":cell[0],"method":cell[1],"class_id":c,
                "n_c":int(matrix[:,c].sum()),"N_c":int(support.sum()),
                "N_eff_ownership":float(matrix[:,c].sum()**2/(matrix[:,c]**2).sum()),
                "support_access":float(access),"N_eff_access":float(access**2/(q[support]**2).sum()),
                "refresh_acc_delta_mean":mean(per_stage[(r,"after_refresh",c)]-per_stage[(r,"after_B_aggregation",c)] for r in REFRESH),
                "final_accuracy":per_last[100][c],"last20_accuracy":mean(per_last[r][c] for r in per_last)})
    write_csv(out/"class_structure_relation.csv",structure)
    for t in topologies:
        rows = [r for r in structure if r["topology"]==t and r["method"]=="c2"]
        correlations.append({"topology":t,"relation":"C2_access_vs_mean_refresh_acc_delta",
                             "spearman":spearman([r["support_access"] for r in rows],[r["refresh_acc_delta_mean"] for r in rows]),"classes":20})
    paired = []
    for c in range(80,100):
        index = {(r["topology"],r["method"]):r for r in structure if r["class_id"]==c}
        paired.append({"class_id":c,"access_Dir_minus_CLT":index[(topologies[1],"c1")]["support_access"]-index[reference]["support_access"],
                       **gap({k:r["last20_accuracy"] for k,r in index.items()})})
    correlations.append({"topology":"paired","relation":"delta_access_vs_last20_class_DiD",
                         "spearman":spearman([r["access_Dir_minus_CLT"] for r in paired],[r["delta_G"] for r in paired]),"classes":20})
    write_csv(out/"paired_class_structure.csv",paired)
    write_csv(out/"exploratory_correlations.csv",correlations)

    mechanism_ready = all((p/"analysis/attribution_summary.json").exists() for p in paths.values())
    mechanism_valid = False
    factors, budgets = [],[]
    if mechanism_ready:
        manifests_a = [js(p/"analysis/attribution_summary.json") for p in paths.values()]
        mechanism_valid = all(v["valid"] for v in manifests_a) and len({tuple(v["normal_rounds"]) for v in manifests_a})==1
        for filename in ("phase_client_effects","phase_class_budgets","attribution_validity","phase_update_norms"):
            rows = [r for path in paths.values() for r in read(path/f"analysis/{filename}.csv")]
            write_csv(out/f"{filename}.csv",rows)
            if filename=="phase_class_budgets":
                budgets = rows
        if mechanism_valid:
            groups = defaultdict(list)
            for r in budgets:
                for c in (int(r["class_id"]),"tail_macro"):
                    groups[(r["topology"],r["method"],r["phase"],c)].append(r)
            for (t,m,phase,c),rows in groups.items():
                w,a,p = [mean(number(r,k) for r in rows) for k in ("W","support_access","positive_support_access")]
                rho,mu = (p/a if a else float("nan")),(w/p if p else float("nan"))
                event_count = len({r["round"] for r in rows})
                factors.append({"seed":seed,"topology":t,"method":m,"phase":phase,"class_id":c,
                    "event_count":event_count,"all_phase_events_observed":event_count==(100 if phase=="normal_B" else 9),
                    "support_access_mean":a,"positive_support_access_mean":p,"rho":rho,"mu":mu,
                    "factorization_abs_error":abs(w-a*rho*mu) if p else None,
                    **{k+"_mean":mean(number(r,k) for r in rows) for k in ("W","H","D","R")},
                    **{k+"_observed_event_sum_per_class":mean(number(r,k) for r in rows)*event_count for k in ("W","H","D","R")}})
            write_csv(out/"phase_factorization.csv",factors)
    report = {"seed":seed,"protocol_valid":True,"mechanism_ready":mechanism_ready,"mechanism_valid":mechanism_valid,
              "dirichlet_partition":dirichlet_partition,"fixed_client_margins":matched,
              "training_code_matched":observations['training_code_matched'],
              "primary_endpoint":"last20","gap_sign":"Dir-minus-CLT","independent_seeds":1,
              "coupling_gap_endpoints":endpoints,"trajectory_decomposition":decomposition,
              "warning":"Single-seed exploratory analysis; sparse normal audits are not full-trajectory cumulative attribution."}
    write_json(out/"summary.json",report)
    plot(out,curves,interactions,immediate,factors)
    lines = ["# C1/C2 topology bridge", "", f"Seed {seed}; primary endpoint: rounds 81–100 mean. Accuracy differences are percentage points.",
             f"Dirichlet: {dirichlet_partition}; fixed client margins: {matched}.",
             f"Training code hashes equal: {observations['training_code_matched']}; cross-code pairs are descriptive comparisons.",
             "Standard Dirichlet keeps the global pool but allows client sizes, aggregation weights and step counts to change.","",
             "| Topology | Method | Overall | Non-tail | Tail |","|---|---|---:|---:|---:|"]
    for cell in paths:
        vals = [next(r["last20"] for r in performance if (r["topology"],r["method"])==cell and r["metric"]==f) for f in FIELDS]
        lines.append(f"| {cell[0]} | {cell[1]} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |")
    lines += ["","| Metric | G_B | G_A | delta G |","|---|---:|---:|---:|"]
    for r in endpoints:
        if r["endpoint"]=="last20":
            lines.append(f"| {r['metric']} | {r['G_B']:.4f} | {r['G_A']:.4f} | {r['delta_G']:.4f} |")
    lines += ["",f"Protocol audit passed. Four-cell attribution available: {mechanism_ready}; valid and matched: {mechanism_valid}.","",
              "![Training curves](training_curves.png)","![Coupling gaps](coupling_gaps.png)",
              "![Refresh interaction](refresh_interactions.png)","",
              "Do not interpret sparse normal-event sums as 100-round cumulative contributions. Refresh sums cover all nine events.",
              "Access is constant within each fixed topology under full participation. Rho/mu and effects depend on the model state.",
              "Trajectory decomposition is temporal bookkeeping, not independent causal responsibility. No across-seed significance is estimated."]
    if factors:
        lines += ["", "![Phase mechanisms](phase_mechanisms.png)"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Summary written: {out/'report.md'}",flush=True)


def plot(out,curves,interactions,immediate,factors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    for ax,f in zip(axes,FIELDS):
        for (t,m),rows in curves.items():
            ax.plot(range(101),[number(rows[r],f) for r in range(101)],label=f"{'CLT' if t==TOPOLOGIES[0] else 'Dir'}-{m.upper()}")
        ax.set(title=f,xlabel="Round",ylabel="Accuracy (%)")
        ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); fig.savefig(out/"training_curves.png",dpi=160); plt.close(fig)
    for name,source in (("coupling_gaps",interactions),("refresh_interactions",immediate)):
        fig,axes = plt.subplots(1,3,figsize=(15,4))
        for ax,f in zip(axes,FIELDS):
            rows = [r for r in source if r["metric"]==f]
            for key in ("G_B","G_A","delta_G"):
                ax.plot([r["round"] for r in rows],[r[key] for r in rows],label=key)
            ax.axhline(0,color="black",linewidth=.7); ax.set(title=f,xlabel="Round",ylabel="Percentage points"); ax.grid(alpha=.2)
        axes[0].legend(); fig.tight_layout(); fig.savefig(out/f"{name}.png",dpi=160); plt.close(fig)
    if factors:
        fig,axes = plt.subplots(2,4,figsize=(19,8))
        for i,normal in enumerate((True,False)):
            rows = [r for r in factors if r["class_id"]=="tail_macro" and (r["phase"]=="normal_B")==normal]
            labels = [f"{'CLT' if r['topology']==TOPOLOGIES[0] else 'Dir'}-{r['method']}" for r in rows]
            x = np.arange(len(rows))
            for j,key in enumerate(("W_mean","H_mean","D_mean","R_mean")):
                axes[i,0].bar(x+(j-1.5)*.18,[r[key] for r in rows],width=.18,label=key)
            for j,key in enumerate(("support_access_mean","rho","mu"),1):
                axes[i,j].bar(x,[r[key] for r in rows],label=key)
            for ax in axes[i]:
                ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title("Normal B" if normal else "Extra B / A refresh"); ax.legend(); ax.grid(alpha=.15)
        fig.tight_layout(); fig.savefig(out/"phase_mechanisms.png",dpi=160); plt.close(fig)
