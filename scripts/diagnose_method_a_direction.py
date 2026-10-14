"""Four fixed, offline direction probes. No training, correction or model selection."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.diagnose_method_a_round80 import read_json, write_json, saved_event, worker_arguments, verify_initial_numerics

RHOS = (0., .01, .03, .10)


def capture_feedback(runtime, state, targets, phase, gradient=False):
    """Observe original CE outputs without changing their graph or reduction.

    Store per-sample losses and reduce diagnostic group summaries in CPU FP64,
    avoiding the nondeterministic FP32 scatter_add used in production logs.
    """
    import torch
    from utils import method_a_margin_feedback as feedback_module
    feedback = runtime.margin_feedback
    batches = list(runtime.bank._fast_feedback.groups(None, True, True))
    ce_values = torch.full((len(feedback.manifest), 2), float('nan'))
    la_margins = torch.empty_like(ce_values)
    original_ce = feedback_module.F.cross_entropy
    calls = 0

    def capture(logits, labels, *args, **kwargs):
        nonlocal calls
        losses = original_ce(logits, labels, *args, **kwargs)
        batch = batches[calls]
        ids = torch.cat([torch.arange(feedback.offsets[q], feedback.offsets[q]+n) for q, v, n, _ in batch])
        views = torch.cat([torch.full((n,), v, dtype=torch.long) for q, v, n, _ in batch])
        if losses.shape != ids.shape:
            raise ValueError('Unexpected CE observation shape')
        ce_values[ids, views] = losses.detach().cpu()
        values, labels_cpu = logits.detach().cpu(), labels.detach().cpu()
        wrong = values.clone()
        wrong.scatter_(1, labels_cpu[:, None], -torch.inf)
        la_margins[ids, views] = values.gather(1, labels_cpu[:, None]).flatten()-wrong.max(1).values
        calls += 1
        return losses

    with patch.object(feedback_module.F, 'cross_entropy', capture):
        observed = runtime.observe_margin(state, 80, phase, targets=targets, gradient=gradient, diagnostics=True)
    if calls != len(batches) or not torch.isfinite(ce_values).all():
        raise ValueError('Incomplete per-witness loss observation')
    observed.update(sample_ce=ce_values, la_margins=la_margins)
    group_ids = feedback.group_ids.cpu()
    mass = feedback.ce_mass.detach().cpu().double()
    groups = [(ce_values.double()[group_ids==g]*mass[group_ids==g, None]).sum().item() for g in range(3)]
    observed['stable_group_ce'] = groups
    observed['stable_class_loss'] = sum(value*len(classes)/sum(map(len, feedback.groups))
                                      for value, classes in zip(groups, feedback.groups))
    gap = (targets['target'].double()-observed['sample_margins'].double()).clamp_min(0)
    observed['stable_keep_loss'] = float((.25*targets['omega'].double()[:, None]*(gap/targets['sigma'].double()).square()).sum())
    return observed


def transition_summary(before, after, mask):
    from utils.method_a_update_diagnostic import transitions
    result = transitions(before[mask], after[mask])
    for name, a, b in (('views', before[mask], after[mask]),
                       ('original_view', before[mask, 0], after[mask, 0]),
                       ('both_views_correct', before[mask].all(1), after[mask].all(1))):
        result[name].update(before_correct=int(a.sum()), after_correct=int(b.sum()))
    return result


def summarize_candidate(runtime, before, observed, targets, ordinary, middle, candidate, rho):
    import torch
    from utils.method_a_update_diagnostic import norm, difference
    feedback = runtime.margin_feedback
    valid = targets['valid']
    before_gap = (targets['target']-before['sample_margins']).clamp_min(0)
    gap = (targets['target']-observed['sample_margins']).clamp_min(0)
    violated_before = (before_gap>0).any(1) & valid
    violated = (gap>0).any(1) & valid
    group_ids = feedback.group_ids.cpu()
    classes = torch.tensor([x['class_id'] for x in feedback.manifest])
    masks = dict(all_witnesses=torch.ones_like(valid), historical_correct=valid,
                 without_valid_history=~valid, original_633_deficits=violated_before,
                 previously_protected=valid & ~violated_before)
    rows = dict(rho=rho, keep_loss=observed['keep_loss'], class_loss=observed['class_loss'],
                stable_keep_loss=observed['stable_keep_loss'], stable_class_loss=observed['stable_class_loss'],
                keep_change=observed['stable_keep_loss']-before['stable_keep_loss'],
                class_change=observed['stable_class_loss']-before['stable_class_loss'],
                violated_count=int(violated.sum()), original_deficits=int(violated_before.sum()),
                repaired_deficits=int((violated_before & ~violated).sum()),
                new_deficits=int((~violated_before & violated).sum()),
                violated_views=int(((gap>0) & valid[:, None]).sum()),
                mean_history_gap=float(gap[valid].double().mean()),
                max_history_gap=float(gap[valid].max()),
                group_classification={}, transitions={}, flip_details=[])
    for group, name in enumerate(('Head', 'Middle', 'Tail')):
        selected = group_ids==group
        rows['group_classification'][name] = dict(
            before=before['stable_group_ce'][group], after=observed['stable_group_ce'][group],
            change=observed['stable_group_ce'][group]-before['stable_group_ce'][group],
            violated_before=int((violated_before & selected).sum()), violated_after=int((violated & selected).sum()),
            repaired_deficits=int((violated_before & ~violated & selected).sum()),
            new_deficits=int((~violated_before & violated & selected).sum()),
            raw=transition_summary(before['correct'], observed['correct'], selected),
            LA=transition_summary(before['la_correct'], observed['la_correct'], selected),
            historical_raw=transition_summary(before['correct'], observed['correct'], selected & valid),
            historical_LA=transition_summary(before['la_correct'], observed['la_correct'], selected & valid))
    rows['per_class'] = []
    for c in classes.unique().tolist():
        selected = classes==c
        mass = feedback.ce_mass.detach().cpu().double()[selected]
        old = float((before['sample_ce'][selected].double()*mass[:, None]).sum()/ (2*mass.sum()))
        new = float((observed['sample_ce'][selected].double()*mass[:, None]).sum()/ (2*mass.sum()))
        rows['per_class'].append(dict(class_id=c, before=old, after=new, change=new-old))
    for name, mask in masks.items():
        rows['transitions'][name] = dict(samples=int(mask.sum()),
            raw=transition_summary(before['correct'], observed['correct'], mask),
            LA=transition_summary(before['la_correct'], observed['la_correct'], mask))
    for classifier, field, margins in (('raw', 'correct', 'sample_margins'), ('LA', 'la_correct', 'la_margins')):
        for index, view in (before[field] != observed[field]).nonzero().tolist():
            rows['flip_details'].append(dict(**feedback.manifest[index], classifier=classifier, view=view,
                group=('Head', 'Middle', 'Tail')[int(group_ids[index])], historical_correct=bool(valid[index]),
                original_deficit=bool(violated_before[index]),
                transition='correct_to_wrong' if before[field][index, view] else 'wrong_to_correct',
                margin_before=float(before[margins][index, view]), margin_after=float(observed[margins][index, view])))
    modules = dict(runtime.trainer.model.named_modules())
    layers = []
    for key in runtime.a_keys:
        bkey = key[:-1]+'B'
        if not torch.equal(candidate[bkey], ordinary[bkey]):
            raise ValueError('Candidate changed B')
        b = ordinary[bkey].cpu().double()
        scale = float(modules[key.rsplit('.', 1)[0]].scaling)
        delta = candidate[key].cpu().double()-ordinary[key].cpu().double()
        reference = ordinary[key].cpu().double()-middle[key].cpu().double()
        layers.append(dict(layer=key, scaling=scale, A_norm=norm(delta),
                           effective_norm=norm(scale*(b@delta)), ordinary_effective_norm=norm(scale*(b@reference))))
    delta = difference(candidate, ordinary, runtime.a_keys)
    ordinary_norm = norm(difference(ordinary, middle, runtime.a_keys))
    effective = math.sqrt(sum(row['effective_norm']**2 for row in layers))
    ordinary_effective = math.sqrt(sum(row['ordinary_effective_norm']**2 for row in layers))
    actual = runtime.trainer.model.state_dict()
    rows.update(delta_A_norm=norm(delta), ordinary_delta_A_norm=ordinary_norm,
        actual_A_ratio=norm(delta)/ordinary_norm, effective_delta_norm=effective,
        ordinary_effective_delta_norm=ordinary_effective, effective_ratio=effective/ordinary_effective,
        B_unchanged=True, layers=layers,
        loaded_model_writeback_max_abs_error=max(float((actual[k].cpu()-candidate[k].cpu()).abs().max()) for k in runtime.keys))
    return rows


def evaluate_direction(runtime, source, prior, out):
    import torch
    from utils.method_a_margin_math import SampleMarginHistory, make_targets
    from utils.method_a_update_diagnostic import norm, difference
    from utils.sfra_math import flatten, unflatten
    from utils.cliplora_a_refresh import state_hash
    prior_result = read_json(prior/'diagnosis.json')
    prior_audit = read_json(prior/'independent_audit.json')
    if not (prior_result['status']=='single_saved_round_replayed' and
            prior_audit['submitted_A_bitwise_equals_original'] and prior_audit['all_history_counts_match']):
        raise ValueError('The prior exact round80 replay is required')
    runtime.base_state = torch.load(source/'checkpoints/base_model.pt', map_location='cpu', weights_only=False)
    frozen_keys = sorted(set(runtime.base_state)-set(runtime.keys))
    if state_hash(runtime.base_state, frozen_keys) != runtime.audit.meta['frozen_model_sha256']:
        raise ValueError('Frozen backbone differs')
    if runtime.margin_feedback.manifest != read_json(source/'margin_witness_manifest.json'):
        raise ValueError('Witness manifest differs')
    # One initial forward restores the previously verified frozen prefix. No history replay.
    initial = dict(runtime.base_state, **saved_event(source, 1, 'normal_B', runtime.keys)['anchor_lora_state'])
    runtime.initial_numerics = []
    initial_obs = runtime.observe_margin(initial, 0, 'verify_cached_prefix')
    verify_initial_numerics(runtime, source, initial, initial_obs)
    history_payload = torch.load(prior/'history_before_r080.pt', map_location='cpu', weights_only=False)
    runtime.history = SampleMarginHistory.restore(history_payload, runtime.margin_feedback.manifest)
    if runtime.history.last_committed_round != 79 or int(runtime.history.valid.sum()) != 4227:
        raise ValueError('Incorrect prior history snapshot')
    event = saved_event(source, 80, 'refresh_A', runtime.keys)
    middle = dict(runtime.base_state, **event['anchor_lora_state'])
    ordinary = dict(runtime.base_state, **event['actual_after_lora_state'])
    prior_commit = torch.load(prior/'diagnosed_commit.pt', map_location='cpu', weights_only=False)
    if state_hash(ordinary, runtime.keys) != state_hash(prior_commit['ordinary_lora'], runtime.keys):
        raise ValueError('Ordinary model differs from prior replay')
    targets = make_targets(runtime.history, runtime.margin_feedback.base_mass,
                           runtime.sfra_config['epsilon'], runtime.sfra_config['margin_scale'])
    targets_copy = copy.deepcopy(targets)
    before = capture_feedback(runtime, ordinary, targets, 'direction_baseline_gradient', gradient=True)
    if before['deficit_count'] != 633 or abs(before['keep_loss']-prior_result['L_keep']) > 1e-7:
        raise ValueError('Baseline does not reproduce the original protection state')
    gradient = before['gradient'].detach().cpu().double()
    g_norm = norm(gradient)
    if abs(g_norm-prior_result['steps'][0]['grad_total_norm_A']) > 1e-7 or not g_norm:
        raise ValueError('Baseline gradient differs from verified round80 replay')
    direction = -gradient/g_norm
    ordinary_norm = norm(difference(ordinary, middle, runtime.a_keys))
    ordinary_a = flatten(ordinary, runtime.a_keys).cpu().double()
    dev = getattr(runtime.trainer.dm, 'val_loader', None)
    if dev is not None:
        raise ValueError('Unexpected validation loader: require independent-data provenance before evaluation')
    result = dict(status='offline_direction_diagnosis_running', round=80, rhos=list(RHOS),
        source_run=str(source), prior_replay=str(prior), history_last_used=79,
        history_sha256=hashlib.sha256((prior/'history_before_r080.pt').read_bytes()).hexdigest(),
        witness_count=len(runtime.margin_feedback.manifest), valid_history_count=int(runtime.history.valid.sum()),
        direction_definition='- weighted original joint A gradient / global gradient norm',
        grad_total_norm_A=g_norm, grad_keep_norm_A=norm(before['component_gradients']['keep']),
        grad_learn_norm_A=norm(before['component_gradients']['learn']), direction_norm=norm(direction),
        ordinary_delta_A_norm=ordinary_norm, coefficients=dict(keep=runtime.sfra_config['keep_weight'], learn=runtime.sfra_config['learn_weight']),
        local_training_steps=0, optimizer_steps=0, history_commits=0, formal_test_evaluations=0,
        development_data=dict(available=False, evaluated=False,
            reason='Original Cifar100_LT provides train_x/data_test only; DataManager val_loader is disabled; no registered independent dev manifest found'),
        historical_correct_definition='Fixed 4227 valid positive-margin history samples; includes those wrong at ordinary A',
        classification_reduction='Original per-sample FP32 LA CE, same coefficients, diagnostic CPU FP64 accumulation',
        initial_numerics=runtime.initial_numerics, candidates=[])
    torch.save(dict(gradient_A=gradient, unit_direction=direction, a_keys=runtime.a_keys,
                    history=history_payload, fixed_targets=targets), out/'fixed_direction_and_history.pt')
    measurements, states = {}, {}
    for rho in RHOS:
        vector = (ordinary_a+rho*ordinary_norm*direction).float()
        candidate = dict(ordinary, **unflatten(vector, ordinary, runtime.a_keys))
        observed = before if rho == 0 else capture_feedback(runtime, candidate, targets, f'direction_rho_{rho:.2f}')
        row = summarize_candidate(runtime, before, observed, targets, ordinary, middle, candidate, rho)
        actual_delta = difference(candidate, ordinary, runtime.a_keys)
        row['rounding_error_norm'] = norm(actual_delta-rho*ordinary_norm*direction)
        row['direction_cosine_actual'] = float(actual_delta.dot(direction)/norm(actual_delta)) if rho else None
        result['candidates'].append(row)
        measurements[str(rho)] = {k:observed[k] for k in ('sample_margins','sample_ce','la_margins','correct','la_correct')}
        states[str(rho)] = {k:candidate[k].cpu().clone() for k in runtime.keys}
        if not all(torch.equal(targets[k], targets_copy[k]) for k in targets):
            raise ValueError('Fixed targets mutated')
        if not all(torch.equal(getattr(runtime.history, k), history_payload[k])
                   for k in ('H','valid','registered_round','block_min')) or runtime.history.last_committed_round != 79:
            raise ValueError('History mutated')
        write_json(out/'direction_diagnosis.json', result)
        print('DIRECTION_RESULT', json.dumps({k:row[k] for k in ('rho','keep_change','class_change','violated_count','repaired_deficits','new_deficits','delta_A_norm')}), flush=True)
    runtime.load_training_state(ordinary)
    result.update(status='offline_direction_diagnosis_complete', history_unchanged=True, targets_unchanged=True,
                  ordinary_restored=True, candidate_selection_performed=False)
    torch.save(dict(measurements=measurements, candidates_lora=states, manifest=runtime.margin_feedback.manifest), out/'candidate_observations.pt')
    write_json(out/'direction_diagnosis.json', result)
    print('DIRECTION_DIAGNOSIS_COMPLETE', str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prior, out = args.prior.resolve(), args.output.resolve()
    receipt = read_json(prior/'launch.json')
    source = Path(receipt['source_run'])
    if out == source or source in out.parents or out == prior or prior in out.parents:
        raise ValueError('Output must be separate from source and prior replay')
    job = read_json(source/'joint_job.json')
    from tools.client_aggregation.numerics import policy_for, prepare_environment, apply_policy
    prepare_environment(policy_for(job))
    apply_policy(policy_for(job))
    import torch
    from tools.client_aggregation.protocol import code_hashes
    from tools.client_aggregation.runtime import runtime_class
    from utils import cliplora_sfra
    current = code_hashes(REPO)
    changed = [k for k,v in job['code_sha256'].items() if current.get(k) != v]
    if set(changed)-{'utils/cliplora_method_a_margin.py','utils/method_a_margin_feedback.py'}:
        raise ValueError('Unexpected source changes: '+str(changed))
    torch.set_num_threads(1)
    work = out/'runtime'
    shutil.copytree(source/'protocol', work/'protocol')
    class DirectionRuntime(runtime_class(job)):
        def train_phase(self, *a, **kw):
            raise RuntimeError('Training disabled in offline direction diagnosis')
        def checkpoint(self, *a, **kw):
            raise RuntimeError('Training checkpoint disabled')
        def publish(self, *a, **kw):
            raise RuntimeError('Publication/test evaluation disabled')
        def correct_ordinary(self, *a, **kw):
            raise RuntimeError('Optimizer correction disabled: fixed offline candidates only')
        def run(self, initial_state):
            evaluate_direction(self, source, prior, out)
    command = worker_arguments(read_json(source/'command.json'))
    def replace(flag, value):
        command[command.index(flag)+1] = str(value)
    index = command.index('--joint-spec')
    del command[index:index+2]
    replace('--output-dir', work)
    replace('--client_schedule_file', work/'protocol/full_schedule.json')
    replace('--lac_partition_manifest', work/'protocol/partition_source.csv')
    replace('--sfra_resume', '')
    write_json(out/'direction_receipt.json', dict(command=command, code_sha256=current,
        source_run=str(source), prior_replay=str(prior), rhos=RHOS, training_enabled=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    original_runtime, argv = cliplora_sfra.SFRARuntime, sys.argv
    try:
        cliplora_sfra.SFRARuntime = DirectionRuntime
        sys.argv = [str(REPO/'federated_main.py'), *command]
        runpy.run_path(str(REPO/'federated_main.py'), run_name='__main__')
    finally:
        cliplora_sfra.SFRARuntime, sys.argv = original_runtime, argv


if __name__ == '__main__':
    main()
