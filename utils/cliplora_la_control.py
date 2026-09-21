"""E0--E5, J joint training, and S dense A refresh under ordinary FedAvg."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from utils.cliplora_a_refresh import (aggregate_refresh_deltas, append_rows, isolated_rng,
    lora_keys, state_hash, train_only)
from utils.cliplora_bridge_audit import BridgeAudit, write_csv, write_json
from utils.cliplora_functional_feedback import TrainingSideFeedback, observational_model, snapshot
from utils.lora_aggregation import aggregate_lora_state
from utils.pfrf import capture_rng_state, restore_rng_state


def load_global_checkpoint(path):
    """Decode the complete model state. This does not rewind a running controller."""
    path = Path(path)
    payload = torch.load(path,map_location='cpu',weights_only=False)
    state = torch.load(path.parents[1]/payload['base_model'],map_location='cpu',weights_only=False)
    state.update(payload['state_dict_overrides'])
    assert state_hash(state,sorted(state)) == payload['full_state_sha256']
    return state,payload


def rng_fingerprint(rng):
    digest = hashlib.sha256(repr(rng['python']).encode())
    digest.update(repr((rng['numpy'][0], *rng['numpy'][2:])).encode())
    digest.update(rng['numpy'][1].tobytes())
    digest.update(rng['torch'].cpu().numpy().tobytes())
    for state in rng['cuda'] or []:
        digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


class DecisionAudit(BridgeAudit):
    def event_directory(self, round_id, phase):
        return self.root / 'events' / self.event_context['event_id']


class LAControlRuntime:
    def __init__(self, trainer, global_trainer, cfg, args, schedule, normal_train):
        self.trainer, self.global_trainer = trainer, global_trainer
        self.cfg, self.args, self.schedule, self.normal_train = cfg, args, schedule, normal_train
        self.root = Path(args.output_dir)
        self.method = args.lac_method
        self.control = self.method in ('e4', 'e5')
        self.periodic = self.method in ('e1', 'e3', 'j', 's')
        self.normal_factor = 'AB' if self.method == 'j' else 'B'
        self.tau = args.lac_la_tau if self.method in ('e2', 'e3', 'e5', 'j', 's') else 0.
        self.h = args.lac_lookahead_rounds
        self.eps_hist = args.lac_tail_tolerance if args.lac_history_tolerance is None else args.lac_history_tolerance
        self.rounds = list(range(1, 91)) if self.method == 's' else list(range(10, 100, 10))
        assert 0 <= self.h < 10 and self.rounds[-1] + self.h <= 100
        assert args.round == 100 and args.local_epochs == 3 and args.a_refresh_epochs == 1
        assert args.a_refresh_interval == 10 and args.lac_patience >= 1
        assert args.lac_a_lr_mult > 0 and args.lac_la_tau >= 0
        assert args.lac_min_gain >= 0 and args.lac_tail_tolerance >= 0 and self.eps_hist >= 0
        assert args.cliplora_aggregation == 'fedavg' and args.frac == 1 and args.num_users == 30
        assert args.cliplora_precision == 'fp32' and cfg.TEST.NO_TEST
        assert not any((args.pfrf_enable, args.cliplora_sca_enable, args.stage3_enable, args.selective_sync_enable))
        assert args.cliplora_freeze_a and args.cliplora_v2 == 'off' and args.a_refresh_variant == 'off'
        assert args.cliplora_rank == 4 and args.cliplora_alpha == 1 and args.cliplora_dropout_rate == 0
        assert args.encoder == 'vision' and args.cliplora_position == 'top3' and args.cliplora_params == ['q','v']
        self.a_keys, self.b_keys = lora_keys(trainer.model, 'A'), lora_keys(trainer.model, 'B')
        self.keys = sorted(self.a_keys + self.b_keys)
        self.scaling = .5
        self.base_state = snapshot(trainer.model)
        self.initial = {k: self.base_state[k].clone() for k in self.keys}
        assert all(not p.requires_grad for n,p in trainer.model.named_parameters() if n not in self.keys)
        self.failure_count = self.accepted = self.attempts = 0
        self.frozen = False
        self.freeze_round = None
        self.best_valid_anchor = self.root_reference = None
        self.events, self.budget, self.evaluations, self.decisions = [], [], [], []
        self.started = time.perf_counter()
        # Metadata/probe setup must not consume the training random stream.
        with isolated_rng():
            audit_args = SimpleNamespace(**{**vars(args), 'a_refresh_variant': self.method})
            self.audit = DecisionAudit(trainer, cfg, audit_args, self, schedule)
            self.feedback = TrainingSideFeedback(trainer, cfg) if self.control else None
        self.sizes = self.audit.sizes.tolist()
        self.q = {k: n / sum(self.sizes) for k,n in enumerate(self.sizes)}
        counts = self.audit.counts.sum(0)
        from tools.eri_closure.protocol import load_protocol
        protocol, _ = load_protocol(self.root / 'protocol')
        self.tail = np.array(protocol['tail_class_ids'], dtype=int)
        assert len(self.tail) == 20
        assert set(self.tail)==set(sorted(range(100),key=lambda c:(int(counts[c]),-c))[:20])
        assert counts.sum().item() == 10847
        self.steps_per_epoch = sum((n+31)//32 for n in self.sizes)
        self.head = np.array(sorted(range(100), key=lambda c:(-int(counts[c]),c))[:20])
        self.middle = np.setdiff1d(np.arange(100), np.concatenate((self.head,self.tail)))
        self.prior = counts.double() / counts.sum()
        trainer.training_logit_adjustment = (self.tau * self.prior.log()).float().to(trainer.device) if self.tau else None
        replay = self.root / 'protocol/source_metadata.json'
        reference = replay if replay.exists() else self.root / 'protocol/reference_metadata.json'
        if reference.exists():
            source = json.loads(reference.read_text(encoding='utf-8'))
            for key in ('pool_sha256','test_sha256','probe_images_sha256','probe_manifest_sha256','schedule_sha256','frozen_model_sha256'):
                assert self.audit.meta[key] == source[key], key
            if replay.exists():
                assert args.partition == source['topology']
                assert self.sizes == source['client_sample_counts']
            if int(source['seed']) == args.seed:
                assert self.audit.meta['initial_lora_sha256'] == source['initial_lora_sha256']
        self.config = {'schema_version':'la_control_v1', 'method':self.method, 'seed':args.seed,
            'protocol_seed':args.split_seed,
            'topology':args.partition, 'dirichlet_beta':args.beta,
            'la_tau':self.tau, 'a_lr_mult':args.lac_a_lr_mult,
            'extra_a_lr':.001*args.lac_a_lr_mult, 'extra_b_lr':.001,
            'normal_trainable_factor':self.normal_factor,
            'normal_a_lr':.001*args.lac_a_lr_mult if self.normal_factor=='AB' else None,
            'normal_a_weight_decay':0. if self.normal_factor=='AB' else None,
            'normal_b_lr':float(cfg.OPTIM.LR), 'normal_b_weight_decay':float(cfg.OPTIM.WEIGHT_DECAY),
            'extra_weight_decay':0., 'aggregation':'sample_weighted_factor_fedavg',
            'control_enabled':self.control,
            'extra_trainable_factor':'controlled' if self.control else ('A' if self.periodic else 'B'),
            'lookahead_rounds':self.h, 'min_gain':args.lac_min_gain,
            'tail_tolerance':args.lac_tail_tolerance, 'history_tolerance':self.eps_hist,
            'patience':args.lac_patience, 'candidate_rounds':self.rounds, 'tail_ids':self.tail.tolist(),
            'head_ids':self.head.tolist(), 'middle_ids':self.middle.tolist(),
            'anchor_scope':'root_and_committed_decision_endpoints_only',
            'feedback':'training-side functional feedback: unscaled cosine margin, fixed views, all training samples',
            'offline_probe':'offline mechanism probe; never used for decisions',
            'rng_protocol':'v2: functional and official evaluations are observational; legacy CE results are historical only',
            'privacy':'simulated class sum/count uploads; class presence visible; no secure aggregation or DP',
            'primary_endpoint':'last20_committed_logical_rounds', 'checkpoint_encoding':'full base state plus exact changed tensors',
            'steps_per_epoch':self.steps_per_epoch,
            'normal_steps_expected':100*3*self.steps_per_epoch,
            'extra_steps_expected':self.steps_per_epoch*len(self.rounds)}
        write_json(self.root / 'control_config.json', self.config)
        write_json(self.root / 'class_prior.json', {'counts':counts.tolist(), 'prior':self.prior.tolist(), 'tail_ids':self.tail.tolist()})
        (self.root / 'resolved_config.yaml').write_text(str(cfg), encoding='utf-8')
        self.audit.meta['control_config'] = self.config
        for name in ('utils/cliplora_la_control.py','utils/cliplora_functional_feedback.py','datasets/cifar100_LT.py',
                     'Dassl/dassl/data/data_manager.py','Dassl/dassl/engine/trainer.py'):
            repo = Path(__file__).resolve().parents[1]
            self.audit.meta['training_code_hashes'][name] = hashlib.sha256((repo/name).read_bytes()).hexdigest()
        write_json(self.root / 'bridge_metadata.json', self.audit.meta)
        (self.root / 'checkpoints').mkdir(exist_ok=True)
        if not getattr(args, 'sfra_resume', ''):
            torch.save(self.base_state, self.root / 'checkpoints/base_model.pt')

    def compressed_state(self, state):
        # Include every changed parameter/buffer, not merely named A/B tensors.
        return {k:v.detach().cpu().clone() for k,v in state.items() if not torch.equal(v.cpu(), self.base_state[k])}

    def save_state(self, path, state, **metadata):
        torch.save({'base_model':'checkpoints/base_model.pt', 'state_dict_overrides':self.compressed_state(state),
                    'full_state_sha256':state_hash(state, sorted(state)), **metadata}, path)

    def observe(self, state, label):
        score, uploads, seconds = self.feedback.observe(state)
        self.evaluations.append({'label':label, 'kind':'training_side_feedback', 'seconds':seconds,
                                 'sample_presentations':sum(self.sizes), 'upload_bytes':30*100*16,
                                 'modeled_downlink_bytes':30*sum(state[k].numel()*state[k].element_size() for k in self.keys)})
        rows = [{'label':label, 'class_id':c, 'margin':float(score[c])} for c in range(100)]
        append_rows(self.root / 'feedback_class_scores.csv', rows)
        # These are client-level diagnostic uploads, not a privacy-preserving server interface.
        write_json(self.root / 'feedback_uploads' / f'{label}.json', uploads)
        return score

    def anchor(self, state, score, rnd, decision_id):
        result = {'model_state_dict':state, 'M':score.copy(), 'J':float(score.mean()),
                  'logical_round':rnd, 'decision_id':decision_id,
                  'ab_sha256':state_hash(state,self.keys), 'la_tau':self.tau,
                  'a_lr':self.config['extra_a_lr']}
        self.save_state(self.root / 'checkpoints' / f'{decision_id}.pt', state,
                        **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items() if k!='model_state_dict'})
        return result

    def bottom(self, values):
        ids = sorted(self.tail.tolist(), key=lambda c:(float(values[c]),c))[:5]
        return float(np.mean(values[ids])), ids

    def comparison(self, a, b, history):
        gap, hist = a-b, a-history
        tail, tail_ids = self.bottom(gap)
        historic, hist_ids = self.bottom(hist)
        return {'G_all':float(gap.mean()), 'G_tail':tail, 'G_hist':historic,
                'bottom5_tail_ids':tail_ids, 'bottom5_history_ids':hist_ids,
                'g_mean':float(gap.mean()), 'g_std':float(gap.std()),
                'tail_g_mean':float(gap[self.tail].mean()), 'tail_g_std':float(gap[self.tail].std()),
                'bottom5_g_std':float(gap[tail_ids].std())}

    def prepare_b_aggregation(self, state, local_states, deltas, selected, rnd):
        """Optional recipient-side work after ALL ordinary local updates are cached."""
        return local_states, deltas

    def train_phase(self, state, rnd, factor='B', extra=False, branch='main', candidate=0,
                    return_deltas=False):
        phase = ('refresh_A' if factor=='A' else 'extra_B') if extra else f'normal_{factor}'
        event_id = f'r{rnd:03d}_c{candidate:03d}_{branch}_{phase}'
        selected = list(map(int,self.schedule[rnd-1]))
        assert sorted(selected) == list(range(30))
        local_states, deltas = {}, {}
        keys = self.keys if factor=='AB' else (self.a_keys if factor=='A' else self.b_keys)
        phase_started = time.perf_counter()
        for client in selected:
            self.trainer.model.load_state_dict(state, strict=True)
            train_only(self.trainer.model, factor)
            if extra:
                from trainers.cliplora import cliplora_optimizer_step
                lr = self.config['extra_a_lr'] if factor=='A' else self.config['extra_b_lr']
                optimizer = torch.optim.SGD([p for p in self.trainer.model.parameters() if p.requires_grad],
                                            lr=lr,momentum=.9,weight_decay=0.)
                steps = samples = 0
                with isolated_rng(self.args.seed+1000003*rnd+1009*client):
                    self.trainer.model.train()
                    for batch in self.trainer.fed_train_loader_x_dict[client]:
                        images, labels = self.trainer.parse_batch_train(batch)
                        cliplora_optimizer_step(self.trainer.model, optimizer, None, 'fp32', images, labels,
                                                logit_adjustment=self.trainer.training_logit_adjustment)
                        steps += 1
                        samples += labels.numel()
                scheduler_steps = 0
            else:
                if factor == 'AB':
                    parameters = dict(self.trainer.model.named_parameters())
                    self.trainer.reset_optimizer_and_scheduler(param_groups=[
                        {'params':[parameters[k] for k in self.a_keys],
                         'lr':self.config['normal_a_lr'], 'weight_decay':0.},
                        {'params':[parameters[k] for k in self.b_keys],
                         'lr':self.config['normal_b_lr'],
                         'weight_decay':self.config['normal_b_weight_decay']},
                    ])
                else:
                    self.trainer.reset_optimizer_and_scheduler()
                # Temporary branches have no official test or committed TensorBoard writes.
                writer = self.trainer._writer
                self.trainer._writer = None
                try:
                    _,_,scheduler_steps,steps = self.normal_train(self.trainer,client,rnd-1,self.args,3)
                finally:
                    self.trainer._writer = writer
                samples = self.sizes[client]*3
            local_states[client] = {k:self.trainer.model.state_dict()[k].detach().cpu().clone() for k in keys}
            deltas[client] = {k:local_states[client][k]-state[k] for k in keys}
            self.budget.append({'event_id':event_id,'round':rnd,'candidate_round':candidate,'branch':branch,
                'phase':phase,'client_id':client,'optimizer_steps':steps,'scheduler_steps':scheduler_steps,
                'sample_presentations':samples,'committed':branch=='main','trainable_factor':factor,
                'a_optimizer_steps':steps if factor in ('A','AB') else 0,
                'b_optimizer_steps':steps if factor in ('B','AB') else 0,
                'upload_bytes':sum(v.numel()*v.element_size() for v in local_states[client].values()),
                'modeled_downlink_bytes':sum(state[k].numel()*state[k].element_size() for k in self.keys)})
        if factor == 'B' and not extra and branch == 'main':
            local_states, deltas = self.prepare_b_aggregation(state, local_states, deltas, selected, rnd)
        if extra:
            after = aggregate_refresh_deltas(state,deltas,self.q,keys)
        else:
            after = aggregate_lora_state(state,local_states,selected,keys,self.q)
        context = {'event_id':event_id,'branch':branch,'candidate_round':candidate}
        self.audit.event_context = context
        if extra:
            self.audit.save(state,after,deltas,selected,self.q,rnd,phase)
        else:
            self.audit.normal(state,after,local_states,selected,self.q,rnd,factor=factor)
        info = json.loads((self.root / 'events' / event_id / 'event.json').read_text(encoding='utf-8'))
        self.events.append({**info,'committed':branch=='main','decision_round':rnd if branch=='main' else None,
                            'seconds':time.perf_counter()-phase_started,'state_path':f'events/{event_id}/state.pt'})
        self.trainer.model.load_state_dict(after,strict=True)
        train_only(self.trainer.model,'B')
        print(f'LA-control phase complete: {event_id}',flush=True)
        return (after, deltas) if return_deltas else after

    def branch(self, start, rnd, factor, rng):
        restore_rng_state(rng)
        state = self.train_phase(start,rnd,factor,True,factor,rnd)
        states = [(rnd,state)]
        instant = self.observe(state,f'c{rnd:03d}_{factor}_instant')
        for future in range(rnd+1,rnd+self.h+1):
            state = self.train_phase(state,future,'B',False,factor,rnd)
            states.append((future,state))
        delayed = self.observe(state,f'c{rnd:03d}_{factor}_delayed') if self.h else instant
        return {'state':state,'states':states,'instant':instant,'M':delayed,'rng':capture_rng_state()}

    def decision(self, start, rnd):
        if self.root_reference is None:
            m = self.observe(start,'root_reference')
            self.root_reference = self.anchor(start,m,rnd,'root_reference')
            self.best_valid_anchor = self.root_reference
        old = self.best_valid_anchor
        rng = capture_rng_state()
        b = self.branch(start,rnd,'B',rng)
        a = self.branch(start,rnd,'A',rng)
        self.attempts += 1
        instant = self.comparison(a['instant'],b['instant'],old['M'])
        delayed = self.comparison(a['M'],b['M'],old['M'])
        passed = [delayed['G_all']>self.args.lac_min_gain,
                  delayed['G_tail']>=-self.args.lac_tail_tolerance,
                  delayed['G_hist']>=-self.eps_hist]
        accept = all(passed)
        chosen, branch = (a,'A') if accept else (b,'B')
        # Rejection counts A's failure, independent of B improving the anchor.
        self.failure_count = 0 if accept else self.failure_count+1
        self.accepted += int(accept)
        restore_rng_state(chosen['rng'])
        end = rnd+self.h
        decision_id = f'decision_{end:03d}'
        eligibility,_ = self.bottom(chosen['M']-old['M'])
        valid = eligibility >= -self.eps_hist
        improved = valid and float(chosen['M'].mean()) > old['J']
        selected_anchor = self.anchor(chosen['state'],chosen['M'],end,decision_id)
        if improved:
            self.best_valid_anchor = selected_anchor
        result = chosen['state']
        frozen_now = self.failure_count >= self.args.lac_patience
        rollback = False
        distance = 0
        if frozen_now:
            self.frozen, self.freeze_round = True,end
            target = self.best_valid_anchor
            rollback = any(not torch.equal(result[k],target['model_state_dict'][k]) for k in result)
            distance = end-target['logical_round'] if rollback else 0
            restore_id = f'restore_{end:03d}'
            folder = self.root / 'events' / restore_id
            folder.mkdir(parents=True,exist_ok=True)
            torch.save({'before_lora_state':{k:result[k] for k in self.keys},
                        'after_lora_state':{k:target['model_state_dict'][k] for k in self.keys}},folder/'state.pt')
            restore_row = {'event_id':restore_id,'round':end,'phase':'restore','branch':'server',
                'candidate_round':rnd,'committed':True,'decision_round':end,'rollback_applied':rollback,
                'rollback_distance_rounds':distance,'restore_from_round':target['logical_round'],
                'before_lora_sha256':state_hash(result,self.keys),'after_lora_sha256':target['ab_sha256'],
                'state_path':f'events/{restore_id}/state.pt','optimizer_steps':0}
            self.events.append(restore_row)
            write_json(folder/'event.json',restore_row)
            append_rows(self.root/'restore_events.csv',[restore_row])
            result = target['model_state_dict']
        for event in self.events:
            if event['candidate_round']==rnd and event['branch'] in ('A','B'):
                event['committed'] = event['branch']==branch
                event['decision_round'] = end
        for row in self.budget:
            if row['candidate_round']==rnd:
                row['committed'] = row['branch']==branch
        self.trainer.model.load_state_dict(result,strict=True)
        train_only(self.trainer.model,'B')
        row = {'candidate_round':rnd,'decision_round':end,'selected_branch':branch,'accepted_A':accept,
            'failure_count':self.failure_count,'accepted_A_count':self.accepted,
            'pass_gain':passed[0],'pass_tail':passed[1],'pass_history':passed[2],
            'anchor_before':old['decision_id'],'anchor_after':self.best_valid_anchor['decision_id'],
            'selected_history_eligible':valid,'anchor_updated':improved,'selected_J':float(chosen['M'].mean()),
            'freeze_A':frozen_now,'rollback_applied':rollback,'rollback_distance_rounds':distance,
            'a_lr':self.config['extra_a_lr'],'min_gain':self.args.lac_min_gain,
            'tail_tolerance':self.args.lac_tail_tolerance,'history_tolerance':self.eps_hist,
            **{f'instant_{k}':v for k,v in instant.items()},**{f'delayed_{k}':v for k,v in delayed.items()},
            'P_all':delayed['G_all']-instant['G_all'],'P_tail':delayed['G_tail']-instant['G_tail'],
            'P_hist':delayed['G_hist']-instant['G_hist'],
            'common_rng_start':rng_fingerprint(rng),'A_rng_end':rng_fingerprint(a['rng']),
            'B_rng_end':rng_fingerprint(b['rng'])}
        for branch_name in ('A','B'):
            event = next(e for e in self.events if e['candidate_round']==rnd and e['branch']==branch_name and e['phase'] in ('extra_B','refresh_A'))
            row[f'{branch_name}_effective_update_norm'] = event['effective_delta_norm']
        self.decisions.append(row)
        append_rows(self.root/'control_decisions.csv',[row])
        append_rows(self.root/'control_class_metrics.csv',[{'candidate_round':rnd,'decision_round':end,'class_id':c,
            'instant_g':float(a['instant'][c]-b['instant'][c]),'delayed_g':float(a['M'][c]-b['M'][c]),
            'history_difference':float(a['M'][c]-old['M'][c]),
            'selected_vs_root':float(chosen['M'][c]-self.root_reference['M'][c]),
            'selected_vs_history':float(chosen['M'][c]-old['M'][c]),
            'bottom5_tail':c in delayed['bottom5_tail_ids'],'bottom5_history':c in delayed['bottom5_history_ids']}
            for c in range(100)])
        append_rows(self.root/'history_best.csv',[{'decision_round':end,'decision_id':decision_id,
            'selected_J':selected_anchor['J'],'eligible':valid,'updated':improved,
            'best_valid_anchor':self.best_valid_anchor['decision_id'],'best_J':self.best_valid_anchor['J']}])
        committed_states = chosen['states'][:-1]+[(end,result)]
        print(f'DECISION {rnd}->{end}: choose={branch} G_all={delayed["G_all"]:.6g} '
              f'G_tail={delayed["G_tail"]:.6g} G_hist={delayed["G_hist"]:.6g} '
              f'failures={self.failure_count} freeze={frozen_now} rollback={rollback}',flush=True)
        return result,committed_states

    def publish(self, state, rnd, decision_round):
        started = time.perf_counter()
        with observational_model(self.global_trainer.model):
            self.global_trainer.model.load_state_dict(state,strict=True)
            result = self.global_trainer.global_test(is_global=True,current_epoch=rnd-1)
        per_class = np.array([float(result[3][c]) for c in range(100)])
        non_tail = np.setdiff1d(np.arange(100),self.tail)
        append_rows(self.root/'round_metrics.csv',[{'epoch':rnd-1,'round':rnd,'decision_round':decision_round,
            'method':self.method,'partition':self.args.partition,'seed':self.args.seed,
            'overall_acc':float(result[0]),'non_tail_acc':float(per_class[non_tail].mean()),
            'head20_acc':float(per_class[self.head].mean()),'middle60_acc':float(per_class[self.middle].mean()),
            'bottom20_tail_acc':float(per_class[self.tail].mean()),'macro_per_class_acc':float(per_class.mean())}])
        write_csv(self.root/f'per_class_accuracy_epoch_{rnd-1}.csv',
                  [{'class_id':c,'per_class_acc':float(per_class[c])} for c in range(100)])
        self.evaluations.append({'label':f'official_round_{rnd}','kind':'official_test',
            'seconds':time.perf_counter()-started,'sample_presentations':len(self.global_trainer.test_loader.dataset),
            'upload_bytes':0,'modeled_downlink_bytes':0})

    def flush(self, completed, state):
        write_csv(self.root/'event_manifest.csv',self.events)
        write_csv(self.root/'budget.csv',self.budget)
        write_csv(self.root/'evaluation_budget.csv',self.evaluations)
        normal = sum(r['optimizer_steps'] for r in self.budget if r['committed'] and r['phase'] in ('normal_B','normal_AB'))
        extra = sum(r['optimizer_steps'] for r in self.budget if r['committed'] and r['phase'] not in ('normal_B','normal_AB'))
        overhead = sum(r['optimizer_steps'] for r in self.budget if not r['committed'])
        progress = {'completed_round':completed,'normal_optimizer_steps':normal,'extra_optimizer_steps':extra,
            'unselected_branch_optimizer_steps':overhead,'total_optimizer_steps':normal+extra+overhead,
            'committed_a_optimizer_steps':sum(r['a_optimizer_steps'] for r in self.budget if r['committed']),
            'committed_b_optimizer_steps':sum(r['b_optimizer_steps'] for r in self.budget if r['committed']),
            'a_refresh_events':sum(r['committed'] and r['phase']=='refresh_A' for r in self.events),
            'normal_ab_events':sum(r['committed'] and r['phase']=='normal_AB' for r in self.events),
            'attempted_decisions':self.attempts,'accepted_A_decisions':self.accepted,
            'failure_count':self.failure_count,'freeze_A_forever':self.frozen,'freeze_round':self.freeze_round,
            'feedback_passes':sum(r['kind']=='training_side_feedback' for r in self.evaluations),
            'official_test_passes':sum(r['kind']=='official_test' for r in self.evaluations),
            'best_valid_anchor':self.best_valid_anchor['decision_id'] if self.best_valid_anchor else None,
            'elapsed_seconds':time.perf_counter()-self.started}
        write_json(self.root/'progress.json',progress)
        self.save_state(self.root/'checkpoints/last.pt',state,completed_round=completed,
                        rng_state=capture_rng_state(),config=self.config,progress=progress)
        if completed==100:
            assert normal==self.config['normal_steps_expected'] and extra==self.config['extra_steps_expected']
            assert overhead==self.attempts*self.steps_per_epoch*(1+3*self.h)
            assert progress['official_test_passes']==101
            if self.control:
                assert progress['feedback_passes']==1+self.attempts*(4 if self.h else 2)
            write_json(self.root/'completion.json',progress)

    def run(self, initial_state):
        # Use a full CPU state throughout; anchors include all buffers as well as A/B.
        self.trainer.model.load_state_dict(initial_state,strict=True)
        state = snapshot(self.trainer.model)
        self.publish(state,0,0)
        rnd = 1
        while rnd <= 100:
            state = self.train_phase(state,rnd,self.normal_factor)
            if rnd in self.rounds and self.control and not self.frozen:
                state,committed = self.decision(state,rnd)
                for logical,checkpoint in committed:
                    self.publish(checkpoint,logical,rnd+self.h)
                rnd += self.h
                # Release temporary full snapshots before the next decision window.
                del committed
            else:
                if rnd in self.rounds:
                    state = self.train_phase(state,rnd,'A' if self.periodic else 'B',True)
                self.publish(state,rnd,rnd)
            self.flush(rnd,state)
            rnd += 1
