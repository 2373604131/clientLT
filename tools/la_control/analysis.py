"""Attribution for both branches; restoration is a separate server intervention."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.cliplora_bridge_audit import write_csv, write_json


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def analyze(run, data_root, device='cuda', normal_rounds='all', segments=1):
    from tools.a_refresh_bridge.analysis import attribute_run
    run = Path(run)
    events = read(run/'event_manifest.csv')
    rounds = list(range(1,101)) if normal_rounds=='all' else sorted(set(map(int,normal_rounds.split(','))))
    # Always include every look-ahead normal event, in both branches.
    chosen = [r for r in events if r['phase']!='restore' and
              (r['phase'] not in ('normal_B','normal_AB') or int(r['round']) in rounds or r['branch'] in ('A','B'))]
    result = attribute_run(run,data_root,device,rounds,segments,[run/r['state_path'] for r in chosen])
    index = {r['event_id']:r for r in events}
    for name in ('phase_client_effects','phase_class_budgets','attribution_validity','phase_update_norms','first_order_budgets'):
        path = run/'analysis'/f'{name}.csv'
        rows = read(path)
        for row in rows:
            event = index[row['event_id']]
            row.update(committed=event['committed'],decision_round=event['decision_round'])
        write_csv(path,rows)
    restores = [r for r in events if r['phase']=='restore']
    restore_rows = []
    if restores:
        from federated_main import setup_cfg
        from trainers.cliplora import build_cliplora_model
        from tools.eri_closure.analysis import TrainOnlyFunctionalEvaluator
        from utils.cusp_minimal import make_flat_spec, flatten_state
        from utils.cliplora_a_refresh import state_hash
        meta = json.loads((run/'bridge_metadata.json').read_text(encoding='utf-8'))
        args = SimpleNamespace(**meta['resolved_args'])
        args.root, args.output_dir = str(data_root),str(run/'analysis/model_build')
        cfg = setup_cfg(args)
        model = build_cliplora_model(cfg,meta['classnames']).to(device).eval()
        keys = sorted(k for k in model.state_dict() if k.endswith(('_lora_A','_lora_B')))
        assert state_hash(model.state_dict(),sorted(set(model.state_dict())-set(keys)))==meta['frozen_model_sha256']
        for row in restores:
            payload = torch.load(run/row['state_path'],map_location='cpu',weights_only=False)
            spec = make_flat_spec(payload['before_lora_state'],keys)
            evaluator = TrainOnlyFunctionalEvaluator(cfg=cfg,trainer=SimpleNamespace(model=model),
                payload={'flatten_spec':spec.as_dict()},protocol_dir=run/'protocol',data_root=data_root,
                device=device,batch_size=10)
            before = flatten_state(payload['before_lora_state'],spec)
            after = flatten_state(payload['after_lora_state'],spec)
            for c in evaluator.class_ids:
                f0,f1 = evaluator.metric(before,c),evaluator.metric(after,c)
                restore_rows.append({'event_id':row['event_id'],'round':row['round'],'class_id':c,
                    'before_F':f0,'after_F':f1,'direct_change':f1-f0,
                    'rollback_applied':row['rollback_applied'],'attribution_type':'server_restore_not_client_W_H_D_R'})
        write_csv(run/'analysis/restore_functional_changes.csv',restore_rows)
    result.update(selected_training_events=len(chosen),restore_events=len(restores),
                  all_normal_rounds_attributed=normal_rounds=='all',
                  unselected_branches_are_counterfactual=True)
    write_json(run/'analysis/la_control_attribution_summary.json',result)
    print(f'Attribution complete: {run / "analysis"}',flush=True)
