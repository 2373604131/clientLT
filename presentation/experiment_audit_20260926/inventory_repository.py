"""Read-only repository census and result extraction; writes only this audit folder."""
from __future__ import annotations
import csv, hashlib, json, math, os, re, statistics, tarfile, zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SKIP = {'.git', '__pycache__', '.ipynb_checkpoints', '.pytest_cache'}
ERRORS = []

def rel(p):
    return Path(p).relative_to(ROOT).as_posix()

def read_json(p):
    try:
        return json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception as e:
        ERRORS.append({'path': rel(p), 'stage': 'json', 'error': str(e)})
        return {}

def read_csv(p):
    try:
        with p.open(encoding='utf-8-sig', errors='replace', newline='') as f:
            return list(csv.DictReader(f))
    except Exception as e:
        ERRORS.append({'path': rel(p), 'stage': 'csv', 'error': str(e)})
        return []

def num(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None

def avg(vals):
    vals=[v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_csv(name, rows, columns=None):
    columns=columns or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def family(p):
    parts=Path(p).relative_to(ROOT/'output').parts
    if parts[0] in ('cifar100_LT', 'cifar10_LT') and len(parts)>1:
        return '/'.join(parts[:2])
    return parts[0]

def config_scalars(p):
    result={}
    for f in sorted(p.glob('*config.json')):
        obj=read_json(f)
        if isinstance(obj,dict):
            for k,v in obj.items():
                if not isinstance(v,(list,dict)):
                    result[f.stem+'.'+k]=v
    return result

def summarize_run(p, files, known):
    metrics_path=p/'round_metrics.csv'
    rows=read_csv(metrics_path) if metrics_path.exists() else []
    raw_rows=len(rows)
    by_round={}; conflicts=[]
    for r in rows:
        t=num(r.get('round'))
        if t is None:
            e=num(r.get('epoch'))
            t=e+1 if e is not None else None
        if t is None:
            continue
        t=int(t)
        if t in by_round and by_round[t] != r:
            conflicts.append(t)
        by_round[t]=r
    completion=read_json(p/'completion.json') if (p/'completion.json').exists() else {}
    partition=read_json(p/'partition_summary.json') if (p/'partition_summary.json').exists() else {}
    if not isinstance(partition,dict): partition={}
    if not isinstance(completion,dict): completion={}
    classes=partition.get('global_class_counts',[])
    tail_ids=partition.get('tail_classes',[])
    if not isinstance(tail_ids,list): tail_ids=[]
    tail_ids={int(x) for x in tail_ids}
    class_files=[]
    for n in files:
        m=re.fullmatch(r'per_class_accuracy_epoch_(-?\d+)\.csv',n)
        if m:
            class_files.append((int(m[1])+1,p/n))
    class_curves={}; worst=0.; comparable=0; mismatches=[]; incomplete_class_files=[]
    # All per-class records are read, including sparse and incomplete trajectories.
    for t,f in sorted(class_files):
        rr=read_csv(f)
        a={int(r['class_id']):num(r.get('per_class_acc')) for r in rr if num(r.get('class_id')) is not None}
        a={k:v for k,v in a.items() if v is not None}
        if not a: continue
        if classes and set(a)!=set(range(len(classes))):
            # Never invent values for absent classes, or average a truncated export.
            incomplete_class_files.append({'round':t,'present':len(a),'expected':len(classes)})
            continue
        if not classes and not tail_ids:
            # Infer no frequency groups from class ID alone.
            vals={'overall':avg(a.values()),'tail':None,'nontail':None,'head':None,'middle':None}
        else:
            order=sorted(range(len(classes)), key=lambda k:(-classes[k],k)) if classes else []
            h=max(1,math.ceil(len(order)*.2)) if order else 0
            if not tail_ids and order:
                tail_ids=set(order[-h:])
            head_ids=set(order[:h]); middle_ids=set(a)-head_ids-tail_ids
            vals={'overall':avg(a.values()),'tail':avg(a[k] for k in tail_ids if k in a),
                  'nontail':avg(v for k,v in a.items() if k not in tail_ids),
                  'head':avg(a[k] for k in head_ids if k in a),
                  'middle':avg(a[k] for k in middle_ids if k in a)}
            if classes:
                vals.update(many=avg(v for k,v in a.items() if classes[k]>100),
                            medium=avg(v for k,v in a.items() if 20<=classes[k]<=100),
                            few=avg(v for k,v in a.items() if classes[k]<20))
        class_curves[t]=vals
        r=by_round.get(t)
        if r:
            for key,col in [('overall','overall_acc'),('tail','bottom20_tail_acc'),('nontail','non_tail_acc')]:
                recorded=num(r.get(col)); recomputed=vals[key]
                if recorded is not None and recomputed is not None:
                    comparable+=1; d=abs(recorded-recomputed);worst=max(worst,d)
                    if d>1e-6: mismatches.append((t,key,d))
    if not by_round and class_curves:
        for t,v in class_curves.items():
            by_round[t]={'overall_acc':v['overall'],'bottom20_tail_acc':v['tail'],'non_tail_acc':v['nontail']}
    times=sorted(by_round)
    last=times[-1] if times else None
    final=by_round[last] if last is not None else {}
    first=by_round[times[0]] if times else {}
    metrics={t:{'overall':num(r.get('overall_acc')),'tail':num(r.get('bottom20_tail_acc')),
                'nontail':num(r.get('non_tail_acc')),
                'head':num(r.get('head20_acc')),'middle':num(r.get('middle60_acc')),
                'many':None,'medium':None,'few':None} for t,r in by_round.items()}
    for t,v in class_curves.items():
        if t in metrics:
            for k in ('head','middle','many','medium','few'):
                if metrics[t][k] is None: metrics[t][k]=v.get(k)
    window=list(range(last-19,last+1)) if last is not None and last>=20 else []
    window_ok=bool(window) and all(t in metrics for t in window)
    late={k:avg(metrics[t][k] for t in window) if window_ok and all(metrics[t][k] is not None for t in window) else None for k in ('overall','tail','nontail','head','middle','many','medium','few')}
    tail_points=[(v['tail'],t) for t,v in metrics.items() if t>=1 and v['tail'] is not None]
    peak=max(tail_points, key=lambda vt:(vt[0],-vt[1])) if tail_points else (None,None)
    final_tail=metrics[last]['tail'] if last is not None else None
    if completion.get('completed_round') and last is not None and last>=completion['completed_round']:
        status='完成标记与曲线一致'
    elif last is not None and last>=100:
        status='已记录到第100轮或以后'
    elif last is not None:
        status='短运行或本地结果不完整'
    else:
        status='只有日志或配置，未发现精度结果'
    cfg=config_scalars(p)
    log=p/'log.txt'; log_text=''
    if log.exists():
        with log.open(encoding='utf-8',errors='replace') as f: log_text=f.read(70000)
    effective=re.search(r'ClipLora effective config:([^\r\n]+)',log_text)
    rank_match=re.search(r'\brank=(\d+)',effective.group(1)) if effective else None
    precision_match=re.search(r'\bprecision=(\w+)',effective.group(1)) if effective else None
    trainer_match=re.search(r'Loading trainer:\s*([^\r\n]+)',log_text)
    ds_match=re.search(r'Loading dataset:\s*([^\r\n]+)',log_text)
    metadata=known.get(rel(metrics_path),{})
    run_id='R'+hashlib.sha256(rel(p).encode()).hexdigest()[:9]
    r={
      'run_id':run_id,'family':family(p),'run_path':rel(p),
      'description':metadata.get('实验具体设置',''),
      'method_recorded':first.get('method',completion.get('variant','')),
      'trainer':trainer_match.group(1).strip() if trainer_match else partition.get('method',''),
      'dataset':ds_match.group(1).strip() if ds_match else ('CIFAR100-LT' if len(classes)==100 else ''),
      'partition':first.get('partition',partition.get('partition','')),
      'seed':first.get('seed',partition.get('seed','')),
      'rank':int(rank_match.group(1)) if rank_match else '',
      'precision':precision_match.group(1) if precision_match else '',
      'known_protocol':metadata.get('rank/精度',''),
      'normal_training':metadata.get('正常训练阶段',''),
      'extra_training':metadata.get('额外更新阶段',''),
      'loss':metadata.get('损失/类别校正',''),
      'aggregation':metadata.get('服务器聚合/接收方式',first.get('aggregation','')),
      'status':status,'first_round':times[0] if times else None,'last_round':last,
      'observed_rounds':len(times),'raw_metric_rows':raw_rows,
      'duplicate_round_conflicts':','.join(map(str,conflicts)),
      'window_start':window[0] if window_ok else None,'window_end':last if window_ok else None,
      'last20_overall':late['overall'],'last20_head20':late['head'],'last20_middle60':late['middle'],
      'last20_tail':late['tail'],'last20_nontail':late['nontail'],
      'last20_many_gt100':late['many'],'last20_medium20to100':late['medium'],'last20_few_lt20':late['few'],
      'final_overall':num(final.get('overall_acc')),'final_tail':final_tail,
      'tail_peak':peak[0],'tail_peak_round':peak[1],
      'tail_peak_to_last':peak[0]-final_tail if peak[0] is not None and final_tail is not None else None,
      'initial_tail':metrics.get(0,{}).get('tail'),
      'num_classes':len(classes) or '', 'num_clients':partition.get('num_clients',''),
      'global_count_hash':hashlib.sha256(json.dumps(classes).encode()).hexdigest() if classes else '',
      'client_count_hash':hashlib.sha256(json.dumps(partition.get('client_sample_counts')).encode()).hexdigest() if partition.get('client_sample_counts') else '',
      'global_lt_fingerprint':partition.get('global_lt_fingerprint',''),
      'tail_carriers':partition.get('tail_num_support_clients_mean',''),
      'client_min':partition.get('client_sample_min',''),'client_max':partition.get('client_sample_max',''),
      'manifest_sha256':digest(p/'partition_manifest.csv') if (p/'partition_manifest.csv').exists() else '',
      'per_class_files':len(class_files),'metric_comparisons':comparable,
      'incomplete_per_class_files':len(incomplete_class_files),
      'incomplete_per_class_examples':incomplete_class_files[:5],
      'max_recomputation_error':worst if comparable else None,
      'recomputation_status':'完整逐类文件仍不一致需核对' if mismatches else ('部分逐类文件缺类，不能完整复算' if incomplete_class_files else ('逐类核对通过' if comparable else '缺少可交叉核对的分组或逐类文件')),
      'normal_optimizer_steps':completion.get('normal_optimizer_steps',''),
      'extra_optimizer_steps':completion.get('extra_optimizer_steps',''),
      'total_local_optimizer_steps':completion.get('total_local_optimizer_steps',''),
      'functional_correction_steps':completion.get('functional_correction_steps',''),
      'elapsed_seconds':completion.get('elapsed_seconds',''),
      'notes':metadata.get('说明',''),
      'curve_source':rel(metrics_path) if metrics_path.exists() else 'per_class CSV',
      'completion_source':rel(p/'completion.json') if completion else '',
      'config_scalars':cfg,
      'metrics_sha256':digest(metrics_path) if metrics_path.exists() else '',
    }
    numeric_signature=[[t,metrics[t]['overall'],metrics[t]['tail'],metrics[t]['nontail']] for t in times]
    r['curve_fingerprint']=hashlib.sha256(json.dumps(numeric_signature).encode()).hexdigest() if len(times)>=5 else ''
    if mismatches:
        ERRORS.append({'path':rel(p),'stage':'per_class_validation','error':mismatches[:12],'count':len(mismatches)})
    return r,[{'run_id':run_id,'round':t,**v} for t,v in sorted(metrics.items())]

def main():
    OUT.mkdir(exist_ok=True)
    files=[]; by_dir=defaultdict(list); family_stats=defaultdict(Counter)
    for base in ('output','exp1_longtail_per_client','experiments','tmp'):
        for current, dirs, names in os.walk(ROOT/base):
            dirs[:]=sorted(d for d in dirs if d not in SKIP)
            for name in sorted(names):
                p=Path(current)/name
                stat=p.stat()
                f=family(p) if base=='output' else base
                files.append({'path':rel(p),'family':f,'extension':p.suffix.lower(),'bytes':stat.st_size})
                by_dir[p.parent].append(name)
                family_stats[f]['files']+=1;family_stats[f]['bytes']+=stat.st_size
                family_stats[f][p.suffix.lower()]+=1
    known_path=ROOT/'presentation/lora_ab_inventory_20260919/全部实验大表.csv'
    known={r['结果来源'].replace('\\','/'):r for r in read_csv(known_path)}
    runs=[]; curves=[]
    for p,names in sorted(by_dir.items()):
        if not p.is_relative_to(ROOT/'output'): continue
        has_class=any(re.fullmatch(r'per_class_accuracy_epoch_-?\d+\.csv',n) for n in names)
        if 'round_metrics.csv' in names or has_class or 'completion.json' in names or 'log.txt' in names:
            r,c=summarize_run(p,names,known)
            runs.append(r);curves.extend(c)
    # Identical exported curves are reported as duplicate candidates, not independent seeds.
    dup=defaultdict(list)
    for r in runs:
        if r['curve_fingerprint']:
            dup[(r['curve_fingerprint'],str(r['seed']),r['partition'])].append(r)
    aliases=[]
    for key,group in dup.items():
        if len(group)<2:continue
        chosen=sorted(group,key=lambda r:(-r['per_class_files'],-r['metric_comparisons'],len(r['run_path']),r['run_path']))[0]
        group_id='D'+key[0][:8]
        for r in group:
            r['duplicate_curve_group']=group_id
            r['preferred_result_path']=chosen['run_path']
            r['curve_copy_role']='推荐核对入口' if r is chosen else '曲线相同的副本候选'
            aliases.append({'group':group_id,'run_id':r['run_id'],'path':r['run_path'],
                            'preferred':chosen['run_path'],'metrics_file_byte_identical':r['metrics_sha256']==chosen['metrics_sha256'],
                            'manifest_identical_when_present':bool(r['manifest_sha256'] and chosen['manifest_sha256'] and r['manifest_sha256']==chosen['manifest_sha256']),
                            'scope':'轨迹/seed/partition相同；不能当额外重复种子，仍保留全部目录'})
    for r in runs:
        r.setdefault('duplicate_curve_group','');r.setdefault('preferred_result_path',r['run_path']);r.setdefault('curve_copy_role','独立目录记录')
    csv_index=[]; reports=[]; archives=[]
    for f in files:
        p=ROOT/f['path'];name=p.name
        if p.suffix.lower()=='.md': reports.append({'family':f['family'],'path':f['path'],'bytes':f['bytes']})
        if p.suffix.lower()=='.csv' and not (re.match(r'per_class_accuracy_epoch_',name) or name in {'client_class_counts.csv','class_topology.csv','partition_manifest.csv'} or any(x in p.parts for x in ('sfra_rounds','b_transfer_rounds','private_diagnostics','events'))):
            try:
                with p.open(encoding='utf-8-sig',errors='replace',newline='') as h:
                    reader=csv.reader(h);header=next(reader,[]); n=sum(1 for _ in reader)
                csv_index.append({'family':f['family'],'path':f['path'],'rows':n,'columns':'; '.join(header),'bytes':f['bytes']})
            except Exception as e:ERRORS.append({'path':f['path'],'stage':'index_csv','error':str(e)})
        if name.endswith(('.tar.gz','.tgz','.zip','.tar')):
            try:
                if name.endswith('.zip'):
                    with zipfile.ZipFile(p) as z:members=z.namelist()
                else:
                    with tarfile.open(p,'r:*') as t:members=[m.name for m in t]
                result_members=[x for x in members if x.endswith(('round_metrics.csv','completion.json','log.txt'))]
                archives.append({'path':f['path'],'bytes':f['bytes'],'members':len(members),'result_members':result_members,'status':'只索引未解包，结果不计为新增运行'})
            except Exception as e:archives.append({'path':f['path'],'bytes':f['bytes'],'status':'归档读取失败','error':str(e)})
    families=[]
    for f,s in sorted(family_stats.items()):
        rr=[r for r in runs if r['family']==f]
        families.append({'family':f,'files':s['files'],'size_mb':s['bytes']/1024**2,
                         'run_directories':len(rr),'complete100_directories':sum((r['last_round'] or 0)>=100 for r in rr),
                         'csv_files':s['.csv'],'json_files':s['.json'],'reports':'; '.join(x['path'] for x in reports if x['family']==f)})
    write_csv('全部文件索引.csv',files)
    write_csv('训练运行总账.csv',runs)
    write_csv('全部标准化训练曲线.csv',curves)
    write_csv('相同曲线与副本候选.csv',aliases)
    write_csv('分析表索引.csv',csv_index)
    write_csv('实验族清单.csv',families)
    write_csv('报告索引.csv',reports)
    write_csv('归档包索引.csv',archives)
    payload={'runs':runs,'families':families,'csv_index':csv_index,'reports':reports,'archives':archives,'errors':ERRORS}
    (OUT/'inventory.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'scanned_files':len(files),'families':len(families),'run_directories':len(runs),
             'complete100_directories':sum((r['last_round'] or 0)>=100 for r in runs),
             'curve_records':len(curves),'duplicate_curve_groups':len(set(a['group'] for a in aliases)),
             'duplicate_member_directories':len(aliases),'recomputed_per_class_files':sum(r['per_class_files'] for r in runs),
             'metric_comparisons':sum(r['metric_comparisons'] for r in runs),'error_count':len(ERRORS),
             'analysis_csv_tables':len(csv_index),'reports':len(reports),'archives':len(archives)}
    (OUT/'scan_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
