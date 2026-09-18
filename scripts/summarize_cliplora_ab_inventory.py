"""Read-only recomputation of completed AB-framework results and split status."""

import csv
import importlib.util
import json
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'output/ab_framework_partition_review_20260918'
LATEST = REPO / 'output/la_control_js_topology_analysis/la_control/seed42'
DESIGNS = {
    'V1_L1': '固定A，私有B，Top1接收保护',
    'V1_L3': '固定A，私有B，Top3接收保护',
    'V1_L5': '固定A，私有B，Top5接收保护',
    'V1_L10': '固定A，私有B，Top10接收保护',
    'V2_fedavg': '固定A，B-only，样本量FedAvg',
    'V2_static': '固定A，B静态客户端调权',
    'V2_progressive': '固定A，B渐进客户端调权',
    'R4_C0': '固定A，B-only；C0基线',
    'R8_C0': '固定A，B-only；rank8',
    'Pilot_c1': 'CE，固定A，九次额外B',
    'Pilot_c2': 'CE，B-only，九次A刷新',
    'Pilot_c3': 'CE，B-only，九次gap加权A刷新',
    'REF_AB_CE': '历史AB联训CE，effective-SVD聚合（参照）',
    'REF_CAPT': 'CAPT对齐配置，提示学习/cluster（参照）',
}
LA_DESIGNS = {
    'e0': '无LA，固定A，九次额外B',
    'e1': '无LA，B-only，九次A刷新',
    'e2': 'LA，固定A，九次额外B',
    'e3': 'LA，B-only，九次A刷新',
    'e5': 'LA，功能控制A/B候选、look-ahead及冻结',
    'j': 'LA，日常AB联训，保留九次额外A',
    's': 'LA，B-only，前90轮逐轮A刷新',
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Reuse the prior source inventory, not its previously computed scores.
    source = REPO / 'output/lora_ab_separation_review_20260916/analyze.py'
    spec = importlib.util.spec_from_file_location('previous_ab_inventory', source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    runs = [dict(r) for r in previous.RUNS if not r['duplicate_of'] and r['family']!='LA_control']
    for path in sorted(LATEST.glob('*/*/*/control_config.json')):
        cfg = json.loads(path.read_text(encoding='utf-8'))
        assert (path.parent / 'completion.json').is_file(), path
        runs.append({'id':f'LA_{cfg["topology"]}_{cfg["method"]}', 'family':'LA_control',
                     'source':path.parent.relative_to(REPO).as_posix(), 'rank':4,
                     'topology':cfg['topology'], 'role':'main', 'seed':cfg['seed'],
                     'design':LA_DESIGNS[cfg['method']]})
    results = []
    for run in runs:
        path = REPO / run['source']
        curve = {}
        for row in read_csv(path / 'round_metrics.csv'):
            rnd = int(row['round']) if 'round' in row else int(row['epoch'])+1
            vals = {k:float(row[k]) for k in ('overall_acc','non_tail_acc','bottom20_tail_acc')}
            if rnd in curve:
                assert curve[rnd] == vals, (path,rnd)
            curve[rnd] = vals
        assert set(range(1,101)) <= set(curve), path
        scores = {key:mean(curve[r][field] for r in range(81,101)) for key,field in
                  [('overall','overall_acc'),('non_tail','non_tail_acc'),('tail','bottom20_tail_acc')]}
        per_class = []
        for rnd in range(81,101):
            class_path = path / f'per_class_accuracy_epoch_{rnd-1}.csv'
            if not class_path.exists():
                break
            classes = {int(r['class_id']):float(r.get('per_class_acc',r.get('accuracy')))
                       for r in read_csv(class_path)}
            assert set(classes)==set(range(100)), class_path
            assert abs(mean(classes.values())-curve[rnd]['overall_acc'])<1e-8, class_path
            assert abs(mean(classes[c] for c in range(80))-curve[rnd]['non_tail_acc'])<1e-8, class_path
            assert abs(mean(classes[c] for c in range(80,100))-curve[rnd]['bottom20_tail_acc'])<1e-8, class_path
            per_class.append(classes)
        if len(per_class)==20:
            scores['head20'] = mean(mean(v[c] for c in range(20)) for v in per_class)
            scores['middle60'] = mean(mean(v[c] for c in range(20,80)) for v in per_class)
        else:
            scores['head20'] = scores['middle60'] = ''
        partition = run['topology']
        config_path = path / 'control_config.json'
        if config_path.exists():
            assert json.loads(config_path.read_text(encoding='utf-8'))['topology']==partition
        design = DESIGNS.get(run['id'], run['design'])
        if run['family']=='CE_bridge':
            design = 'CE桥接，固定A，九次额外B' if run['id'].endswith('c1') else 'CE桥接，B-only，九次A刷新'
        status = ('补跑普通Dirichlet；旧结果保留为固定容量控制' if partition=='matched-dirichlet'
                  else '保留；不因Dirichlet协议修订而重跑')
        results.append({'id':run['id'],'family':run['family'],'design':design,
                        'partition':partition,'rank':run['rank'],'seed':run['seed'],'role':run['role'],
                        **scores,'final_overall':curve[100]['overall_acc'],
                        'final_tail':curve[100]['bottom20_tail_acc'],
                        'rerun_status':status,'source':run['source']})
    main_rows = [r for r in results if r['role']=='main']
    references = [r for r in results if r['role']!='main']
    assert len(main_rows)==26 and sum(r['partition']=='matched-dirichlet' for r in main_rows)==7
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/'all_results.csv',main_rows+references)
    reruns = [r for r in main_rows if r['partition']=='matched-dirichlet']
    write_csv(OUT/'dirichlet_rerun_inventory.csv',reruns)
    lines = ['# A/B框架结果总表与Dirichlet协议修订','',
             '日期：2026-09-18。主指标为第81–100轮平均准确率（%），全部seed42。',
             '26个去重配置；另列历史AB联训和CAPT参照。仅重算CSV，不运行模型。',
             'CLT=Client-LT；Matched=固定Client-LT客户端容量的Dirichlet-like控制，不是普通Dirichlet。',
             '后续普通Dirichlet使用noniid-labeldir-fine，beta=0.5，不固定客户端容量或FedAvg权重。','',
             '| 实验 | 特点 | 划分 | rank | Overall | Non-tail80 | Tail20 |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in main_rows+references:
        label = 'Matched' if row['partition']=='matched-dirichlet' else 'CLT'
        lines.append(f'| {row["id"]} | {row["design"]} | {label} | {row["rank"]} | {row["overall"]:.4f} | {row["non_tail"]:.4f} | {row["tail"]:.4f} |')
    lines += ['', '## 哪些需要重跑','',
              '七组现有Matched结果不符合新的普通Dirichlet主对照要求：CE桥接C1/C2；LA系列E2/E3/E5/J/S。',
              '旧实验本身不是无效数据，保留为固定容量控制；不能改名充当普通Dirichlet。',
              'Client-LT、V1/V2/rank/pilot及CAPT参照不因本次协议修订而强制重跑。',
              'E0/E1此前没有Dirichlet完整结果；若新增普通Dirichlet E0/E1，是补充实验，不是纠正已有错误划分。',
              '优先级：E3/J/S，然后E2/E5，最后补齐旧CE桥接C1/C2。完整替换共七组。','',
              '## 去重与可比性','',
              '- rank2基线即V2 FedAvg；C0即rank4基线，不重复计数。',
              '- CE桥接Client-LT C1/C2与pilot对应曲线相同，只计一次；桥接提供了额外归因。',
              '- LA旧四组、六组、消融包与最新J/S包的重复结果只取最新完整包。',
              '- C1/C2和E0/E1不是同一代码/RNG协议，不合并为一组。',
              '- V1为AMP、rank2及私有B；后续多为FP32、rank4；不能跨行作纯组件因果比较。',
              '- S有更多训练预算；E5包含未提交候选和look-ahead成本。',
              '- Head20/Middle60仅在完整末20轮逐类文件存在时填入CSV；缺失不推造。',
              '- 更早的SupportNormalized/LA-aggregation等rank2 AB联训，不属于确立固定A/分阶段框架后的主线；不与本表混作结构消融。',
              '- 普通Dirichlet与CLT仍使用同一全局LT池，但n_k、q_k及batch取整步骤可不同。',
              '- 不重跑CLT时，旧CLT/新Dir是跨代码版本描述性比较；汇总会显式标记，不冒充同版本。',
              '- 新Dir成绩与旧Matched成绩必须分列；旧Matched不遗忘不能推断普通Dir也不遗忘。','',
              '命令与实现说明见仓库 docs/cliplora_standard_dirichlet_rerun.md。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(main_rows)} configurations, {len(references)} references, {len(reruns)} standard-Dirichlet reruns')
    print(OUT/'report.md')


if __name__=='__main__':
    main()
