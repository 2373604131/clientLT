"""Build a descriptive, source-linked inventory; read existing results only."""
import csv
import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'presentation' / 'lora_ab_inventory_20260919'
TOPO = {'client-longtail': 'Client-LT', 'matched-dirichlet': '固定容量狄利克雷',
        'noniid-labeldir-fine': '普通狄利克雷 β=0.5'}
ROWS = []


def read(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def add(name, topology, normal, extra, loss='普通 CE', aggregation='按样本量聚合',
        rank=4, precision='FP32', source='', category='分离训练主线', note='',
        status='已完成100轮', lora='有', metrics=None):
    row = {'类别': category, '实验具体设置': name, '是否LoRA': lora,
           '数据划分': TOPO.get(topology, topology), 'rank/精度': f'{rank}/{precision}' if lora == '有' else '不适用',
           '正常训练阶段': normal, '额外更新阶段': extra, '损失/类别校正': loss,
           '服务器聚合/接收方式': aggregation, '状态与统计口径': status,
           'Overall': '', 'Non-tail80': '', 'Tail20': '', 'Tail第100轮': '',
           'Tail峰值到最终下降pp': '', '说明': note, '结果来源': source}
    if source.endswith('round_metrics.csv'):
        curve = {}
        for r in read(source):
            rnd = int(r['round']) if 'round' in r else int(r['epoch']) + 1
            values = {k: float(r[k]) for k in ('overall_acc', 'non_tail_acc', 'bottom20_tail_acc')}
            assert rnd not in curve or curve[rnd] == values, source
            curve[rnd] = values
        assert set(range(1, 101)) <= curve.keys(), source
        for field, metric in [('Overall', 'overall_acc'), ('Non-tail80', 'non_tail_acc'), ('Tail20', 'bottom20_tail_acc')]:
            row[field] = mean(curve[r][metric] for r in range(81, 101))
        row['Tail第100轮'] = curve[100]['bottom20_tail_acc']
        row['Tail峰值到最终下降pp'] = max(curve[r]['bottom20_tail_acc'] for r in range(1, 101)) - row['Tail第100轮']
    if metrics:
        row.update(metrics)
    ROWS.append(row)
    return row


def main():
    old = read('output/ab_framework_partition_review_20260918/all_results.csv')
    for r in old:
        key, family = r['id'], r['family']
        kw = dict(topology=r['partition'], rank=r['rank'], source=r['source'] + '/round_metrics.csv')
        if family == 'V1':
            n = key.split('_L')[1]
            add(f'固定A、保留私有B，接收时保护{n}个受损类别', normal='冻结A，只更新客户端私有B',
                extra='无独立A更新', aggregation=f'私有B持续保留；Top-{n}类别投影/回溯接收保护', precision='AMP', **kw)
        elif family == 'V2':
            method = key.split('_')[1]
            names = {'fedavg': '固定A，只训练B，样本量聚合', 'static': '固定A，只训练B，静态调整客户端权重',
                     'progressive': '固定A，只训练B，逐步调整客户端权重'}
            agg = {'fedavg': 'B按样本量q聚合', 'static': 'B权重=0.12q+0.88p',
                   'progressive': 'B权重=(1−β)q+βp；β=clip((轮次−5)/15,0,1)'}
            add(names[method], normal='冻结A，只更新B；每轮3 epoch', extra='无', aggregation=agg[method],
                note='p由类别先验平衡求解；客户端训练仍为CE，未加训练logit校正', **kw)
        elif family == 'Rank':
            add(f'固定A，只训练B，rank={r["rank"]}', normal='冻结A，只更新B；每轮3 epoch', extra='无',
                note='alpha=1，实际缩放=1/√rank；改变rank也改变缩放与初始子空间', **kw)
        elif family in ('A_refresh', 'CE_bridge'):
            method = key.rsplit('_', 1)[1]
            extra = '9次额外B更新，各1 epoch' if method == 'c1' else '9次额外A更新，各1 epoch；冻结共同B'
            name = {'c1': '固定A，正常训练B并额外训练B（早期协议）',
                    'c2': '正常只训练B，间歇额外更新A（早期协议）',
                    'c3': '正常只训练B，以本地功能缺口加权额外更新A'}[method]
            add(name, normal='冻结当前A，只更新B；每轮3 epoch', extra=extra,
                loss='正常CE；额外A用频数归一化的缺口加权CE' if method == 'c3' else '普通 CE',
                note='额外阶段在第10、20、…、90轮；早期评估/RNG协议，与后期CE组分别保留', **kw)
        elif family == 'LA_control':
            method = key.rsplit('_', 1)[1]
            configure_control(method, kw)
        elif key == 'REF_AB_CE':
            add('A/B全程联合更新，普通CE，有效矩阵平均后SVD', normal='A、B同时更新；每轮3 epoch', extra='无独立A阶段',
                aggregation='先平均有效LoRA矩阵，再截断SVD回rank2', category='历史联训参照',
                note='与固定A主线在rank和聚合等方面有差异，不能当纯冻结消融', **kw)
        else:
            add('CAPT提示学习与客户端协作对照', normal='更新提示及CAPT原有可训练组件', extra='无A/B更新',
                loss='CAPT原生目标', aggregation='cluster协作；每轮全局聚合', lora='无', category='非LoRA参照',
                note='预算对齐配置；模型结构与LoRA不同', **kw)
        # Check previous inventory against original per-round data.
        for field, old_field in [('Overall', 'overall'), ('Non-tail80', 'non_tail'), ('Tail20', 'tail')]:
            assert abs(ROWS[-1][field] - float(r[old_field])) < 1e-8

    fresh = ROOT / 'output/la_control_standard_dirichlet_e3_j_s_analysis/la_control/seed42/noniid-labeldir-fine'
    for method in ('e3', 'j', 's'):
        paths = list((fresh / method).glob('*/round_metrics.csv'))
        assert len(paths) == 1
        configure_control(method, dict(topology='noniid-labeldir-fine', rank=4, source=paths[0].relative_to(ROOT).as_posix()))
    assert len(ROWS) == 31
    main_rows = list(ROWS)

    # Earlier joint-A/B aggregation experiments belong to the history, not a freeze-only ablation.
    for topology in ('clientlt', 'dirichlet'):
        for method in ('fedavg', 'support_normalized'):
            source = f'output/cifar100_LT/ClipLora_SupportNormalized_2x2_seed42/seed42/{topology}/{method}/round_metrics.csv'
            add('A/B联合更新，' + ('样本量聚合' if method == 'fedavg' else '按尾类支持者归一化聚合'),
                'client-longtail' if topology == 'clientlt' else 'noniid-labeldir-fine',
                normal='A、B同时更新；每轮3 epoch', extra='无', rank=2, precision='AMP',
                aggregation='分别平均A/B；' + ('权重q=n/N' if method == 'fedavg' else '每尾类在支持客户端内按样本量归一化，再对尾类等权平均'),
                source=source, category='历史联训与聚合探索', note='早期联合训练协议；不是固定容量狄利克雷')
    for p in sorted((ROOT / 'output/la_lambda_sweep_seed42_results').rglob('round_metrics.csv')):
        summary = read(p.parent.relative_to(ROOT) / 'lora_aggregation_summary.csv')[0]
        reg = float(summary['la_regularization'])
        add(f'A/B联合更新，类别先验平衡聚合，正则系数={reg:g}', 'client-longtail',
            normal='A、B同时更新；每轮3 epoch', extra='无', rank=2, precision='AMP（入口配置）',
            aggregation='类别先验平衡求客户端权重；有效矩阵平均后SVD', source=p.relative_to(ROOT).as_posix(),
            category='历史联训与聚合探索', note='此处LA作用于服务器聚合权重，不是训练logit校正；温度=1')
    names = {
        'matched_memory_ce': ('A/B联合更新，加本地记忆样本CE', '第3 epoch加入记忆样本CE'),
        'class_reweighted_ce': ('A/B联合更新，加类别重加权的记忆CE', '第3 epoch加入本地频数逆平方根加权记忆CE'),
        'instantaneous_target': ('A/B联合更新，保持本轮已达到的功能目标', '第3 epoch加入本轮目标的平方正部缺口损失'),
        'pfrf_max': ('A/B联合更新，保持跨轮历史最高功能目标', '第3 epoch加入持久历史最高目标的缺口损失'),
        'pfrf_add': ('A/B联合更新，累计未兑现目标与本轮功能增益', '第3 epoch加入累计目标的缺口损失；目标上限1'),
    }
    for method, (name, loss) in names.items():
        add(name, 'client-longtail', normal='A、B同时更新；2 epoch CE提案＋1 epoch校正',
            extra='无独立A阶段；辅助损失合入第3 epoch', loss='CE；' + loss, rank=2,
            aggregation='有效矩阵平均后SVD；样本量权重', category='历史功能保持探索',
            source=f'output/pfrf_kill_v1/runs/{method}/seed42/round_metrics.csv',
            note='与历史普通CE参照同一六组实验；辅助损失系数1；不是A/B分离')

    DEST.mkdir(parents=True, exist_ok=True)
    assert len(ROWS) == 44
    assert all((ROOT / r['结果来源']).is_file() for r in ROWS)
    for r in read('presentation/source_standard_dirichlet.csv'):
        if r['topology'] != 'noniid-labeldir-fine':
            continue
        found = [x for x in main_rows if f'/{r["method"]}/' in x['结果来源'] and x['数据划分'] == TOPO['noniid-labeldir-fine']]
        assert len(found) == 1
        assert abs(found[0]['Overall'] - float(r['overall'])) < 1e-8
        assert abs(found[0]['Tail20'] - float(r['tail'])) < 1e-8
    fields = list(ROWS[0])
    with (DEST / '全部实验大表.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ROWS)
    intro = [
        '# LoRA A/B 分离及直接相关实验总表', '',
        '整理日期：2026-09-19。表内使用具体设置，不使用实验代号。所有成绩重新读取原始CSV；没有重新训练模型。', '',
        f'共{len(ROWS)}组已完成100轮训练的结果（含非LoRA的CAPT参照）。仅统计完整100轮训练，不列单轮诊断和未完成配置。', '',
        '范围：固定A/B-only、独立A刷新、AB联训对照、接收保护、相关聚合调权及历史功能保持。未把仓库其他PromptFL、FedTEF或纯拓扑扫描混成A/B分离实验。', '',
        'Overall、Non-tail80、Tail20均为第81–100轮平均准确率（%），seed=42。CSV另保留第100轮尾类准确率及尾类峰值到最终的下降幅度。', '',
        'Client-LT、普通狄利克雷（noniid-labeldir-fine，β=0.5）、固定容量狄利克雷（matched-Dirichlet）严格分列。固定容量版本匹配Client-LT客户端样本量；普通版本不匹配。', '',
        '完整LoRA训练通常为CLIP ViT-B/16、视觉top3 Q/V、alpha=1、30客户端全参与、100轮、正常3个本地epoch、batch32；rank与精度逐行列出。CAPT结构与LoRA不同。', '',
        '“额外9次”是第10、20、…、90轮正常B聚合后，各加1个本地epoch；更新A时冻结本轮聚合后的共同B。“密集90次”是第1–90轮每轮加1个A epoch，第91–100轮只训练B。', '',
        '训练logit校正LA（τ=1）只改训练损失，测试不添加校正。历史“类别先验平衡聚合”只改服务器权重，两者不能混称。', '',
        '去重：重复打包的训练结果只取完整包；早期Client-LT桥接曲线与对应A刷新试验相同，合并。rank2与rank4的重复基线合并。早期CE刷新与后期CE刷新存在评估/RNG协议差异，即使得分接近也分别保留。', '',
    ]
    table_fields = ['类别', '实验具体设置', '是否LoRA', '数据划分', 'rank/精度', '正常训练阶段', '额外更新阶段',
                    '损失/类别校正', '服务器聚合/接收方式', '状态与统计口径', 'Overall', 'Non-tail80', 'Tail20', '说明', '结果来源']
    lines = intro + render(ROWS, table_fields) + ['',
        '预算说明：Client-LT固定A纯B训练为105600步；9次额外A/B的路径为108768步；密集90次A为137280步。普通狄利克雷相应9次/90次路径为109386/138060步。联合AB对照每步开放两个因子，步数相同不等于FLOPs相同。', '',
        '功能控制组实际尝试9次，接受7次A更新；包含未选分支及2轮前瞻，总130944步，并在第92轮永久冻结A。不能当成普通9次A刷新或同计算量对照。', '',
        '历史/跨版本结果用于描述性比较；所有100轮成绩为单seed结果，不表示跨种子稳定性或显著性。', '']
    (DEST / '全部实验大表.md').write_text('\n'.join(lines), encoding='utf-8')
    write_html(ROWS, table_fields, intro)
    (DEST / '主线31组.md').write_text('\n'.join(intro[:4] + render(main_rows, table_fields)), encoding='utf-8')
    (DEST / '核对记录.json').write_text(json.dumps({'total_rows': len(ROWS), 'categories': dict(Counter(r['类别'] for r in ROWS)),
        'full_training_runs': sum(r['结果来源'].endswith('round_metrics.csv') for r in ROWS),
        'original_main_rows_recomputed_and_matched': len(old), 'new_standard_dirichlet_runs': 3,
        'scope': 'completed_100_round_training_only', 'training_started': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'rows': len(ROWS), 'full_runs': sum(r['结果来源'].endswith('round_metrics.csv') for r in ROWS),
                      'destination': str(DEST)}, ensure_ascii=False))


def configure_control(method, kw, pending=False):
    names = {'e0': '固定A，正常与额外阶段都只更新B（后期CE协议）',
             'e1': '正常只更新B，间歇额外更新A（后期CE协议）',
             'e2': '加训练LA，固定A，正常与额外阶段都更新B',
             'e3': '加训练LA，正常只更新B，间歇额外更新A',
             'e5': '加训练LA，功能控制选择额外A或B更新',
             'j': '加训练LA，正常A/B联合更新，并保留额外A更新',
             's': '加训练LA，正常只更新B，前90轮每轮额外更新A'}
    extra = {'e0': '9次额外B，各1 epoch', 'e1': '9次额外A，各1 epoch；冻结共同B',
             'e2': '9次额外B，各1 epoch', 'e3': '9次额外A，各1 epoch；冻结共同B',
             'e5': 'A/B两候选＋2轮前瞻；门控、冻结与有效锚点恢复',
             'j': '9次额外A，各1 epoch；冻结共同B',
             's': '90次额外A，各1 epoch；冻结共同B'}
    note = {'e0': '早期CE协议结果不与本行合并', 'e1': '早期CE协议结果不与本行合并',
            'e2': '全程A不更新', 'e3': 'A只在9个额外阶段更新',
            'e5': '实际接受7次A；第92轮永久冻结；含额外分支/前瞻计算',
            'j': '不是单纯AB联训：9次额外A仍保留；A/B分别平均，无SVD',
            's': '预算高于9次刷新；第91–100轮只训练B'}[method]
    add(names[method], normal='A、B同时更新；每轮3 epoch' if method == 'j' else '冻结当前A，只更新B；每轮3 epoch',
        extra=extra[method], loss='普通CE，τ=0' if method in ('e0', 'e1') else '训练logit校正LA，τ=1',
        note=note if not pending else '补跑入口已定义；当前仓库未发现对应完成结果',
        category='已定义但未见结果' if pending else '分离/联合/交替训练对照',
        status='仓库未见完成结果' if pending else '已完成100轮', **kw)


def render(rows, fields):
    lines = ['| ' + ' | '.join(fields) + ' |', '|' + '|'.join('---' for _ in fields) + '|']
    for r in rows:
        values = []
        for field in fields:
            v = r[field]
            if field == '结果来源' and v:
                v = f'[原始记录](../../{v})'
            elif isinstance(v, float):
                v = f'{v:.4f}'
            values.append(str(v).replace('|', '／').replace('\n', ' '))
        lines.append('| ' + ' | '.join(values) + ' |')
    return lines


def write_html(rows, fields, intro):
    head = ''.join(f'<th>{html.escape(f)}</th>' for f in fields)
    body = []
    for row in rows:
        cells = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                value = f'{value:.4f}'
            value = html.escape(str(value))
            if field == '结果来源' and value:
                value = f'<a href="../../{value}">原始记录</a>'
            cells.append(f'<td>{value}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    description = ''.join(f'<p>{html.escape(p)}</p>' for p in intro[2:] if p)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>LoRA A/B相关实验大表</title><style>
body{font:14px/1.6 system-ui,sans-serif;color:#203047;margin:24px;background:#f7f9fc}
h1{font-size:24px;margin:0 0 12px}p{max-width:1300px}input,select{padding:9px;margin:5px;border:1px solid #bbc8d8;border-radius:5px}
input{width:360px}.wrap{overflow:auto;max-height:76vh;border:1px solid #ccd5df;background:white}
table{border-collapse:separate;border-spacing:0}th{position:sticky;top:0;background:#e6eef8;z-index:2;text-align:left}
td,th{padding:9px 12px;border-bottom:1px solid #dbe2eb;min-width:105px;vertical-align:top}
td:nth-child(2),th:nth-child(2){min-width:265px}td:nth-child(6),td:nth-child(7),td:nth-child(9),td:nth-child(14){min-width:235px}
tbody tr:nth-child(even){background:#f7faff}a{color:#195aa7}details{margin:12px 0}#count{margin:0 12px}</style>
<h1>LoRA A/B 分离及直接相关实验大表</h1>
<p>仅统计44组已完成100轮训练的结果。Overall、Non-tail80、Tail20均为第81–100轮平均准确率（%），seed=42。</p>
<input id="search" placeholder="搜索具体设置，例如：冻结A、额外、功能保持">
<select id="topology"><option value="">全部数据划分</option><option>Client-LT</option><option>普通狄利克雷 β=0.5</option><option>固定容量狄利克雷</option></select>
<span id="count"></span><details><summary>统计口径、去重规则与可比性说明</summary>DESCRIPTION</details>
<div class="wrap"><table><thead><tr>HEAD</tr></thead><tbody>BODY</tbody></table></div>
<script>const trs=[...document.querySelectorAll('tbody tr')];function filter(){let n=0;for(const r of trs){const ok=r.textContent.toLowerCase().includes(document.getElementById('search').value.toLowerCase())&&(!document.getElementById('topology').value||r.cells[3].textContent===document.getElementById('topology').value);r.hidden=!ok;if(ok)n++;}document.getElementById('count').textContent=n+' / '+trs.length+' 行';}for(const id of ['search','topology'])document.getElementById(id).addEventListener('input',filter);filter();</script></html>'''
    page = page.replace('DESCRIPTION', description).replace('HEAD', head).replace('BODY', ''.join(body))
    (DEST / '全部实验大表.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
