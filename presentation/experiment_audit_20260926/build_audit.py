"""Build a narrative-aligned experiment ledger without changing source results."""
from __future__ import annotations
import csv, hashlib, html, io, json, re, statistics, tarfile, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape
from inventory_repository import ROOT, OUT, write_csv

DATA=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
RUNS=DATA['runs']
BYID={r['run_id']:r for r in RUNS}
CORE='必须保留'; WAIT='待定用途'; ARCHIVE='建议归档'; INFRA='工程或索引'
RULES={}
def rule(names,category,slot,use,conclusion,action):
    for name in names.split('|'):
        RULES[name]=dict(category=category,story_slot=slot,use=use,conclusion=conclusion,next_action=action)

rule('eri_seed42_share',CORE,'1 现象与分配对照','开篇 FedAvg＋LoRA 训练轨迹；固定边际拓扑对照','Client-LT 尾类峰值68.80→37.15；matched Dir 68.75→49.40。支持曾有能力随后退化。','正文保留两条完整曲线；同时列固定边际与参与率。')
rule('eri_closure_v1',CORE,'2 功能机制','W/H/D/R 正负贡献预算与尾类保持','直接支持者正贡献 W 下降更明显；非持有者既可帮助也可损害。H2部分支持，H3未运行。','保留机制解释及边界；不得宣称完整因果闭环。')
rule('cifar100_LT/ClipLora_SupportNormalized_2x2_seed42',CORE,'2 功能机制','普通 Dirichlet 下支持归一化干预，补充检验聚合影响','Client-LT final Tail 36.75→45.80；Dir 48.70→49.40。协议与固定边际 ERI 不同。','保留四格与代价；放附录或机制补充，不能充当 ERI 的 H3。')
rule('la_control_ablation_analysis|la_control_js_topology_analysis|la_control_standard_dirichlet_e3_j_s_analysis|la_control_dirichlet_supplement_analysis',CORE,'3 适配与保持','CE/LA × 固定/开放 A；E3/J/S 训练方式对照','LA 下 E3 比固定 A 的 E2 同时提高 Overall/Tail；密集 S 和联合 J 有额外尾类回落。','使用完整结果及逐类指标；E5单独归档；普通和 matched Dir 分栏。')
rule('la_control_analysis|la_control_six_runs_analysis',ARCHIVE,'3 适配与保持','早期 LA 结果回传包','已被后续更完整包覆盖；部分目录只有22/34轮。','保留引用映射，停止当作新增实验或待补跑任务。')
rule('sfra_v1_analysis',CORE,'4 主方法 A','来源、历史功能目标与功能修正的早期对照','Full λ10较同频 S 提高 Tail，牺牲部分 Overall；Current/Flat 是无CP的旧消融。','Full λ10作无CP消融；其余作为历史证据或超参附录。')
rule('sfra_cp_analysis',CORE,'4 主方法 A','Full-CP 主方法与分类正则 μ 扫描','μ1 late Tail70.935、Overall70.633；μ3提高头中部但降低尾类保持。','μ1作为当前约定主设置；μ0.3/3完整保留为敏感性，勿按测试峰值选优。')
rule('sfra_cp_analysis_new',CORE,'4 主方法 A','Flat-CP 去掉来源优先级','Full-CP比Flat-CP末20轮Tail高0.880点；Overall低0.0795点；回落少1.10点。','保留单因素规则对照，注明fast执行路径差异；补齐同版Current-CP。')
rule('sfra_b_shared_transfer_analysis',CORE,'5 辅助方法 B','当前共享 B 迁移及 A-only 对照','对 A-only late Tail+0.1525，Overall+0.0695，Few30约+0.3717；单seed增益较小。','若论文保留B模块，此对照必须保留，但不足以声称显著抗遗忘。')
rule('sfra_b_aggregation_analysis',CORE,'5 辅助方法 B','uniform8 × 有无旧B迁移的必要负对照','uniform8下B额外late Tail仅+0.0225；更换权重并未显示大幅增益。','保留弱收益证据；不能与A-only样本权重基线跨配方直接归因。')
rule('sfra_b_transfer_run2_analysis',WAIT,'5 辅助方法 B','旧版接收端 B 迁移学习率扫描','lr0.03/0.1/0.3均只有很小正向差异；与共享B实现不同。','作为版本追踪或附录；停止扩大旧版扫描，优先检验共享B实际价值。')
rule('factor_roles_analysis',CORE,'边界与反证','检查 A/B 是否天然分工为客户端/类别知识','6个共同锚点均未通过预定义角色交叉条件，幅度匹配后也不支持。','保留反证以约束写法；不必占正文主图，停止用天然知识分工解释模块。')
rule('a_refresh_topology_bridge_analysis',WAIT,'2 功能机制','CE下 C1/C2 × 两种固定边际拓扑的阶段归因','C2两种拓扑都退化；Client-LT放大late Tail损失1.68点；后续B阶段有明显路径依赖。','机制附录候选；不与LA下E3学习收益混为同一协议。')
rule('a_refresh_c1_c2_c3_analysis',ARCHIVE,'旧 A 方案','CE/gap刷新A pilot','C3相对C2 late Tail更低3.55点；更新幅度也更大；当前配方失败。','停止扩展gap配方；C1/C2已有桥接复用，留一份失败结论。')
rule('rank4_fedavg_analysis|rank8_fedavg_analysis|rank_sweep_analysis',WAIT,'配置与稳健性','固定 A rank2/4/8 对照','rank4略优；alpha固定导致rank与缩放同时变化，不能得出纯容量结论。','作为配置出处/附录；不需继续无目的扫rank。')
rule('v2_capt_analysis',WAIT,'外部方法与旧聚合','固定 A 的 FedAvg/static/progressive 与 CAPT','static/progressive对Tail有小幅改善；CAPT整体更高、Tail更低；协议需明确。','rank2旧方法比较可留附录；最终外部基线先审客户端起点、预算与测试反馈。')
rule('selective_sync_v1_training_check',ARCHIVE,'旧 A/B 方案','private B / selective sync L1/L3/L5/L10','该路线与最终单一共享模型及当前SFRA方法不同；不能替代主方法A/B消融。','保留结果和实现，停止作为当前论文主实验扩展。')
rule('la_lambda_sweep_seed42_results',ARCHIVE,'旧聚合方案','旧服务器 LA 聚合正则扫描','这是服务器聚合权重方案，不是当前本地LA训练；不能混淆同名λ。','版本归档；当前主线不再扩展此扫描。')
rule('pfrf_kill_v1',ARCHIVE,'旧抗遗忘方案','PFRF max/add、瞬时目标、匹配记忆与类别加权','PFRF-max late Tail36.905，低于matched-memory40.10和class-reweighted41.325。','保留失败与负对照；停止用旧PFRF命名统领当前SFRA故事。')
rule('e1_seed42_results_for_analysis',WAIT,'边界与反证','strength/breadth 旧机制门槛','预注册结论 STRONG_BUT_NOT_NARROW；不支持“学得强但功能必然更窄”。','保留反证；不能作为来源稀缺必然低功能广度的正证据。')
rule('carrier_access_audit',WAIT,'2 功能机制','carrier与候选donor访问、局部重适配诊断','训练侧私有信号有正向排序信息；不同条件主要为margin变化，非完整训练收益。','若需要解释B的反馈信号可引用；先对齐当前B协议。')
rule('post_write_rewrite_audit',WAIT,'2 功能机制','固定更新的写入后改写与风险重放','D1不支持完整donor→rewriter转变链；D2支持风险与保持关系。','机制补充候选；注明固定更新重放，不是动态联邦重训练。')
rule('compatibility_retention_bridge',WAIT,'2 功能机制','局部更新兼容性到一步保持的桥接','该局部实验门槛通过；不能解释全部长程拓扑精度差。','与当前功能保持有联系时放附录，否则归档。')
rule('functional_breadth_p0_p1_seed42',ARCHIVE,'旧机制','功能广度P0/P1旧导出','后续v2修正有效单元口径；旧P0数量不应继续引用。','只保留版本历史，优先使用v2。')
rule('functional_breadth_p0_p1_seed42_v2',WAIT,'边界与反证','修正版功能广度门槛','全参与Client-LT P0合格单元为0；P1只部分支持（4/20尾类）。','保留证据边界；不能当作普遍机制结论。')
rule('v2_v3_semantic_acquisition',ARCHIVE,'旧语义形成假说','语义共现→功能形成的机制链','V2 NO_FUNCTIONAL_SUPPORT；联合结论FORMATION_CHAIN_NOT_SUPPORTED。','停止沿“语义共现不足已被证明”写故事；留负结果索引。')
rule('p0_v1_context_colocation_v2',WAIT,'设定与代理指标','语义邻居共现的受控代理分析','发现代理广度差异；不是梯度作用、保持效果或准确率因果验证。','只作设定分析备选，不替代功能归因。')
rule('topology_breadth_phase2_seed42',WAIT,'2 功能机制','固定边际更新功能广度诊断','仿真侧机制审计；单seed，full availability与frac0.4需区分。','如要引用逐项复核其门槛；当前主线由ERI承担。')
rule('functional_cusp_gate_seed42',ARCHIVE,'旧 gate 方案','functional CUSP gate','已有两拓扑门槛汇总结论FAIL。','保留失败记录，不继续并入当前A方法。')
rule('boundary_gate_clientlt_seed42|cusp_minimal_refactor_20260801_163123|cusp_minimal_seed42_refactor_20260801_042324|stage3_two_round_seed42',ARCHIVE,'旧 gate / oracle','短程gate、oracle及两轮验证','不能替代100轮单共享模型的适配与保持验证。','停止作为论文主结果；留工程/可行性历史。')
rule('stage2c',WAIT,'旧机制','共享基底与残差时间错配诊断','旧残差放到新共享模型退化；属于另一套残差方法协议。','必要时作设计动机附录，勿当成当前LoRA A/B因果分工。')
rule('online_sca_seed42_v2',ARCHIVE,'旧 SCA 方案','80轮online SCA与残差FedAvg/CAPT','拓扑gate无方向性支持；旧prompt/residual协议不同。','保留80轮原协议结果，停止继续扩展SCA路线。')
rule('PromptFL_fedavg_vit_b16_batchSize32|cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32|cd_two_seed_summary|expD_dirichlet_vs_clientlt',WAIT,'跨方法现象','PromptFL Panel C / Experiment D、局部epoch和历史机制','可说明现象在旧prompt协议下存在；有副本、失败seed与不同localE。','附录稳健性候选；不能拼成当前LoRA多seed结果。')
rule('expC_lambda_response_frac02',WAIT,'设定敏感性','PromptFL/CAPT/zero-shot × specialization λ × seed','不同方法对拓扑响应不同；CAPT采用稀疏聚合/评估，zero-shot只需一次评测。','设定附录候选；不按100轮连续指标强行平均，不直接排序跨协议。')
rule('capt_global_start_seed42_results|cifar100_LT/CAPT_cluster_vit_b16_batchSize32|cifar100_LT/capt_main_matched',WAIT,'外部方法协议','CAPT历史版本、global-start与cluster诊断','高尾类精度受类别提示门槛影响；部分版本有测试反馈或优化器状态共享。','保留协议诊断；未统一之前不作最终严格主表。')
rule('capt_code_audit_20260914',CORE,'证据质量','CAPT协议代码审计','用于识别客户端模型起点、测试驱动调度和聚合规则差异。','保留审计以定义公平外部基线，审计文件不是新实验。')
rule('cifar100_LT/clientLT_seed42',ARCHIVE,'无效历史结果','旧LoRA生命周期异常运行','已有修复报告明确不作为有效LoRA训练结果。','排除论文数值与重复种子统计；原始文件留档。')
rule('cifar100_LT/lora',ARCHIVE,'旧单边pilot','修复后早期LoRA Client-LT单边曲线','只可说明趋势，缺同协议配对；已被2×2和ERI替代。','保留历史，不再单独作为主对照。')
rule('phase1_exposure_topology|experiment1_global_longtail_verification|strict_exp1_fresh_topology|topology_sweep|partition_checks',WAIT,'设定与结构','划分正确性、覆盖/暴露和参数扫','能验证数据分配结构；不能直接证明遗忘机制。','保留准确对应当前分配版本的一套，其他放附录或版本历史。')
rule('phase2_failure_mechanisms',WAIT,'旧机制','暴露→本地收益→聚合/保持诊断','含必要成对日志规范；分析版本早于当前ERI与SFRA。','按需查数据来源；不重复制作同类主图。')
rule('many_medium_few_reanalysis_20260925',CORE,'指标与复算','按频数重分 Many/Medium/Few','35/35/30与Head20/Middle60/Tail20不是同一分组。','保留统一指标表；这是已有训练结果重算，不新增运行。')
rule('ab_framework_partition_review_20260918|lora_ab_separation_review_20260916',INFRA,'历史审阅','旧A/B总表与审查','保留历史配置解释；其中“待补”状态可能已过时。','本次总账为索引入口，引用原始运行而非重复计数报告。')
rule('capt_shared_schedules|expF_shared_schedules|pfrf_smoke_unit_audit|pfrf_smoke_unit_local|test_v0_synthetic|test_v0_synthetic_final|test_v0_synthetic_final2|tmp|experiments|exp1_longtail_per_client',INFRA,'工程或旧图','日程、入口脚本、合成测试、临时参考图','不属于独立科学实验结果；仅有图片不能还原训练证据。','保留工程用途；不进入论文运行数。')
rule('test',ARCHIVE,'旧日志结果','test目录的旧ACPFL评测日志','日志末次Overall94.04%；Head/Medium/Tail标签为旧75%/95%/5%口径，不可当当前Tail20。','保存日志独立摘要；不并入当前CIFAR100-LT正式主表。')
rule('.DS_Store|cifar10_LT/.DS_Store|cifar100_LT/2dfea979a307d2b5c17c368507cc5474.jpg|cifar100_LT/output.tar.gz|cifar100_LT/gate_summary.json|d4a_per_class_round.csv',INFRA,'零散文件与归档','附件、压缩包或派生统计','归档内容另行核对；不把每个文件算一个实验族。','建立索引，不自动删除。')
rule('cifar10_LT/CAPT_cluster_vit_b16_batchSize32',ARCHIVE,'未形成证据','CIFAR10旧日志目录','未发现可汇总的逐轮准确率文件。','只记本地无完整结果，不声称跨数据集已验证。')
for f in DATA['families']:
    if '/fedtef_' in f['family']:
        rule(f['family'],ARCHIVE,'旧 FedTEF 路线','prompt/adapter/memory/router/acquisition多版本历史探索','研究对象与当前SFRA A/B路线不同；v5组件阶梯中融合、TailAgg和rescue未显示预期收益。','停止扩展旧分支；保留各版数字和负结果，缺类/同曲线seed单独标注。')
assert set(f['family'] for f in DATA['families']) == set(RULES), 'Every family must have an explicit decision.'

def protocol(r):
    f,p=r['family'],r['run_path'];part=r['partition']
    if f=='eri_seed42_share':return 'P1 原始LoRA CE/rank2/AMP/固定边际双拓扑'
    if 'SupportNormalized' in f:return 'P2 原始LoRA CE/rank2/AMP/普通Dir 2×2'
    if f.startswith('a_refresh'):return 'P3 CE/rank4/FP32/旧刷新与固定边际桥接'
    if f.startswith('la_control'):return 'P4 LA-control/rank4/FP32/'+str(part)
    if f.startswith('sfra'):return 'P5 SFRA/rank4/FP32/LA/Client-LT（执行版本另核）'
    if f.startswith('rank') or f=='v2_capt_analysis':return 'P6 固定A/rank扫描或预算对齐CAPT（逐行区分）'
    if 'fedtef' in f:return 'H1 历史FedTEF，不能并入SFRA主表'
    if 'PromptFL' in f:return 'H2 历史PromptFL，按参与率/localE/seed另分组'
    if f.startswith('capt') or 'CAPT_' in f or 'capt_main' in f:return 'H3 历史CAPT，检查MAB/起点/优化器/稀疏评估'
    if f=='expC_lambda_response_frac02':return 'H4 frac0.2的λ响应（method/seed独立）'
    return 'H5 其他历史协议；仅在同配置内比较'

cross=defaultdict(list)
LOG_ONLY=[]
for r in RUNS:
    if r['curve_fingerprint']:cross[r['curve_fingerprint']].append(r)
cross_rows=[]
for fingerprint,rs in cross.items():
    if len({str(r['seed']) for r in rs})>1:
        for r in rs:
            r['cross_seed_identical_curve']='X'+fingerprint[:8]
            cross_rows.append({'组':'X'+fingerprint[:8],'run_id':r['run_id'],'记录seed':r['seed'],'路径':r['run_path'],'处理':'不同seed的整条记录精度相同；须追溯来源，不能据此估计跨seed方差。'})

# Prefer complete, centralized result bundles. This is an index choice, not deletion.
priority=['la_control_js_topology_analysis','a_refresh_topology_bridge_analysis','la_control_ablation_analysis','la_control_six_runs_analysis','la_control_analysis']
for group in {r['duplicate_curve_group'] for r in RUNS if r['duplicate_curve_group']}:
    rs=[r for r in RUNS if r['duplicate_curve_group']==group]
    preferred=min(rs,key=lambda r:(priority.index(r['family']) if r['family'] in priority else 99,-r['per_class_files'],len(r['run_path'])))
    for r in rs:
        r['preferred_result_path']=preferred['run_path'];r['is_preferred']=r is preferred

for r in RUNS:
    r.update(RULES[r['family']]);r['protocol_group']=protocol(r)
    r['original_category']=r['category'];r['quality_notes']=[]
    f,p,m=r['family'],r['run_path'],r['method_recorded']
    if f.startswith('la_control') and m=='e5':
        r.update(category=ARCHIVE,use='功能controller/前瞻旧路线',conclusion='表现接近E3，但含较高候选/前瞻成本，不是当前主方法。',next_action='保留旧路线对照；不再优先补E5。')
    if f=='sfra_v1_analysis' and not ('/full/' in p and '/lambda10_' in p):
        r.update(category=WAIT,next_action='旧无CP消融或λ敏感性；不得用current代替current-cp。')
    if f=='sfra_cp_analysis' and '/lambda10_mu1_' not in p:
        r.update(category=WAIT,next_action='保留μ敏感性全结果，说明头中部恢复与尾部保持的权衡。')
    if 'method=zeroshot' in p:
        r['status']='zero-shot单点评测；无需100轮';r['quality_notes'].append('拓扑λ不改变zero-shot测试预测；15份记录不是15次训练。')
    if f=='online_sca_seed42_v2':r['status']='旧协议80轮已记录；不是当前100轮主表'
    if f=='capt_global_start_seed42_results':r['status']='报告核对外层100轮结束；37次服务器评估，最后评估第99轮'
    elif ('capt' in p.lower() or 'CAPT' in p) and (r['last_round'] or 0)<100 and r['observed_rounds']:
        r['quality_notes'].append('CAPT可能按MAB稀疏聚合/评测；最后记录轮<100不能直接判定训练中断。')
    if r.get('incomplete_per_class_files'):
        r['quality_notes'].append(f"{r['incomplete_per_class_files']}份逐类CSV缺类；不填零、不重建缺失类别成绩；保留记录值但不进当前正式表。")
    if r.get('cross_seed_identical_curve'):
        r['quality_notes'].append('不同seed记录的整条精度相同，复现实验独立性待查。')
    if r['duplicate_curve_group'] and not r.get('is_preferred',True):
        r['category']=ARCHIVE;r['next_action']='已有相同曲线的推荐入口；此副本保留追溯，不算新增重复实验。'
    if r['last_round'] is None:
        log=ROOT/r['run_path']/'log.txt'
        text=log.read_text(encoding='utf-8',errors='replace') if log.exists() else ''
        matches=list(re.finditer(r'^Overall accuracy:\s*([\d.]+)%',text,re.M))
        if matches:
            tail=text[matches[-1].start():matches[-1].start()+700]
            vals=dict(re.findall(r'^(Overall accuracy|Head accuracy[^:]*|Medium accuracy[^:]*|Tail accuracy[^:]*):\s*([\d.]+)%',tail,re.M))
            r['status']='只有日志评测摘要，未发现标准逐轮CSV'
            r['quality_notes'].append('日志有准确率，已另表保存；分组与轮次未按当前协议核实，不冒充Tail20曲线。')
            LOG_ONLY.append({'编号':r['run_id'],'来源':r['run_path']+'/log.txt','显式Overall记录数':len(matches),'最后记录':vals,'说明':'日志原文标签；未验证分组语义或重算，不能与正式CSV混算。'})
        else:r['quality_notes'].append('本地未发现标准精度CSV；部分诊断入口按设计在全局测试前结束。')
    if r['last20_tail'] is not None and r['window_end']!=100:
        r['quality_notes'].append(f"这里的末20轮是{r['window_start']}–{r['window_end']}，不是81–100，不进入正式主表。")
    r['quality_notes']='；'.join(r['quality_notes'])

FAMILIES=[]
for f in DATA['families']:
    rr=[r for r in RUNS if r['family']==f['family']]
    FAMILIES.append({**f,**RULES[f['family']],
        'run_categories':dict(Counter(r['category'] for r in rr)),
        'note':'族级保留建议不代表其中每个版本都应进入正文；以逐运行表为准。'})

HEADERS={'run_id':'编号','category':'整理建议','story_slot':'故事环节','use':'实验用途','conclusion':'已有结果如何理解','next_action':'下一步','quality_notes':'证据质量提醒','protocol_group':'协议组','family':'实验族','method_recorded':'记录方法','partition':'划分','seed':'记录seed','rank':'LoRA rank','precision':'精度类型','status':'本地状态','first_round':'首记录轮','last_round':'末记录轮','observed_rounds':'记录点数','window_start':'末20轮起点','window_end':'末20轮终点','last20_overall':'末20轮Overall','last20_head20':'末20轮Head20','last20_middle60':'末20轮Middle60','last20_tail':'末20轮Tail20','last20_nontail':'末20轮NonTail80','last20_many_gt100':'末20轮Many大于100','last20_medium20to100':'末20轮Medium20至100','last20_few_lt20':'末20轮Few小于20','final_overall':'最后记录Overall','final_tail':'最后记录Tail','tail_peak':'观测Tail峰值','tail_peak_round':'观测峰值轮','tail_peak_to_last':'观测峰值至最后回落pp','recomputation_status':'逐类复算状态','incomplete_per_class_files':'缺类逐类文件数','duplicate_curve_group':'相同曲线副本组','cross_seed_identical_curve':'跨seed同曲线组','normal_training':'正常训练','extra_training':'额外训练','loss':'损失','aggregation':'聚合','normal_optimizer_steps':'正常steps','extra_optimizer_steps':'额外steps','total_local_optimizer_steps':'本地总steps','functional_correction_steps':'功能修正steps','run_path':'原始路径','preferred_result_path':'推荐结果入口','curve_source':'曲线来源'}
CN_RUNS=[{cn:r.get(k,'') for k,cn in HEADERS.items()} for r in RUNS]
CN_FAMILIES=[{'实验族':f['family'],'整理建议':f['category'],'故事环节':f['story_slot'],'用途':f['use'],'已有结论':f['conclusion'],'建议行动':f['next_action'],'文件数':f['files'],'大小MB':f['size_mb'],'运行目录数':f['run_directories'],'记录到100轮目录数':f['complete100_directories'],'逐运行建议':f['run_categories'],'报告入口':f['reports']} for f in FAMILIES]
write_csv('全部实验_中文分类总表.csv',CN_RUNS)
write_csv('全部实验族_用途与去留.csv',CN_FAMILIES)
write_csv('跨seed相同曲线_需核对.csv',cross_rows)
write_csv('仅日志评测摘要.csv',LOG_ONLY)
write_csv('相同曲线与副本候选.csv',[{'group':r['duplicate_curve_group'],'run_id':r['run_id'],'path':r['run_path'],'preferred':r['preferred_result_path'],'scope':'Overall/Tail/Non-tail轨迹及记录seed/partition相同；推荐完整集中包，仅作索引选择，不删除。'} for r in RUNS if r['duplicate_curve_group']])

def get_one(family=None,method=None,partition=None,contains=None):
    candidates=[r for r in RUNS if (family is None or r['family']==family) and (method is None or r['method_recorded']==method) and (partition is None or r['partition']==partition) and (contains is None or contains in r['run_path'])]
    assert len(candidates)==1,(family,method,partition,contains,len(candidates))
    return candidates[0]

SELECT=[]
def pick(r,label,block,note=''):
    SELECT.append({'分组':block,'实验':label,'编号':r['run_id'],'协议组':r['protocol_group'],'划分':r['partition'],'seed':r['seed'],
    'Overall81_100':r['last20_overall'] if r['window_end']==100 else None,'Head20_81_100':r['last20_head20'] if r['window_end']==100 else None,
    'Middle60_81_100':r['last20_middle60'] if r['window_end']==100 else None,'Tail20_81_100':r['last20_tail'] if r['window_end']==100 else None,
    'Many81_100':r['last20_many_gt100'] if r['window_end']==100 else None,'Medium81_100':r['last20_medium20to100'] if r['window_end']==100 else None,'Few81_100':r['last20_few_lt20'] if r['window_end']==100 else None,
    'Tail峰值':r['tail_peak'],'峰值轮':r['tail_peak_round'],'Tail最后':r['final_tail'],'最后记录轮':r['last_round'],'峰值到最后回落pp':r['tail_peak_to_last'],'说明':note,'原始路径':r['run_path']})

for r in RUNS:
    if r['family']=='eri_seed42_share':pick(r,'FedAvg＋LoRA','1 开篇：固定边际双拓扑')
    if 'SupportNormalized' in r['family']:pick(r,r['aggregation'] or r['run_path'].split('/')[-1],'2 支持归一化：独立协议')
for part in ['client-longtail','matched-dirichlet']:
    for method in (['e0','e1','e2','e3','j','s','e5'] if part=='client-longtail' else ['e2','e3','j','s','e5']):
        pick(get_one('la_control_js_topology_analysis',method,part),method.upper(),'3 LA与A训练方式','E5为旧路线备查' if method=='e5' else '')
for method in ['e0','e1','e2','e5']:
    pick(get_one('la_control_dirichlet_supplement_analysis',method),'普通Dir '+method.upper(),'3 LA与A训练方式')
for method in ['e3','j','s']:
    pick(get_one('la_control_standard_dirichlet_e3_j_s_analysis',method),'普通Dir '+method.upper(),'3 LA与A训练方式')
for r in RUNS:
    if r['family'] in ['sfra_v1_analysis','sfra_cp_analysis','sfra_cp_analysis_new']:
        pick(r,r['method_recorded']+' '+r['run_path'].split('/')[-1],'4 A效果与消融','Current无CP不能替代Current-CP' if r['method_recorded']=='sfra_current' else '')
    if r['family'].startswith('sfra_b_'):
        pick(r,r['method_recorded']+' '+r['run_path'].split('/')[-1],'5 B辅助增强')
write_csv('主线实验精选_分协议.csv',SELECT)

GAPS=[
 {'优先级':'P0','项目':'同版Current-CP及五组正文结果','目前状态':'Full-CP、Flat-CP、Full无CP、S有seed42历史完整记录；未发现current-cp完整结果，未发现method_a_maintext新入口的五组完整回传。','为什么需要':'独立检验历史目标；不能用无CP的Current冒充。','最小行动':'先确认服务器结果；若确实缺失，只补缺失组。复用其他组前核对初始化、版本、划分、见证及执行路径。','入口':'scripts/run_method_a_maintext.py','原协议':'docs/method_a_maintext_experiments.md'},
 {'优先级':'P0','项目':'主方法关键配对的重复种子','目前状态':'核心SFRA主要为seed42；旧PromptFL/FedTEF不同seed不能替代。','为什么需要':'验证A来源/历史增益和小幅B增益是否稳定。','最小行动':'沿已定计划42/43/44优先完成S、Full-CP、关键消融；已有42先核对再复用。固定数据划分的训练seed验证不等于跨划分泛化。','入口':'scripts/run_method_a_maintext.py','原协议':'docs/method_a_maintext_experiments.md'},
 {'优先级':'P0（B若保留）','项目':'B的收益、代价与独立价值','目前状态':'共享B Tail+0.1525pp、Overall+0.0695pp；已有uniform8负对照；未发现shared_tradeoff三组本地结果。','为什么需要':'不足以声称B已显著改善长期保持；新tradeoff属于收益优化而非因果验证。','最小行动':'先回收w=0.5/0.35/0.2已计划结果；若稳定有效再补同预算普通更新/无定向选择对照，避免立即扩展更多模块。','入口':'scripts/run_cliplora_b_shared_tradeoff.py','原协议':'docs/cliplora_b_shared_tradeoff.md'},
 {'优先级':'P1','项目':'公平外部方法与普通Dir最终主表','目前状态':'普通Dir E0/E1/E2/E3/E5/J/S已有完整结果；同配置CAPT标准Dir尚未找到；SFRA主方法双拓扑完整结果未找到。','为什么需要':'旧matched Dir不可替代普通Dir；CAPT各版训练流程差异大。','最小行动':'纠正旧状态表；先定CAPT无测试反馈、起点/优化器/预算协议。是否补SFRA普通Dir按最终主表范围决定。','入口':'scripts/run_main_table_standard_dirichlet.py','原协议':'docs/main_table_standard_dirichlet_status_20260920.md'},
 {'优先级':'P1（机制若作强主张）','项目':'新A/B方法下的功能机制验证','目前状态':'ERI属于旧LoRA；SFRA有功能/来源日志，但旧归因不能直接归给新方法。','为什么需要':'证明方法改变了相关功能作用，而不只终值改善。','最小行动':'优先分析现有SFRA修正前后和历史目标日志；若缺必要状态再决定补采集。来源稀缺与支持总量分别控制。','入口':'scripts/run_cliplora_sfra.py','原协议':'docs/cliplora_method_a_full_cp.md'},
 {'优先级':'P2（可选）','项目':'固定边际ERI H3干预','目前状态':'ERI报告H3 NOT RUN；现有support-normalized普通Dir 2×2不是该H3。','为什么需要':'只有要声称严格因果机制时才需要补闭环。','最小行动':'保留当前机制为有边界证据；如坚持因果结论，再按ERI原协议补干预，不重开旧项目全套。','入口':'scripts/run_eri_closure.py','原协议':'docs/evidence_rewrite_imbalance_closure_experiment.md'},
 {'优先级':'不建议补','项目':'旧路线与大规模无目标参数扫描','目前状态':'PFRF、FedTEF、CUSP/SCA、gap-C3、private-B及语义形成链已留结果或失败门槛。','为什么需要':'不服务当前主方法的必要证据链。','最小行动':'保留原始和失败结论，停止追加训练；不因效果差而删除记录。','入口':'','原协议':''},
]
write_csv('缺口与补实验优先级.csv',GAPS)

# Inspect archive metrics in memory. Never unpack over original files.
ARCHIVES=[]
local_hash=defaultdict(list)
for r in RUNS:
    if r['metrics_sha256']:local_hash[r['metrics_sha256']].append(r['run_id'])
for a in DATA['archives']:
    p=ROOT/a['path']
    if p.suffix=='.zip':
        with zipfile.ZipFile(p) as z:members=[(n,z.read(n)) for n in z.namelist() if n.endswith('round_metrics.csv')]
    else:
        with tarfile.open(p,'r:*') as t:members=[(m.name,t.extractfile(m).read()) for m in t if m.isfile() and m.name.endswith('round_metrics.csv')]
    for name,b in members:
        rows=list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))));last=rows[-1] if rows else {}
        h=hashlib.sha256(b).hexdigest();matches=local_hash[h]
        ARCHIVES.append({'归档包':a['path'],'成员':name,'记录行数':len(rows),'最后epoch':last.get('epoch'),'最后round':last.get('round'),'最后Overall':last.get('overall_acc'),'最后Tail':last.get('bottom20_tail_acc'),'字节相同本地运行':';'.join(matches),'结论':'与已登记结果完全相同，不新增计数' if matches else '未找到字节相同本地曲线；此归档表单独保留，不推定新增独立实验'})
write_csv('归档包结果核对.csv',ARCHIVES)

# Code/plan coverage: index every script, doc and existing presentation artifact.
ASSETS=[]
for rootname in ('scripts','experiments','docs','presentation'):
    for p in sorted((ROOT/rootname).rglob('*')):
        if not p.is_file() or p.is_relative_to(OUT) or '__pycache__' in p.parts:continue
        if p.suffix.lower() not in ('.py','.md','.csv','.json','.html','.svg','.pdf','.png'):continue
        if rootname=='presentation':kind='已有汇总、图或审阅（派生产物，不新增训练）'
        elif p.suffix=='.py' and p.name.startswith('run_'):kind='实验入口（存在代码不代表已完成训练）'
        elif rootname=='docs':kind='协议或计划（状态需对照最新原始结果）'
        else:kind='分析/检查脚本或说明'
        ASSETS.append({'类型':kind,'路径':p.relative_to(ROOT).as_posix(),'字节':p.stat().st_size})
write_csv('实验代码_计划_历史图表索引.csv',ASSETS)

# Extract stated diagnostic decisions, preserving their exact source and scope.
DECISIONS=[]
for p in sorted((ROOT/'output').rglob('*.json')):
    if not any(w in p.name.lower() for w in ['summary','report','analysis','verdict','gate']):continue
    if p.name=='partition_summary.json':continue
    try:obj=json.loads(p.read_text(encoding='utf-8-sig'))
    except (ValueError,OSError):continue
    if not isinstance(obj,dict):continue
    for key,value in obj.items():
        if any(w in key.lower() for w in ['verdict','gate_pass','gate_checks','claim_boundary','evidence_boundary','recommend','conclusion']):
            DECISIONS.append({'来源':p.relative_to(ROOT).as_posix(),'字段':key,'记录内容':value})
write_csv('诊断实验_原始结论索引.csv',DECISIONS)

COUNTS=Counter(r['category'] for r in RUNS)
SUMMARY={'run_directory_counts':dict(COUNTS),'family_count':len(FAMILIES),'core_selected_rows':len(SELECT),'cross_seed_groups':len(set(r['组'] for r in cross_rows)),
 'cross_seed_directories':len(cross_rows),'incomplete_class_directories':sum(bool(r['incomplete_per_class_files']) for r in RUNS),'archive_metric_tables':len(ARCHIVES),
 'archive_unmatched_tables':sum(not r['字节相同本地运行'] for r in ARCHIVES),'assets':len(ASSETS),'diagnostic_decisions':len(DECISIONS)}
(OUT/'audit_summary.json').write_text(json.dumps(SUMMARY,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'classified_inventory.json').write_text(json.dumps({'runs':RUNS,'families':FAMILIES,'selected':SELECT,'gaps':GAPS,'summary':SUMMARY},ensure_ascii=False,indent=2),encoding='utf-8')

# Minimal standards-compliant XLSX writer; no additional package installation.
def excel_col(n):
    s=''
    while n:n,k=divmod(n-1,26);s=chr(65+k)+s
    return s
def make_xlsx(path,sheets):
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        types=['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>','<Default Extension="xml" ContentType="application/xml"/>','<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
        workbook=[];rels=[]
        for i,(name,rows) in enumerate(sheets,1):
            columns=list(dict.fromkeys(k for r in rows for k in r)); matrix=[columns]+[[r.get(k,'') for k in columns] for r in rows]
            end=f'{excel_col(len(columns))}{len(matrix)}';body=[]
            for ri,row in enumerate(matrix,1):
                cells=[]
                for ci,v in enumerate(row,1):
                    ref=excel_col(ci)+str(ri)
                    if isinstance(v,(int,float)) and not isinstance(v,bool):cells.append(f'<c r="{ref}"><v>{v}</v></c>')
                    else:
                        if isinstance(v,(list,dict)):v=json.dumps(v,ensure_ascii=False)
                        txt=escape('' if v is None else str(v))
                        cells.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{txt}</t></is></c>')
                body.append(f'<row r="{ri}">'+''.join(cells)+'</row>')
            xml=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="{ns}"><dimension ref="A1:{end}"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="18"/><cols><col min="1" max="{len(columns)}" width="22" customWidth="1"/></cols><sheetData>'+''.join(body)+f'</sheetData><autoFilter ref="A1:{end}"/></worksheet>'
            z.writestr(f'xl/worksheets/sheet{i}.xml',xml)
            types.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
            workbook.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
            rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'+''.join(types)+'</Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><bookViews><workbookView/></bookViews><sheets>'+''.join(workbook)+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(rels)+'</Relationships>')
make_xlsx(OUT/'实验总账.xlsx',[('阅读说明',[{'项目':'范围','说明':'263个本地运行/日志目录；包括副本、短程和zero-shot，不是263个独立完整实验。'},{'项目':'数字口径','说明':'准确率单位%；差值pp。正式主表81–100；末20轮起止在逐运行表明确列出。'},{'项目':'分类含义','说明':'必须保留=证据角色必需，不表示结论已证明；建议归档=停止投入或副本，不删除原始结果。'},{'项目':'空白值','说明':'本地缺失、稀疏窗口或不能可靠复算；绝不补零。'}]),('主线精选',SELECT),('所有运行',CN_RUNS),('所有实验族',CN_FAMILIES),('补实验优先级',GAPS),('跨seed同曲线',cross_rows),('归档包核对',ARCHIVES),('诊断结论',DECISIONS),('仅日志摘要',LOG_ONLY),('代码计划图表',ASSETS)])

def fmt(x,digits=3):return '—' if x is None or x=='' else f'{x:.{digits}f}' if isinstance(x,(int,float)) else str(x)
def href(path):return quote('../../'+path,safe='/')
def link(path,label):return f'<a href="{href(path)}">{html.escape(label)}</a>'
def tables_html(rows,columns):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(c,'')))+'</td>' for c in columns)+'</tr>' for r in rows)+'</tbody></table>'

cards=''.join(f'<div class="card"><strong>{COUNTS[c]}</strong><span>{c}的目录记录</span></div>' for c in [CORE,WAIT,ARCHIVE,INFRA])
rows_html=[]
for r in RUNS:
    details='<details><summary>'+html.escape(r['run_path'])+'</summary><p>'+html.escape(r['use'])+'</p><p>'+html.escape(r['conclusion'])+'</p><p>'+html.escape(r['next_action'])+'</p><p class="warn">'+html.escape(r['quality_notes'])+'</p>'+link(r['run_path'],'打开原始目录')+' · '+link(r['preferred_result_path'],'推荐核对入口')+'</details>'
    cells=[r['run_id'],r['category'],r['story_slot'],r['family'],r['method_recorded'],r['partition'],r['seed'],r['last_round'],f"{r['window_start']}–{r['window_end']}" if r['window_start'] else '无连续20轮',r['last20_overall'],r['last20_tail'],r['tail_peak_to_last'],r['status']]
    rows_html.append('<tr data-cat="'+r['category']+'" data-slot="'+r['story_slot']+'">'+''.join('<td>'+html.escape(fmt(x))+'</td>' for x in cells)+'<td>'+details+'</td></tr>')
selected_html=''
for group in dict.fromkeys(r['分组'] for r in SELECT):
    selected_html+='<h3>'+group+'</h3><div class="scroll">'+tables_html([r for r in SELECT if r['分组']==group],['实验','编号','划分','Overall81_100','Head20_81_100','Middle60_81_100','Tail20_81_100','Few81_100','峰值到最后回落pp'])+'</div>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>实验总账 · 2026-09-26</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#182433;font:15px/1.65 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1560px;margin:auto;padding:36px}h1{font-size:32px;margin-bottom:8px}h2{font-size:23px;margin-top:38px}h3{font-size:18px}p{max-width:1100px}a{color:#156f99}header{border-bottom:3px solid #176f80;padding-bottom:22px}.cards{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.card{background:white;border:1px solid #dce3e9;border-radius:10px;padding:18px 24px;min-width:205px}.card strong{font-size:32px;display:block}.card span{color:#516273}.note{background:#e7f1f5;border-left:4px solid #176f80;padding:16px 20px}.warn{color:#965118}.scroll{overflow:auto;max-height:670px;border:1px solid #dce3e9;border-radius:6px;background:white}table{border-collapse:collapse;width:100%;font-size:13px}th{position:sticky;top:0;background:#e9f0f4;text-align:left;white-space:nowrap;z-index:1}th,td{padding:10px 12px;border-bottom:1px solid #e5eaf0;vertical-align:top}td{min-width:90px}td:last-child{min-width:380px}tr:hover{background:#f1f7fa}details summary{cursor:pointer;overflow-wrap:anywhere}details p{max-width:550px}select,input,button{padding:10px;border:1px solid #b8c7d1;border-radius:5px;background:white;margin:4px}input{width:360px}nav{display:flex;gap:18px;flex-wrap:wrap}small{color:#536676}.filters{position:sticky;top:0;background:#f5f7fa;padding:10px 0;z-index:3}footer{margin-top:32px;color:#536676}@media(max-width:800px){main{padding:18px}input{width:100%}}</style>
<main><header><h1>围绕“尾类遗忘”的实验总账</h1><p>现象 → 固定边际对照 → 功能机制 → 适配与保持 → 主方法 A → 辅助 B。所有原始结果保留；只整理索引、用途与证据边界。</p><nav><a href="实验审计与取舍建议.md">完整审计说明</a><a href="实验总账.xlsx">下载 Excel 总账</a><a href="#selected">主线精选</a><a href="#all">全部运行</a><a href="#families">全部实验族</a><a href="#gaps">补实验优先级</a></nav></header>'''+cards+'''
<p class="note">扫描 34,403 份结果/辅助文件，登记 263 个运行或日志目录；183 个目录记录到第 100 轮。这里包含副本、旧协议和单点评测，不能当作独立完成实验数。相同 seed/划分的完整精度曲线有 39 组副本候选；另有 3 组跨 seed 同曲线记录待核对。</p>
<h2>先看这几个判断</h2><ol><li>开篇用原始 FedAvg＋LoRA：Client-LT Tail 从 68.80 降到 37.15；matched Dir 从 68.75 降到 49.40。</li><li>LA 下开放 A 有学习价值；不能写成“解冻就必然遗忘”。E2→E3 的 Overall/Tail 均提高。</li><li>Full-CP 对同频 S：Tail +2.2725 点、Overall −0.2770 点；当前是有收益和代价的保持改进。</li><li>Full-CP 对 Flat-CP：Tail +0.8800 点；Current-CP 仍缺本地完整结果。旧无CP Current不能替代。</li><li>共享 B 对 A-only：Tail +0.1525 点、Overall +0.0695 点；先验证收益稳定性，不能提前写成强贡献。</li><li>PFRF、FedTEF、CUSP/SCA、gap-C3和旧private-B归档；失败与负对照仍保留。</li></ol>
<h2 id="selected">主线精选：只在同一协议内作差</h2><p>所有准确率为百分数；Tail 回落为观测峰值到最后记录轮的百分点差。表内保留历史扫描供核对，不表示全部应进入正文。Head20/Middle60/Tail20 与 Many/Medium/Few 分组不同。</p>'''+selected_html+'''
<h2 id="all">全部运行：可搜索与筛选</h2><div class="filters"><input id="search" placeholder="搜索方法、路径、seed、用途或证据提醒"><select id="cat"><option value="">全部整理建议</option>'''+''.join('<option>'+c+'</option>' for c in [CORE,WAIT,ARCHIVE,INFRA])+'''</select><select id="slot"><option value="">全部故事环节</option>'''+''.join('<option>'+s+'</option>' for s in sorted({r['story_slot'] for r in RUNS}))+'''</select><button id="reset">重置</button><span id="counter"></span></div><div class="scroll"><table id="runs"><thead><tr>'''+''.join('<th>'+c+'</th>' for c in ['编号','建议','故事环节','实验族','方法','划分','seed','最后轮','统计窗口','Overall','Tail','Tail回落','状态','用途、结论与原始入口'])+'</tr></thead><tbody>'+''.join(rows_html)+'''</tbody></table></div>
<h2 id="families">全部实验族与零散资产</h2><p>91个索引项包含真正实验族、分析包、脚本、日程和零散文件；不是91个独立科研实验。逐运行的分类优先于族级建议。</p><div class="scroll">'''+tables_html(CN_FAMILIES,['实验族','整理建议','故事环节','用途','已有结论','建议行动','运行目录数'])+'''</div>
<h2 id="gaps">只补主线真正缺少的证据</h2><div class="scroll">'''+tables_html(GAPS,['优先级','项目','目前状态','为什么需要','最小行动','入口'])+'''</div>
<footer>生成于2026-09-26。没有训练、移动、删除或改写原始结果。逐类复算仅验证CSV统计一致性，不代表重新运行模型。完整文件索引、标准化曲线与可复现脚本在本目录。</footer></main>
<script>const search=document.querySelector('#search'),cat=document.querySelector('#cat'),slot=document.querySelector('#slot'),rs=[...document.querySelectorAll('#runs tbody tr')];function update(){let n=0;for(const r of rs){const ok=(!cat.value||r.dataset.cat===cat.value)&&(!slot.value||r.dataset.slot===slot.value)&&r.textContent.toLowerCase().includes(search.value.toLowerCase());r.hidden=!ok;if(ok)n++}document.querySelector('#counter').textContent=`显示 ${n} / ${rs.length} 个目录`}for(const e of [search,cat,slot])e.addEventListener('input',update);document.querySelector('#reset').onclick=()=>{search.value=cat.value=slot.value='';update()};update();</script></html>'''
(OUT/'实验总览.html').write_text(page,encoding='utf-8')
print(json.dumps(SUMMARY,ensure_ascii=False))
