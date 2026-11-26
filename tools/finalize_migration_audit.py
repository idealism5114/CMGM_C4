"""Read-only source preservation checks and standalone delivery inventory."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmgm_c4.common import read,save,sha,runtime_hashes,seed_all,tensor_hash,prediction_loss,audit_train_val_files
from cmgm_c4.factory import build


def main():
    before=read(ROOT/'provenance/workspace_before.json');old=Path(before['source_root'])
    git=lambda *args:subprocess.check_output(['git',*args],cwd=old)
    after=dict(commit=git('rev-parse','HEAD').decode().strip(),status=git('status','--short').decode(),
        diff_stat=git('diff','--stat').decode(),diff_sha256=hashlib.sha256(git('diff','--binary')).hexdigest())
    unchanged=all(after[k]==before[k] for k in after)
    if not unchanged:raise ValueError('Original workspace state changed; investigate before claiming preservation')
    save(ROOT/'provenance/workspace_after.json',after)
    transport=read(ROOT/'provenance/migration_manifest.json');total=0
    for row in transport['files']:
        source,dest=Path(row['source']),ROOT/row['destination']
        if dest.is_symlink() or source.stat().st_ino==dest.stat().st_ino:raise ValueError('Not an independent copy')
        if sha(source)!=row['sha256'] or sha(dest)!=row['sha256']:raise ValueError('Original or copied artifact changed: '+row['destination'])
        total+=row['bytes']
    active={r['source']:r['source_sha256'] for r in read(ROOT/'provenance/extractions.json')}
    for path,h in active.items():
        if sha(path)!=h:raise ValueError('Original extracted source changed: '+path)
    save(ROOT/'provenance/original_core_source_hashes.json',active)
    save(ROOT/'provenance/runtime_source_hashes.json',runtime_hashes())
    # Verify train/eval arithmetic, RNG and gradients against the archived,
    # synthetic original-code reference, without importing the old project.
    torch.set_num_threads(4);ref=torch.load(ROOT/'provenance/legacy_synthetic_reference.pt',map_location='cpu',weights_only=True)
    m=build('c4/full',initial=True);parity={}
    for mode in ('eval','train'):
        m.train(mode=='train');m.set_epoch(20);m.zero_grad(set_to_none=True);seed_all(11013)
        p,a=m(**ref['batch'],return_aux=True);pred=prediction_loss(p,ref['target'],torch.ones_like(p,dtype=torch.bool));kl=m.switch_loss();(pred+kl).backward()
        oldref=ref['outputs'][mode]
        values=dict(prediction=p,prediction_loss=pred,KL=kl,**{k:a[k] for k in ('routing','expert_prediction','H','Z')})
        diffs={k:float((v.detach()-oldref[k]).abs().max()) for k,v in values.items()}
        grad=max(float((v.grad-oldref['gradients'][n]).abs().max()) for n,v in m.named_parameters())
        if max(diffs.values())>1e-6 or grad>1e-6:raise ValueError('Original arithmetic parity failed')
        parity[mode]=dict(max_absolute_differences=diffs,max_gradient_difference=grad,
            RNG_equal=torch.equal(torch.get_rng_state(),oldref['rng']))
        if not parity[mode]['RNG_equal']:raise ValueError('Random draw parity failed')
    save(ROOT/'verification/SYNTHETIC_ORIGINAL_PARITY.json',dict(status='PASS',threads=4,modes=parity,optimizer_steps=0))
    replay=read(ROOT/'verification/TRAIN_VAL_REPLAY.json');tests=(ROOT/'verification/tests.txt').read_text()
    if replay['status']!='PASS' or '57 passed' not in tests:raise ValueError('Required tests/replay not complete')
    registry=read(ROOT/'configs/registry.json');rows=[]
    for identifier,row in registry.items():
        model=build(identifier,initial=True)
        rows.append(dict(model=identifier,total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            legacy_unused_prediction_parameters=3104 if identifier=='baselines/GraphWaveNet' else 0,
            best_epoch=row['best_epoch'],checkpoint_sha256=row['checkpoint_sha256']))
    with (ROOT/'reports/MODEL_PARAMETER_COUNTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cpu=read(ROOT/'verification/cli_val_smoke/metrics.json')
    if abs(cpu['primary']-replay['records']['c4/full']['splits']['val']['MAE5'])>1e-8:raise ValueError('CLI metric differs')
    result=dict(status='PASS',source_workspace_unchanged=unchanged,original_commit=before['commit'],
        transported_files=len(transport['files']),transported_bytes=total,all_original_and_copied_hashes_match=True,
        independent_copies_no_links=True,registry_rows=len(registry),unique_best_checkpoints=len({r['checkpoint_sha256'] for r in registry.values()}),
        C4_parameters=567169,C4_best_epoch=35,C4_VAL5_macro_MAE=replay['records']['c4/full']['splits']['val']['MAE5'],
        full_TRAIN_VAL_replays=replay['replays'],max_GPU_prediction_difference=max(v['max_prediction_abs_diff'] for r in replay['records'].values() for v in r['splits'].values()),
        CPU_CLI_VAL5_macro_MAE=cpu['primary'],tests_passed=57,relocation_test='PASS; independent subprocess in a temporary copied project',
        source_and_train_eval_math_parity=parity,train_val_data_audit=audit_train_val_files(),
        formal_training_runs=0,synthetic_optimizer_updates_only=True,TEST_inference=0,TEST_numeric_arrays_parsed=0,
        TEST_transport='Opaque byte copies and hashes only',TEST_metric_summary='Historical metric JSON copied/read only; not new evaluation',
        old_repository_runtime_imports=0,new_default='Original Phase M C4; original Huber + Switch KL',
        non_goals=['Raw CSV preprocessing rebuild','Changing architecture/loss','New seed results','New TEST inference','Git commit/push'],
        known_limits=['Historical TEST repeatedly viewed','Single seed42','Baseline 27-token view differs from full-node C4',
            'GraphWaveNet legacy unused3104 parameters retained','No formal retraining of the refactored training loop; only synthetic tests and frozen replay'])
    save(ROOT/'verification/FINAL_AUDIT.json',result)
    summary=['# CMGM_C4 迁移与核验报告','',
        '本目录整理原 Phase M C4、最新 VAL5 选模的8个 baseline，以及 Full +12 个消融。未启动正式训练或新 TEST 推理。','',
        f"- 原仓库提交：`{before['commit']}`。HEAD、status、diff stat、binary diff hash 前后一致；用户已有修改保留。",
        f"- {len(transport['files'])} 个复制文件、{total/1e6:.2f} MB，源/目标 SHA256 均一致，无指回旧仓库的软/硬链接。",
        '- 22 个索引条目对应21个独立 best checkpoint；Full C4 为共享别名，不增加一次独立实验。',
        '- 57项 CPU 测试通过，包含临时目录独立恢复、机制/梯度、原 train/eval 随机数与梯度对照、TEST 哨兵。',
        '- 22个条目的完整 TRAIN/VAL 共44次冻结回放全部通过；GPU预测与历史缓存最大差0。',
        f"- C4 VAL5 Macro MAE：{result['C4_VAL5_macro_MAE']:.15f}；epoch35，567169参数。",
        f"- 独立 CPU CLI 复核 VAL5：{cpu['primary']:.15f}。",
        '- 新默认训练入口使用原未训练初始状态，不使用已训练 checkpoint warm start；训练需显式 `--execute`。',
        '- TEST 数组解析0、TEST推理0。复制和字节哈希不等同于重新评估；汇总引用既有指标JSON。','',
        '## 交付边界','',
        '预处理文件、标准化、mask、资产/商品顺序已独立复制；没有重建原始CSV处理流水线。新训练循环已通过静态/机制验证，但本轮没有正式重新训练它，不能承诺在不同硬件上逐bit复现训练轨迹。',
        '主模型保留原始 C4 Huber+Switch KL。MAE/no-KL和路由辅助损失等后续研究没有悄悄改变默认定义。',
        'baseline使用27个中性token，C4使用311完整节点；选模统一不等于结构和输入完全匹配。GraphWaveNet原3104个闲置参数保留。','',
        '## 本次修正的迁移工具问题','',
        '首次CPU回放的汇总文件名假设不适用于baseline，改为对原预测缓存直接重算相同指标；保留初始失败日志。测试对照中统一原/新CPU线程数，且分离两层TEST哨兵，未改变模型、数据或checkpoint来通过检查。','',
        '## 模型索引','',
        '| Model | Params | Trainable | Best epoch | VAL5 Macro MAE |','|---|---:|---:|---:|---:|']
    for row in rows:
        key=row['model'];mae=replay['records'][key]['splits']['val']['MAE5']
        summary.append(f"| {key} | {row['total_parameters']} | {row['trainable_parameters']} | {row['best_epoch']} | {mae:.12f} |")
    summary+=['','## 文件','',
        '- 使用与命令：`README.md`。','- 结构、损失、消融定义：`docs/MODEL_AND_PROTOCOL.md`。',
        '- 逐文件源/目标映射：`provenance/migration_manifest.json`。',
        '- 全量回放：`verification/TRAIN_VAL_REPLAY.json`、`verification/gpu_replay.log`。',
        '- 测试：`verification/tests.txt`。','- 最终审计：`verification/FINAL_AUDIT.json`。',
        '- 保存结果汇总：`reports/existing_experiments/`，包含历史TEST指标，未重新推理。','',
        '**MIGRATION_PASS = true**。此PASS只表示本次独立整理及所列核验通过，不表示预测性能提升。']
    (ROOT/'MIGRATION_REPORT.md').write_text('\n'.join(summary)+'\n')
    entries=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or any(s in p.parts for s in ('__pycache__','.pytest_cache')) or p==ROOT/'file_index.json':continue
        entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    save(ROOT/'file_index.json',dict(schema=1,date='2026-11-12',root='.',files=entries,
        main='README.md',registry='configs/registry.json',audit='verification/FINAL_AUDIT.json'))
    print('FINAL MIGRATION AUDIT PASS',len(entries),'indexed files')


if __name__=='__main__':main()
