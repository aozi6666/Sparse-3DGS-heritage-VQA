"""仅CPU保存本轮已完成检查及真实预算/字幕源审计，不运行推理或API。"""
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

PROJECT=Path('/root/autodl-tmp/Sparse-3DGS-heritage-VQA')
CODE=PROJECT/'VQA/mage_vl/practical'
sys.path.insert(0,str(CODE))
from practical_common import ROOT,sha,signature,write_once
from practical_vote import disk_guard
from subtitle_completion_common import RESULTS,context

assert subprocess.check_output(['git','-C',str(PROJECT),'branch','--show-current'],text=True).strip()=='vlm'
disk_guard(ROOT,10,1)
args=SimpleNamespace(scope='budget',min_free=10,budget=1)
ctx=context(args)
assert not ctx['units'] and not (RESULTS/'selected_short_config.json').exists()
assert not list((ROOT/'cache/subtitle_completion').glob('*/runtime.json'))
assert not list((ROOT/'cache/subtitle_completion').glob('*/units.jsonl'))

names=['subtitle_completion_common.py','subtitle_completion_run.py','subtitle_completion_analyze.py',
       'test_subtitle_completion.py','run_subtitle_completion.sh']
for name in names:
    if name.endswith('.py'): ast.parse((CODE/name).read_text())
subprocess.run(['bash','-n',str(CODE/'run_subtitle_completion.sh')],check=True)

# 对照既有归档逐文件核验；只读白名单结果和源码，不接触deepseek.env。
review=ROOT/'13_method_review/20261103_v1'
inventory=json.loads((review/'original_results_inventory.json').read_text())
for item in inventory['files']:
    path=PROJECT/item['path']
    assert path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
sources=json.loads((review/'source_manifest.json').read_text())
for name,item in sources.items():
    assert sha(PROJECT/name)==item['sha256'],name

per_video={}
old_total=new_total=0
for video in sorted(ctx['evidence']):
    previous=ctx['parent']['evidence'][video]; current=ctx['evidence'][video]['continuous']
    rows=[r for r in ctx['rows'] if r['video_id']==video]
    changed=previous['text']!=current['text']
    assert not current['omitted_neighbor_ids']
    assert all((r['index'] in ctx['changed'])==changed for r in rows)
    old_total+=len(rows)*previous['text_tokens']; new_total+=len(rows)*current['text_tokens']
    per_video[video]=dict(indices=[r['index'] for r in rows],changed=changed,
        old_tokens=previous['text_tokens'],new_tokens=current['text_tokens'],
        old_omitted_neighbor_ids=previous['omitted_neighbor_ids'],new_omitted_neighbor_ids=[],
        old_evidence_sha256=signature(previous),new_evidence_sha256=signature(current))
assert old_total==234552
assert sum(p['changed'] for p in per_video.values())==8
audit=dict(version='subtitle_budget_evidence_audit_v1',signature=ctx['signature'],
    questions=180,videos=60,changed_questions=24,changed_videos=8,unchanged_questions=156,
    old_logical_subtitle_tokens=old_total,new_logical_subtitle_tokens=new_total,
    additional_logical_subtitle_tokens=new_total-old_total,
    additional_text_percent=100*(new_total/old_total-1),per_video=per_video,
    actual_input_validation='Pending user-run: compare old2048/new3072 tensors and saved old input/visual identities before reuse.',
    new_inference_calls_executed_by_assistant=0)
write_once(RESULTS/'evidence_budget_audit_v1.json',audit)

short=json.loads((RESULTS/'short219_sources.json').read_text())
checks=dict(version='subtitle_completion_cpu_checks_v1',signature=ctx['signature'],
    unit_tests=dict(passed=133,new=22,existing_regression=111,
        command='python -B -m unittest test_subtitle_completion test_practical_next test_subtitle_holdout test_practical_subtitles test_practical_mage test_practical_coverage',
        provenance='Completed earlier in this implementation turn via remote Mage environment; recording its passed result without rerunning the suite.'),
    ast_parse=True,bash_syntax=True,budget_preflight=True,old_results_unchanged=len(inventory['files']),
    old_source_and_report_files_unchanged=len(sources),new_script_sha256={n:sha(CODE/n) for n in names},
    budget=dict(changed_questions=24,verified_reuse_pending=156),
    short=dict(questions=len(short['indices']),videos=len(short['sources']),
        source_status=dict(Counter(s['status'] for s in short['sources'].values())),configuration_frozen=False),
    documentation_sha256=sha(PROJECT/'docs/PLAN_subtitle_budget_short_completion.md'),
    assistant_inference_calls=0,assistant_api_calls=0,assistant_video_decodes=0,
    gpu_runtime_files_created=False,answer_journals_created=False)
assert not any(name in sys.modules for name in ['torch','transformers','decord'])
write_once(RESULTS/'cpu_checks_v1.json',checks)
print(json.dumps(dict(unit_tests_passed=133,preserved_result_files=len(inventory['files']),
    preserved_source_files=len(sources),subtitle_tokens=[old_total,new_total],
    new_text_percent=round(audit['additional_text_percent'],2),short_sources=checks['short'],
    checks=str(RESULTS/'cpu_checks_v1.json')),ensure_ascii=False))
