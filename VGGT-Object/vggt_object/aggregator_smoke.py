"""Runtime smoke evidence. Called only by an explicit GPU training command."""
from pathlib import Path
import json
import statistics
import torch
from .common import json_dump,environment
from .model import create,predict
from .checkpoint import load_adapter
from .aggregator_lora import disabled


@torch.no_grad()
def finish(model,cfg,root,probe,updates):
    model.eval()
    image=probe.to('cuda')
    expected={k:v.detach().cpu() for k,v in predict(model,image).items()}
    with disabled(model):
        without=predict(model,image)
        difference=max(float((expected[k]-without[k].cpu()).abs().max()) for k in expected)
    if difference<=1e-8:raise AssertionError('Updated LoRA has no measurable inference effect')
    allocated=torch.cuda.max_memory_allocated()/1024**3
    reserved=torch.cuda.max_memory_reserved()/1024**3
    # Drop the completed model's storage without copying a 5GB model into host RAM.
    # Caller releases optimizer/parameter references before entering this function.
    model.to('meta');torch.cuda.empty_cache()
    fresh,_=create(cfg);load_adapter(root/'latest',fresh,cfg);fresh.eval()
    restored=predict(fresh,image)
    errors={k:float((v.cpu()-expected[k]).abs().max()) for k,v in restored.items()}
    for k,value in restored.items():
        if not torch.allclose(value.cpu(),expected[k],atol=1e-4,rtol=1e-3):
            raise AssertionError(f'Checkpoint prediction mismatch: {errors}')
    rows=[json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
    report=dict(status='passed',optimizer_steps=len(rows),rank=cfg['aggregator_lora']['rank'],
        resolution=cfg['resolution'],views=cfg['views'],environment=environment(cfg),
        gpu=torch.cuda.get_device_name(),torch_cuda_build=torch.version.cuda,
        initial_loss=rows[0]['total'],final_loss=rows[-1]['total'],
        mean_step_seconds=statistics.mean(r['step_seconds'] for r in rows),
        median_tail_step_seconds=statistics.median(r['step_seconds'] for r in rows[-10:]),
        peak_training_allocated_GiB=allocated,peak_training_reserved_GiB=reserved,
        lora_enabled_disabled_max_difference=difference,reload_max_errors=errors,
        checkpoint_output_consistent=True,oom=False,nonfinite_detected=False,
        update_evidence=updates,note='20-step engineering smoke test only; not accuracy evidence. GPU output reload tolerances atol=1e-4 rtol=1e-3.')
    json_dump(root/'memory_report.json',report);json_dump(root/'smoke_verification.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report
