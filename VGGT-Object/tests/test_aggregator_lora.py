import copy
from pathlib import Path
import random
import tempfile
import unittest
import torch
from torch import nn
from vggt_object.common import config,setup_vggt
from vggt_object.model import create,configure,train_mode
from vggt_object.aggregator_lora import LoRALinear,disabled,parameter_groups,UpdateInspector
from vggt_object.checkpoint import save,load_adapter,restore


def settings(checkpoint=True):
    cfg=config(experiment='A16',phase='agg_smoke')
    cfg=copy.deepcopy(cfg);cfg['aggregator_lora']['verify_vggt_dimensions']=False
    cfg['aggregator_lora']['dropout']=0.;cfg['aggregator_lora']['gradient_checkpointing']=checkpoint
    return cfg


def tiny_model():
    setup_vggt(config())
    from vggt.layers.block import Block
    class TinyAggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.frame_blocks=nn.ModuleList([Block(dim=16,num_heads=4,mlp_ratio=2.) for _ in range(24)])
            self.global_blocks=nn.ModuleList([Block(dim=16,num_heads=4,mlp_ratio=2.) for _ in range(24)])
        def forward(self,x):
            for frame,glob in zip(self.frame_blocks,self.global_blocks):x=glob(frame(x))
            return x
    m=nn.Module();m.aggregator=TinyAggregator();m.depth_head=nn.Linear(16,2)
    m.camera_head=nn.Linear(16,3);m.point_head=None;m.track_head=None
    return m


class AggregatorTests(unittest.TestCase):
    def test_actual_vggt_rank16_and_rank8_structure(self):
        for rank,count in [(16,589824),(8,294912)]:
            m,stats=create(config(experiment='A16' if rank==16 else 'A8',phase='agg_smoke'),meta_only=True)
            self.assertEqual(stats['aggregator_lora_parameters'],count)
            self.assertEqual(stats['depth_head'],32654562)
            self.assertEqual(stats['trainable'],32654562+count)
            self.assertEqual(len(stats['aggregator_lora_targets']),12)
            for row in stats['aggregator_lora_targets']:
                self.assertEqual(row['in_features'],1024)
                self.assertEqual(row['out_features'],3072 if row['name'].endswith('qkv') else 1024)
            self.assertTrue(all(not p.requires_grad for p in m.camera_head.parameters()))

    def test_zero_initialization_and_reject_wrong_target(self):
        base=nn.Linear(16,48);layer=LoRALinear(base,16,32,.05)
        x=torch.randn(2,3,16);self.assertTrue(torch.equal(layer(x),base(x)))
        cfg=settings();cfg['aggregator_lora']['target_blocks']=[17,18,19,20,21,22]
        with self.assertRaises(ValueError):configure(tiny_model(),cfg)

    def test_real_block_backward_through_frozen_frames_and_checkpoint(self):
        torch.manual_seed(8);m=tiny_model();cfg=settings();frozen=copy.deepcopy(m.state_dict())
        configure(m,cfg);train_mode(m)
        opt=torch.optim.AdamW(parameter_groups(m,cfg),weight_decay=.01)
        x=torch.randn(1,5,16);inspector=UpdateInspector(m,opt)
        for step in range(2):
            opt.zero_grad(set_to_none=True);y=m.depth_head(m.aggregator(x))
            y.square().mean().backward()
            first=m.aggregator.global_blocks[18].attn.qkv
            self.assertGreater(float(first.lora_B.grad.abs().sum()),0)
            if step==1:self.assertGreater(float(first.lora_A.grad.abs().sum()),0)
            self.assertGreater(float(m.depth_head.weight.grad.abs().sum()),0)
            opt.step()
            if step==0:self.assertTrue(inspector.verify()['lora_updated'])
        for name,p in m.named_parameters():
            if not p.requires_grad:
                self.assertIsNone(p.grad)
                self.assertTrue(torch.equal(p,frozen[name.replace('.base.','.')]))
        m.eval()
        with torch.no_grad():
            active=m.depth_head(m.aggregator(x))
            with disabled(m):inactive=m.depth_head(m.aggregator(x))
        self.assertFalse(torch.equal(active,inactive))

    def test_checkpoint_and_optimizer_restore(self):
        torch.manual_seed(20);m=tiny_model();base=copy.deepcopy(m.state_dict());cfg=settings()
        configure(m,cfg);train_mode(m)
        opt=torch.optim.AdamW(parameter_groups(m,cfg));sch=torch.optim.lr_scheduler.LambdaLR(opt,[lambda x:1.,lambda x:1.])
        x=torch.randn(1,5,16);m.depth_head(m.aggregator(x)).square().mean().backward();opt.step()
        class Sampler:rng=random.Random(42)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'latest';save(folder,m,opt,sch,1,cfg,Sampler(),.1,'manifest')
            self.assertTrue((folder/'aggregator_lora.safetensors').exists())
            self.assertTrue((folder/'depth_head.safetensors').exists())
            self.assertFalse((folder/'depth_head_lora.safetensors').exists())
            other=tiny_model();other.load_state_dict(base);configure(other,cfg);load_adapter(folder,other,cfg)
            m.eval();other.eval()
            with torch.no_grad():self.assertTrue(torch.allclose(m.depth_head(m.aggregator(x)),other.depth_head(other.aggregator(x)),atol=1e-6))
            opt2=torch.optim.AdamW(parameter_groups(other,cfg));sch2=torch.optim.lr_scheduler.LambdaLR(opt2,[lambda x:1.,lambda x:1.])
            self.assertEqual(restore(folder,other,opt2,sch2,cfg,Sampler(),'manifest'),(1,.1))
            self.assertEqual(len(opt2.param_groups),2)

    def test_checkpointing_preserves_forward_and_gradients(self):
        torch.manual_seed(4);a=tiny_model();state=copy.deepcopy(a.state_dict())
        b=tiny_model();b.load_state_dict(state)
        configure(a,settings(True));configure(b,settings(False))
        # Match adapter initialization too, then make B nonzero to test all paths.
        b.load_state_dict(a.state_dict())
        for m in (a,b):
            with torch.no_grad():
                for name,p in m.named_parameters():
                    if name.endswith('lora_B'):p.fill_(.001)
            train_mode(m)
        x=torch.randn(1,5,16)
        ya=a.depth_head(a.aggregator(x));yb=b.depth_head(b.aggregator(x))
        self.assertTrue(torch.allclose(ya,yb,atol=1e-6));ya.square().mean().backward();yb.square().mean().backward()
        for (n,p),(k,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.requires_grad:self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-4),n)


if __name__=='__main__':unittest.main()
