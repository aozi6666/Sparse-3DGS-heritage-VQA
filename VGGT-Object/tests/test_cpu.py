import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from torch import nn

from vggt_object.common import config
from vggt_object.model import LoRAConv2d,configure
from vggt_object.geometry import backproject,warp,multiview,normalize_scene,co3d_camera
from vggt_object.data import resize_frame,read_object
from vggt_object.losses import gradient_loss,objective
from vggt_object.metrics import depth_metrics,pose_metrics
from vggt_object.prepare import read_float,build_manifest,emit_object


def fake_model():
    m=nn.Module();m.aggregator=nn.Linear(3,3);m.camera_head=nn.Linear(3,3)
    m.depth_head=nn.Module();m.depth_head.projects=nn.ModuleList([nn.Conv2d(3,8,1)])
    m.depth_head.scratch=nn.Module();m.depth_head.scratch.fusion=nn.Conv2d(8,2,3,padding=1)
    return m


class LoRATests(unittest.TestCase):
    def test_zero_init_and_gradients_and_merge(self):
        torch.manual_seed(4)
        base=nn.Conv2d(3,4,3,padding=1,stride=2)
        original=base.weight.detach().clone();lora=LoRAConv2d(base,2,4,0.)
        x=torch.randn(2,3,12,12)
        self.assertTrue(torch.equal(base(x),lora(x)))
        opt=torch.optim.SGD([lora.lora_A,lora.lora_B],lr=.1)
        loss=lora(x).square().mean();loss.backward()
        self.assertGreater(float(lora.lora_B.grad.abs().sum()),0)
        opt.step();opt.zero_grad();lora(x).square().mean().backward()
        self.assertGreater(float(lora.lora_A.grad.abs().sum()),0)
        self.assertTrue(torch.equal(base.weight,original));self.assertIsNone(base.weight.grad)
        merged=copy.deepcopy(base)
        with torch.no_grad():
            merged.weight.add_((lora.lora_B[:,:,0,0]@lora.lora_A.flatten(1)).reshape_as(base.weight)*lora.scale)
        self.assertTrue(torch.allclose(merged(x),lora(x),atol=1e-6,rtol=1e-5))

    def test_only_depth_adapters_train(self):
        cfg=config(experiment='E2');model=fake_model();stats=configure(model,cfg)
        self.assertEqual(stats['aggregator_trainable'],0)
        self.assertEqual(len(stats['injected_modules']),2)
        self.assertTrue(all(n.startswith('depth_head.') and n.endswith(('.lora_A','.lora_B'))
                            for n,p in model.named_parameters() if p.requires_grad))

    def test_depth_full_keeps_aggregator_frozen(self):
        cfg=config(experiment='F0');model=fake_model();stats=configure(model,cfg)
        self.assertEqual(stats['aggregator_trainable'],0);self.assertEqual(stats['lora'],0)
        self.assertGreater(stats['depth_head'],0)
        self.assertTrue(all(p.requires_grad for p in model.depth_head.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.aggregator.parameters()))

    def test_full_head_checkpoint(self):
        from vggt_object.checkpoint import save,load_adapter
        cfg=config(experiment='F0');model=fake_model();configure(model,cfg)
        opt=torch.optim.AdamW(model.depth_head.parameters());sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda x:1.)
        class Sampler:rng=random.Random(42)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'best';save(folder,model,opt,sch,1,cfg,Sampler(),.2,'manifest',resume=False)
            other=fake_model();configure(other,cfg);load_adapter(folder,other,cfg)
            self.assertTrue((folder/'depth_head.safetensors').exists())
            self.assertFalse((folder/'depth_head_lora.safetensors').exists())
            self.assertTrue(all(torch.equal(v,other.depth_head.state_dict()[k]) for k,v in model.depth_head.state_dict().items()))

    def test_checkpoint_roundtrip_and_resume(self):
        from vggt_object.checkpoint import save,load_adapter,restore
        cfg=config(experiment='E2');model=fake_model();configure(model,cfg)
        for n,p in model.named_parameters():
            if n.endswith('lora_B'):
                with torch.no_grad():p.fill_(.12)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]);sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda x:1.)
        class Sampler:
            rng=random.Random(42)
        sampler=Sampler()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'latest'
            save(folder,model,opt,sch,7,cfg,sampler,.3,'test-manifest')
            other=copy.deepcopy(model)
            with torch.no_grad():
                for n,p in other.named_parameters():
                    if p.requires_grad:p.zero_()
            load_adapter(folder,other,cfg)
            for name,p in other.named_parameters():
                if p.requires_grad:self.assertTrue(torch.equal(p,dict(model.named_parameters())[name]))
            self.assertFalse((folder/'depth_head.safetensors').exists())
            self.assertFalse((folder/'aggregator_lora.safetensors').exists())
            self.assertEqual(restore(folder,model,opt,sch,cfg,sampler,'test-manifest'),(7,.3))
            x=torch.randn(1,3,8,8)
            def forward(m):
                return m.depth_head.scratch.fusion(m.depth_head.projects[0](x))
            model.eval();other.eval()
            self.assertTrue(torch.equal(forward(model),forward(other)))
            with self.assertRaises(ValueError):restore(folder,model,opt,sch,cfg,sampler,'changed-manifest')

    def test_real_dpt_forward_and_backward(self):
        # Exercise actual local VGGT DPT without a 1B-model forward or any GPU.
        from vggt_object.common import setup_vggt
        setup_vggt(config())
        from vggt.heads.dpt_head import DPTHead
        m=fake_model();m.depth_head=DPTHead(dim_in=2048,output_dim=2,activation='exp',conf_activation='expp1')
        configure(m,config(experiment='E2'));m.eval()
        tokens=[torch.randn(1,2,9,2048) if i in (4,11,17,23) else None for i in range(24)]
        d,c=m.depth_head(tokens,torch.rand(1,2,3,28,28),5,frames_chunk_size=1)
        self.assertEqual(d.shape,(1,2,28,28,1));self.assertEqual(c.shape,(1,2,28,28))
        (d.mean()+c.mean()).backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0
                            for n,p in m.depth_head.named_parameters() if n.endswith('lora_B')))

    def test_real_checkpoint_cpu_load_without_forward(self):
        from vggt_object.model import create
        model,stats=create(config(),device='cpu')
        self.assertEqual(stats['aggregator_trainable'],0)
        self.assertFalse(any(t.is_meta for t in list(model.parameters())+list(model.buffers())))
        self.assertTrue(all(p.device.type=='cpu' for p in model.parameters()))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.K=torch.tensor([[20.,0,7.5],[0,20.,7.5],[0,0,1.]])
        self.E=torch.eye(4);self.d=torch.ones(16,16)*2

    def test_identity_backprojection_and_warp(self):
        points=backproject(self.d,self.K,self.E)
        self.assertTrue(torch.allclose(points[...,2],self.d))
        grid,z,mask=warp(self.d,self.K,self.E,self.K,self.E)
        self.assertEqual(int(mask.sum()),256);self.assertTrue(torch.allclose(z,self.d))

    def test_multiview_grad_and_occlusion(self):
        gt=torch.stack([self.d,self.d]);pred=(gt+.01).clone().requires_grad_()
        E=torch.stack([self.E,self.E]);K=torch.stack([self.K,self.K]);mask=torch.ones_like(gt,dtype=torch.bool)
        loss,n=multiview(pred,gt,K,E,mask,stride=1)
        self.assertGreater(n,0);self.assertAlmostEqual(float(loss),0.,places=5)
        loss.backward();self.assertTrue(torch.isfinite(pred.grad).all())
        occluded=gt.clone();occluded[1]=1
        _,n=multiview(pred.detach(),occluded,K,E,mask,stride=1)
        self.assertEqual(n,0)
        outside=E.clone();outside[1,0,3]=1000
        _,n=multiview(pred.detach(),gt,K,outside,mask,stride=1)
        self.assertEqual(n,0)

    def test_bad_prediction_cannot_hide_from_multiview(self):
        gt=torch.stack([self.d,self.d]);pred=gt.clone();pred[0]*=3;pred.requires_grad_()
        K=torch.stack([self.K,self.K]);E=torch.stack([self.E,self.E]);mask=torch.ones_like(gt,dtype=torch.bool)
        loss,n=multiview(pred,gt,K,E,mask,stride=1)
        self.assertGreater(n,0);self.assertGreater(float(loss),0)
        loss.backward();self.assertGreater(float(pred.grad.abs().sum()),0)

    def test_validation_subsets_fixed_and_distinct(self):
        from vggt_object.evaluate import validation_cases
        o={'id':'independent_val','frames':24}
        a=validation_cases(o,4,42,6);b=validation_cases(o,4,42,6)
        self.assertEqual(a,b);self.assertEqual(len(set(tuple(x) for x in a)),6)

    def test_nested_469_split_has_shared_heldout_views(self):
        from vggt_object.downstream import nested_camera_split
        frames=[]
        for angle in np.linspace(0,2*np.pi,24,endpoint=False):
            e=np.eye(4);e[:3,3]=[np.cos(angle),np.sin(angle),0]
            frames.append({'E':e.tolist()})
        four,test=nested_camera_split(frames,4);six,test6=nested_camera_split(frames,6);nine,test9=nested_camera_split(frames,9)
        self.assertTrue(set(four)<=set(six)<=set(nine));self.assertEqual(test,test6);self.assertEqual(test,test9)
        self.assertFalse(set(nine)&set(test))

    def test_shared_normalization(self):
        d=torch.stack([self.d,self.d*2]);K=torch.stack([self.K,self.K]);E=torch.stack([self.E,self.E])
        norm,e,scale=normalize_scene(d,K,E,torch.ones_like(d,dtype=torch.bool))
        self.assertTrue(torch.allclose(norm[1],norm[0]*2));self.assertTrue(torch.allclose(e[0],torch.eye(4)))
        self.assertGreater(float(scale),0)

    def test_masked_gradient(self):
        p=torch.ones(1,1,8,8,requires_grad=True);g=p.detach().clone();g[...,4:]=100
        mask=torch.ones_like(g,dtype=torch.bool);mask[...,4:]=False
        self.assertEqual(float(gradient_loss(p,g,mask)),0)

    def test_co3d_ndc_conversion(self):
        view=dict(R=np.eye(3).tolist(),T=[1.,2.,3.],focal_length=[2.,2.],principal_point=[.1,.2])
        K,E=co3d_camera(view,(100,200))
        self.assertTrue(np.allclose(K,[[200,0,90],[0,100,40],[0,0,1]]))
        self.assertTrue(np.allclose(E[:3,3],[-1,-2,3]))
        self.assertAlmostEqual(np.linalg.det(E[:3,:3]),1.)

    def test_loss_finite_and_backward(self):
        gt=torch.stack([self.d,self.d])[None];pred=(gt+.01).clone().requires_grad_();conf=torch.ones_like(gt,requires_grad=True)
        batch=dict(depth=gt,valid=torch.ones_like(gt,dtype=torch.bool),K=torch.stack([self.K,self.K])[None],
                   E=torch.stack([self.E,self.E])[None],supervision_weight=1.)
        loss,terms=objective(dict(depth=pred,confidence=conf),batch,config(),1000)
        loss.backward();self.assertTrue(torch.isfinite(pred.grad).all());self.assertGreater(float(pred.grad.abs().sum()),0)
        batch['valid'].fill_(False)
        with self.assertRaises(ValueError):objective(dict(depth=pred,confidence=conf),batch,config(),1000)

    def test_camera_scale_without_gt_depth(self):
        from vggt_object.downstream import camera_scale
        E=torch.eye(4).repeat(4,1,1)
        E[:,:3,3]=torch.tensor([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
        known=E.clone();known[:,:3,3]*=3
        scale,rmse=camera_scale(E,known)
        self.assertAlmostEqual(float(scale),3.,places=5);self.assertLess(rmse,1e-5)

    def test_colmap_export_api(self):
        import pycolmap
        from vggt_object.downstream import write_colmap
        with tempfile.TemporaryDirectory() as tmp:
            write_colmap(Path(tmp),[('test.png',self.K.numpy(),np.eye(4))],16)
            self.assertEqual(len(pycolmap.Reconstruction(tmp).images),1)


class DataTests(unittest.TestCase):
    def test_float16_png(self):
        a=np.array([[1.,2.],[.25,8.]],dtype=np.float16)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'depth.png';Image.fromarray(a.view(np.uint16)).save(p)
            self.assertTrue(np.array_equal(read_float(p,packed=True),a.astype(np.float32)))

    def test_omni_offline_converter(self):
        from vggt_object.prepare import omni
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'raw/category/object/render';(root/'images').mkdir(parents=True);(root/'depths').mkdir()
            frames=[]
            for i in range(8):
                Image.new('RGBA',(32,32),(120,100,80,255)).save(root/'images'/f'{i}.png')
                np.save(root/'depths'/f'{i}.npy',np.ones((32,32),np.float32)*2)
                frames.append(dict(file_path=f'./images/{i}',transform_matrix=np.eye(4).tolist()))
            (root/'transforms.json').write_text(json.dumps(dict(camera_angle_x=.8,frames=frames)))
            with self.assertRaises(ValueError):omni(Path(tmp)/'raw',Path(tmp)/'bad',None,1,8,32)
            omni(Path(tmp)/'raw',Path(tmp)/'out','camera_z_depth',1,8,32)
            output=list((Path(tmp)/'out').rglob('cameras.json'))
            self.assertEqual(len(output),1)
            self.assertTrue(np.allclose(json.loads(output[0].read_text())['frames'][0]['E'],np.diag([1,-1,-1,1])))

    def test_co3d_offline_converter_and_review_guard(self):
        import gzip
        from vggt_object.prepare import co3d
        from vggt_object.data import SceneSampler
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);raw=tmp/'raw';seq=raw/'cup/seq'
            for name in ('images','depths','masks','depth_masks'):(seq/name).mkdir(parents=True)
            rows=[]
            for i in range(8):
                Image.new('RGB',(32,32),(90,60,30)).save(seq/'images'/f'{i}.jpg')
                Image.fromarray(np.full((16,16),2.,np.float16).view(np.uint16)).save(seq/'depths'/f'{i}.png')
                Image.new('L',(32,32),255).save(seq/'masks'/f'{i}.png')
                Image.new('L',(16,16),255).save(seq/'depth_masks'/f'{i}.png')
                rows.append(dict(sequence_name='seq',frame_number=i,image=dict(path=f'cup/seq/images/{i}.jpg',size=[32,32]),
                                 depth=dict(path=f'cup/seq/depths/{i}.png',scale_adjustment=1.5,mask_path=f'cup/seq/depth_masks/{i}.png'),
                                 mask=dict(path=f'cup/seq/masks/{i}.png'),viewpoint=dict(R=np.eye(3).tolist(),T=[0,0,0],
                                 focal_length=[2.,2.],principal_point=[0.,0.])))
            with gzip.open(raw/'cup/frame_annotations.jgz','wt') as f:json.dump(rows,f)
            splits=tmp/'splits.json';splits.write_text(json.dumps({'cup/seq':'train'}))
            with self.assertRaises(ValueError):co3d(raw,tmp/'bad',1,8,32)
            co3d(raw,tmp/'processed',1,8,32,False,splits)
            meta=json.loads(next((tmp/'processed').rglob('cameras.json')).read_text())
            self.assertEqual(meta['provenance']['official_split'],'train');self.assertFalse(meta['depth_reviewed'])
            dep=np.load(tmp/'processed/co3dv2/cup__seq/depths/00000.npy')
            self.assertTrue(np.allclose(dep,3.))
            manifest=tmp/'manifest.json';build_manifest(tmp/'processed',manifest)
            cfg=config();cfg['mix']={'co3dv2':1.}
            with self.assertRaises(ValueError):SceneSampler(manifest,cfg)

    def test_resize_updates_K_and_padding_mask(self):
        im=Image.new('RGB',(40,20));d=np.ones((20,40),np.float32);m=np.ones((20,40),bool)
        K=torch.tensor([[10.,0,20],[0,10.,10],[0,0,1.]])
        rgb,dep,mask,k=resize_frame(im,d,m,K,28)
        self.assertEqual(rgb.shape,(3,28,28));self.assertEqual(float(k[0,0]),7.)
        self.assertEqual(float(k[1,2]),14.);self.assertFalse(mask[:7].any())

    def test_prepared_reader_and_split_reproducibility(self):
        def frame():
            return Image.new('RGB',(32,32),(120,80,40)),np.ones((32,32),np.float32)*2,np.ones((32,32),bool),np.array([[20.,0,16],[0,20.,16],[0,0,1.]]),np.eye(4)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);obj=tmp/'processed/omniobject3d/object_1'
            emit_object(obj,'object_1','omniobject3d',[frame]*8,True,32,{'source':'synthetic'})
            manifest=tmp/'manifest.json';build_manifest(tmp/'processed',manifest)
            first=manifest.read_bytes();build_manifest(tmp/'processed',manifest)
            self.assertEqual(first,manifest.read_bytes())
            o=json.loads(manifest.read_text())['objects'][0]
            b=read_object(o,2,28,random.Random(42))
            self.assertEqual(b['images'].shape,(1,2,3,28,28));self.assertTrue(torch.isfinite(b['depth']).all())
            from vggt_object.data import SceneSampler
            cfg=config();cfg.update(mix={'omniobject3d':1.},views=2,resolution=28,fixed_views=True,rgb_augmentation=True)
            augmented=SceneSampler(manifest,cfg).next()
            clean=SceneSampler(manifest,{**cfg,'rgb_augmentation':False}).next()
            for key in ('depth','K','E','valid'):self.assertTrue(torch.equal(augmented[key],clean[key]))
            self.assertFalse(torch.equal(augmented['images'],clean['images']))

    def test_metrics(self):
        d=torch.ones(2,8,8);metrics=depth_metrics(d,d,d.bool())
        self.assertEqual(metrics['AbsRel'],0);self.assertEqual(metrics['delta1'],1)
        E=torch.eye(4).repeat(2,1,1);E[1,0,3]=1
        result=pose_metrics(E,E)
        self.assertEqual(result['rotation_mean_deg'],0);self.assertAlmostEqual(result['AUC30'],1.)


if __name__=='__main__':
    unittest.main()
