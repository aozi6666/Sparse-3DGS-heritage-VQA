import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from vggt_object.bundle import remap,upload_status,import_bundle,read_catalog
from vggt_object.data import read_object


def make_bundle(root):
    bundle=root/'data/VGGT_finetune_v1';(bundle/'manifests').mkdir(parents=True)
    rows=[]
    for idx,split in enumerate(('train','val','test')):
        identifier=f'object_{idx}';obj=bundle/'smithsonian'/identifier
        for folder in ('images','depth','masks'):(obj/folder).mkdir(parents=True)
        views=[]
        for i in range(6):
            Image.new('RGB',(32,32),(90,100,110)).save(obj/'images'/f'{i}.png')
            np.save(obj/'depth'/f'{i}.npy',np.full((32,32),2,np.float32))
            Image.new('L',(32,32),255).save(obj/'masks'/f'{i}.png')
            views.append(dict(image=f'images/{i}.png',depth=f'depth/{i}.npy',mask=f'masks/{i}.png',
                              fx=20,fy=20,cx=16,cy=16,c2w=np.eye(4).tolist()))
        (obj/'cameras.json').write_text(json.dumps(dict(convention='OpenCV c2w (x right, y down, z forward)',views=views)))
        rows.append(dict(id=identifier,dataset='smithsonian',split=split,root=f'smithsonian/{identifier}',n_views=6))
    (bundle/'manifests/all_objects.json').write_text(json.dumps({'objects':rows}))
    for row in rows:
        (bundle/f'manifests/{row["split"]}.json').write_text(json.dumps({'ids':[row['id']]}))
    return bundle


class BundleTests(unittest.TestCase):
    def test_remap_legacy_mac_root(self):
        result=remap(dict(dataset='omni3d',id='vase_012',root='/Users/old/mac/omni3d/vase_012'),Path('/bundle'),Path('/remote/omni3d'))
        self.assertEqual(str(result),'/remote/omni3d/vase_012')

    def test_upload_snapshot_detects_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);omni=root/'omni';omni.mkdir()
            (omni/'placeholder.jpg').touch();(omni/'ready.json').write_text('{}')
            result=upload_status(omni,root/'missing')
            self.assertEqual(result['sources']['omni3d']['zero_bytes'],1)
            self.assertEqual(result['sources']['omni3d']['bytes'],2)

    def test_zero_copy_index_and_preserved_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=make_bundle(root);before=(bundle/'manifests/all_objects.json').read_bytes()
            manifest=root/'data/manifests/objects.json'
            with patch('vggt_object.bundle.PROJECT',root):
                result=import_bundle(bundle,root/'omni',manifest)
            self.assertEqual(result['copied_media_bytes'],0)
            self.assertEqual(result['split_counts'],dict(train=1,val=1,test=1))
            self.assertEqual(before,(bundle/'manifests/all_objects.json').read_bytes())
            objects=json.loads(manifest.read_text())['objects']
            batch=read_object(objects[0],2,28,random.Random(42))
            self.assertEqual(tuple(batch['images'].shape),(1,2,3,28,28))
            self.assertEqual(len(list((root/'data/processed').rglob('*.npy'))),0)

    def test_empty_upload_blocks_training_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=make_bundle(root)
            (bundle/'smithsonian/object_0/images/0.png').write_bytes(b'')
            manifest=root/'data/manifests/objects.json'
            with patch('vggt_object.bundle.PROJECT',root),self.assertRaisesRegex(ValueError,'zero-byte'):
                import_bundle(bundle,root/'omni',manifest)
            self.assertFalse(manifest.exists())

    def test_source_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=make_bundle(root)
            (bundle/'manifests/val.json').write_text(json.dumps({'ids':['object_0']}))
            with self.assertRaisesRegex(ValueError,'overlap'):read_catalog(bundle,root/'omni')


if __name__=='__main__':unittest.main()
