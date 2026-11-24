"""CPU checks for file compatibility, depth units, and preservation of old defaults."""
from pathlib import Path
import argparse
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.full_scene import resolve_image_file,read_depth,effective_mask,full_scene_options,fit_full_image
from arguments import ModelParams


class FullSceneCompatibility(unittest.TestCase):
    def test_full_image_edges_survive_lora_resize(self):
        image=np.zeros((4,12,3),dtype=np.float32)
        image[:,:2,0]=1
        image[:,-2:,2]=1
        fitted=fit_full_image(image,12)
        self.assertTrue((fitted[:,:,0]>0).any())
        self.assertTrue((fitted[:,:,2]>0).any())
        np.testing.assert_allclose(fit_full_image(image*2-1,12,signed=True),fitted*2-1)

    def test_png_thumbnail_for_jpg_colmap_name(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            Image.new("RGB",(4,3),(11,22,33)).save(p/"Frame.PNG")
            self.assertEqual(Path(resolve_image_file(p,"frame.jpg")).name,"Frame.PNG")
            Image.new("RGB",(4,3)).save(p/"frame.jpeg")
            with self.assertRaises(FileNotFoundError):
                resolve_image_file(p,"frame.jpg")

    def test_float_depth_above_uint16_limit_and_legacy_png(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"depth.png"
            Image.fromarray(np.array([[1500,2500]],dtype=np.uint16)).save(p)
            np.testing.assert_allclose(read_depth(p),[[1.5,2.5]])
            np.save(p.with_suffix(".npy"),np.array([[128.0,304.0]],dtype=np.float32))
            np.testing.assert_allclose(read_depth(p),[[128.0,304.0]])

    def test_full_scene_mask_preserves_every_rgb_pixel(self):
        image=np.arange(36,dtype=np.uint8).reshape(3,4,3)
        mask=effective_mask((4,3),"/nonexistent/mask.png",True)/255.0
        preserved=image.copy()
        preserved[mask<.5]=255
        np.testing.assert_array_equal(preserved,image)
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"object.png"
            expected=np.array([[0,255],[255,0]],dtype=np.uint8)
            Image.fromarray(expected).save(p)
            np.testing.assert_array_equal(effective_mask((2,2),p,False),expected)

    def test_original_options_unchanged_without_opt_in(self):
        args=argparse.Namespace(full_scene=False,use_mask=True,white_background=True,
                                random_background=True,lambda_silhouette=.01)
        before=vars(args).copy()
        full_scene_options(args)
        self.assertEqual(vars(args),before)
        args.full_scene=True
        full_scene_options(args)
        self.assertFalse(args.use_mask)
        self.assertFalse(args.random_background)
        self.assertFalse(args.white_background)
        self.assertEqual(args.lambda_silhouette,0)

    def test_cli_default_is_object_mode(self):
        parser=argparse.ArgumentParser()
        ModelParams(parser)
        self.assertFalse(parser.parse_args([]).full_scene)
        self.assertTrue(parser.parse_args(["--full_scene"]).full_scene)


if __name__=="__main__":
    unittest.main()
