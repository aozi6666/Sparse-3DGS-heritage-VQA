"""只做CPU控制／像素处理检查；不加载真实模型或解码视频。"""
from copy import deepcopy
import importlib.util
import math
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import air_mage_adapter as a


# 给透明包装器提供已加载原生模块的模拟接口；Torch只在CPU测试时导入。
def smart_resize(height, width, patch_size=16, min_pixels=None, max_pixels=None, align_patch_size=None):
    return 4, 8


def format_timestamp(seconds):
    minute = int(seconds // 60)
    return f'{minute:02d}:{seconds - minute * 60:09.6f}'


def time_str_to_seconds(value):
    minute, sec = value.split(':')
    return int(minute) * 60 + float(sec)


def build_patch_positions(grid_thw, spatial_merge_size=2, frame_indices=None):
    import torch
    result = []
    for i, row in enumerate(grid_thw.tolist()):
        t, h, w = row
        ids = list(range(t)) if frame_indices is None else frame_indices[i].tolist()
        # 2x2网格时块布局与行主序相同；真正原生布局由用户GPU回归逐tensor检查。
        result.extend([[idx, y, x] for idx in ids for y in range(h) for x in range(w)])
    return torch.tensor(result, dtype=torch.int64)


class FakeBase:
    def __init__(self):
        self.fixed_num_frames = 32
        self.max_frames = 384
        self.target_fps = None
        self.resize_frames = False
        self.min_pixels = 3136
        self.max_pixels = 150000
        self.patch_size = 16
        self.spatial_merge_size = 2
        self.calls = 0

    def __call__(self, videos, return_tensors='pt', **kwargs):
        import numpy as np
        import torch
        frames = videos[0]
        self.calls += 1
        values = torch.tensor(np.stack([np.asarray(f) for f in frames]).reshape(-1, 3).copy())
        grid = torch.tensor([[len(frames), 2, 2]], dtype=torch.int64)
        return dict(pixel_values_videos=values, video_grid_thw=grid,
                    patch_positions=build_patch_positions(grid),
                    frame_timestamps=[[float(i) for i in range(len(frames))]])


class Validation(unittest.TestCase):
    def test_valid_nonuniform_original_ids(self):
        self.assertEqual(a.validate_selection([None]*3, [0, 70, 299], 29.97, 300), (0, 70, 299))

    def test_invalid_identity_and_order(self):
        for ids in [[], [0, 0], [2, 1], [-1, 0], [0, 300], [False, 1], [0.0, 1]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                a.validate_selection([None]*len(ids), ids, 30., 300)

    def test_invalid_fps_count_and_alignment(self):
        for fps in [0, -1, float('nan'), float('inf'), None, True]:
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                a.validate_selection([None], [0], fps, 300)
        for count in [0, -1, True, 3.5]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                a.validate_selection([None], [0], 30., count)
        with self.assertRaises(ValueError):
            a.validate_selection([None], [0, 1], 30., 300)

    def test_import_does_not_load_runtime(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            "import air_mage_adapter,sys; assert not any(n in sys.modules for n in ['torch','transformers','decord','numpy'])"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_native_processor_refused(self):
        with self.assertRaises(TypeError):
            a.SelectedFramesVideoProcessor(SimpleNamespace(), [None], [0], 30., 300)

    def test_image_prompt_preserved(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.message = messages
                return 'CHAT:' + messages[0]['content'][1]['text']
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return kwargs
        p = Processor()
        query = 'Question\nA. a\nB. b\nC. c\nD. d'
        result = a.image_inputs(p, 'image', query)
        self.assertTrue(result['text'][0].endswith('D. d'))
        self.assertEqual(result['images'], ['image'])
        self.assertNotIn('videos', result)
        self.assertEqual(p.message[0]['content'][0]['type'], 'image')
        with self.assertRaises(ValueError):
            a.image_inputs(p, 'image', '')


TORCH = importlib.util.find_spec('torch') is not None
PIL = importlib.util.find_spec('PIL') is not None
NUMPY = importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(TORCH and PIL and NUMPY, 'requires existing CPU Torch/Pillow/NumPy')
class RuntimeCPU(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.frames = [np.full((2, 2, 3), n, dtype=np.uint8) for n in [10, 20, 30]]
        self.base = FakeBase()
        self.processor = SimpleNamespace(video_processor=self.base)

    def test_original_time_axis_and_unchanged_pixels(self):
        import torch
        with a.selected_video_inputs(self.processor, self.frames, [0, 70, 299], 29.97, 300, 'v.mp4') as wrapper:
            result = wrapper(videos=['v.mp4'])
            self.assertEqual(wrapper.indices, (0, 70, 299))
            self.assertEqual(wrapper.seconds, (0., 2.335669, 9.976643))
            self.assertEqual(result['frame_timestamps'], [list(wrapper.seconds)])
            self.assertEqual(result['patch_positions'][:, 0].unique_consecutive().tolist(), [0, 70, 299])
            expected = self.base(videos=[[__import__('PIL.Image', fromlist=['Image']).fromarray(x) for x in self.frames]])
            self.assertTrue(torch.equal(result['pixel_values_videos'], expected['pixel_values_videos']))
            self.assertEqual(wrapper.calls, 1)
        self.assertIs(self.processor.video_processor, self.base)

    def test_delegate_and_restore_after_failure(self):
        initial = (self.base.fixed_num_frames, self.base.max_frames, self.base.target_fps)
        with self.assertRaisesRegex(RuntimeError, 'body failed'):
            with a.selected_video_inputs(self.processor, self.frames, [0, 1, 2], 30., 300) as wrapper:
                wrapper.fixed_num_frames = 3
                wrapper.max_frames = 10
                wrapper.target_fps = 1.
                self.assertEqual(self.base.fixed_num_frames, 3)
                raise RuntimeError('body failed')
        self.assertIs(self.processor.video_processor, self.base)
        self.assertEqual((self.base.fixed_num_frames, self.base.max_frames, self.base.target_fps), initial)

    def test_nested_selection_refused(self):
        with a.selected_video_inputs(self.processor, self.frames, [0, 1, 2], 30., 300):
            with self.assertRaises(RuntimeError):
                with a.selected_video_inputs(self.processor, self.frames, [0, 1, 2], 30., 300):
                    pass

    def test_wrong_source_batch_and_grid_refused(self):
        with a.selected_video_inputs(self.processor, self.frames, [0, 1, 2], 30., 300, 'v.mp4') as wrapper:
            for videos in [['other.mp4'], [], ['v.mp4', 'other.mp4']]:
                with self.subTest(videos=videos), self.assertRaises(ValueError):
                    wrapper(videos=videos)
            with self.assertRaises(ValueError):
                wrapper(videos=['v.mp4'], return_tensors='np')
            with patch.object(a, '_prepare_frames', return_value=[]):
                with self.assertRaises(Exception):
                    wrapper(videos=['v.mp4'])

    def test_all_tensor_fields_compared(self):
        import torch
        original = dict(input_ids=torch.tensor([[1,2]]), attention_mask=torch.tensor([[1,1]]),
                        pixel_values=torch.tensor([[10.,20.]]), patch_positions=torch.tensor([[70,0,0]]))
        self.assertTrue(a.assert_inputs_equal(original, deepcopy(original)))
        for name in original:
            modified = deepcopy(original)
            modified[name].view(-1)[0] += 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                a.assert_inputs_equal(original, modified)
        with self.assertRaises(ValueError):
            a.assert_inputs_equal(original, dict(input_ids=original['input_ids']))
        changed = deepcopy(original); changed['pixel_values'] = changed['pixel_values'].to(torch.float16)
        with self.assertRaises(ValueError):
            a.assert_inputs_equal(original, changed)

    def test_invalid_frame_pixels_refused(self):
        import numpy as np
        from PIL import Image
        for frames in [[np.zeros((2,2), dtype=np.uint8)], [np.zeros((2,2,3), dtype=np.float32)],
                       [Image.new('L',(2,2))], [np.zeros((0,2,3), dtype=np.uint8)]]:
            with self.subTest(shape=str(type(frames[0]))), self.assertRaises(ValueError):
                a._prepare_frames(self.base, frames, smart_resize)
        with self.assertRaises(ValueError):
            a._prepare_frames(self.base, [np.zeros((2,2,3), dtype=np.uint8), np.zeros((3,2,3), dtype=np.uint8)], smart_resize)

    @unittest.skipUnless(importlib.util.find_spec('torchvision') is not None, 'requires existing torchvision')
    def test_pre_resize_exact_native_formula(self):
        import numpy as np
        import torch
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        base = FakeBase();base.resize_frames=True
        rng = np.random.default_rng(17)
        raw = rng.integers(0,256,(3,11,17,3),dtype=np.uint8)
        actual = np.stack([np.asarray(x) for x in a._prepare_frames(base, list(raw), smart_resize)])
        expected = transforms.functional.resize(torch.from_numpy(raw).permute(0,3,1,2).contiguous(),
            [4,8], interpolation=InterpolationMode.BICUBIC, antialias=True).permute(0,2,3,1).contiguous().numpy()
        self.assertTrue(np.array_equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
