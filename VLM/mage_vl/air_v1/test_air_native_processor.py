"""真实Mage AutoProcessor的CPU契约测试，不加载模型权重，不解码真实视频。

合成RGB数组仅为单元测试夹具，不保存为数据集，也不参与实验准确率统计。
使用原生文件入口（仅替换解码函数）与显式帧适配入口比较所有最终Tensor。
"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

MODEL = Path(os.environ.get('AIR_TEST_MODEL_DIR',
    '/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VLM/mage_vl/models'))
RESULTS = Path('/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VLM/eval_results/practical_mage_v1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_MODULES_CACHE', str(RESULTS/'18_air_frames_v1/cpu_hf_modules'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from air_mage_adapter import selected_video_inputs, image_inputs, assert_inputs_equal


class Capture:
    """与父工程运行时相同的透明外层包装，验证嵌套属性覆盖／恢复。"""
    def __init__(self, base):
        object.__setattr__(self, 'base', base)
        object.__setattr__(self, 'last', None)
    def __getattr__(self, name):
        return getattr(self.base, name)
    def __setattr__(self, name, value):
        if name in ('base', 'last'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.base, name, value)
    def __call__(self, *args, **kwargs):
        result = self.base(*args, **kwargs)
        object.__setattr__(self, 'last', result)
        return result


AVAILABLE = MODEL.is_dir() and all(importlib.util.find_spec(n) is not None
    for n in ['torch', 'transformers', 'numpy', 'PIL', 'torchvision'])


@unittest.skipUnless(AVAILABLE, 'requires existing local Mage processor and CPU libraries')
class NativeProcessorCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from transformers import AutoProcessor
        # 只加载处理器、tokenizer及本地Python模块；没有AutoModel或from_config。
        cls.processor = AutoProcessor.from_pretrained(
            str(MODEL), trust_remote_code=True, local_files_only=True,
        )
        for part in [cls.processor.video_processor, cls.processor.image_processor]:
            part.min_pixels = 3136
            part.max_pixels = 150000
        cls.np = np
        cls.torch = torch
        cls.native_vp = cls.processor.video_processor
        cls.native_module = sys.modules[type(cls.native_vp).__module__]
        cls.question = ('These are the frames of a video. Select the best answer.\n'
                        'What appears at the end?\nA. first\nB. second\nC. third\nD. final-option-marker')

    def _raw(self, count):
        # 奇数尺寸会触发原生预缩放，能发现“只走image processor”的像素差异。
        return self.np.random.default_rng(781).integers(0, 256, (count, 101, 203, 3), dtype=self.np.uint8)

    def _prepared_native_formula(self, raw):
        """独立复写原生解码后的缩放公式，不调用待测适配器的预处理函数。"""
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        vp = self.native_vp
        h, w = raw.shape[1:3]
        nh, nw = h, w
        if vp.resize_frames and (vp.min_pixels or vp.max_pixels):
            nh, nw = self.native_module.smart_resize(
                h, w, vp.patch_size, min_pixels=vp.min_pixels,
                max_pixels=vp.max_pixels, align_patch_size=vp.patch_size * 2,
            )
        tensor = self.torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous()
        if (nh, nw) != (h, w):
            tensor = transforms.functional.resize(
                tensor, [nh, nw], interpolation=InterpolationMode.BICUBIC, antialias=True,
            )
        arrays = tensor.permute(0, 2, 3, 1).contiguous().numpy()
        return [Image.fromarray(frame) for frame in arrays]

    def _exercise(self, indices, fps, frame_count, capture_enabled):
        processor = self.processor
        original = processor.video_processor
        self.assertIs(original, self.native_vp)
        wrapped = Capture(original) if capture_enabled else original
        processor.video_processor = wrapped
        original_sampling = (original.fixed_num_frames, original.max_frames, original.target_fps)
        raw = self._raw(len(indices))
        prepared = self._prepared_native_formula(raw)
        times = {str(i): self.native_module.format_timestamp(i/fps) for i in indices}
        expected_seconds = [self.native_module.time_str_to_seconds(times[str(i)]) for i in indices]
        text = processor.apply_chat_template([
            {'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': self.question}]}
        ], tokenize=False, add_generation_prompt=True)
        calls = []
        fake_path = '__AIR_CPU_UNIT_FIXTURE_NOT_A_REAL_VIDEO__.mp4'

        def fake_extract(**kwargs):
            self.assertEqual(kwargs['video_path'], fake_path)
            self.assertEqual(kwargs['fixed_num_frames'], len(indices))
            self.assertEqual(kwargs['max_pixels'], 150000)
            self.assertEqual(kwargs['min_pixels'], 3136)
            calls.append(kwargs)
            return list(prepared), self.torch.tensor(indices, dtype=self.torch.int64), dict(times)

        try:
            with patch.object(self.native_module, 'extract_video_frames_to_pil', side_effect=fake_extract):
                native = processor(text=[text], videos=[fake_path], video_backend='frames',
                                   num_frames=len(indices), max_pixels=150000,
                                   return_tensors='pt', padding=True)
                if capture_enabled:
                    # 运行时必须在显式入口前保存本次原生秒数；中间Capture不能代替adapter.seconds。
                    native_captured_seconds = list(wrapped.last['frame_timestamps'][0])
                    self.assertEqual(native_captured_seconds, expected_seconds)
                    wrapped.last = None
                with selected_video_inputs(processor, raw, indices, fps, frame_count,
                                           video_path=fake_path) as selected:
                    # 运行时仍传uniform默认32；显式选择必须尊重本次原始indices而不重新采样。
                    adapted = processor(text=[text], videos=[fake_path], video_backend='frames',
                                        num_frames=32, max_pixels=150000,
                                        return_tensors='pt', padding=True)
                    self.assertEqual(list(selected.seconds), expected_seconds)
                    self.assertEqual(selected.calls, 1)
                    self.assertEqual(list(selected.indices), indices)
                self.assertIs(processor.video_processor, wrapped)
            # 仅原生路径触发mock解码；适配入口完全使用传入的RGB数组。
            self.assertEqual(len(calls), 1)
            assert_inputs_equal(native, adapted)
            self.assertEqual(native['patch_positions'][:, 0].unique_consecutive().tolist(), indices)
            self.assertEqual(adapted['patch_positions'][:, 0].unique_consecutive().tolist(), indices)
            self.assertEqual(adapted['image_grid_thw'].shape[0], len(indices))
            for key, value in adapted.items():
                self.assertEqual(value.device.type, 'cpu', key)
            decoded = processor.tokenizer.decode(adapted['input_ids'][0], skip_special_tokens=False)
            self.assertIn('D. final-option-marker', decoded)
            self.assertEqual((original.fixed_num_frames, original.max_frames, original.target_fps), original_sampling)
        finally:
            processor.video_processor = original

    def test_uniform32_full_input_parity(self):
        indices = self.native_module.select_frame_indices(13356, 32)
        for capture in [False, True]:
            with self.subTest(capture=capture):
                self._exercise(indices, 29.97002997002997, 13356, capture)

    def test_shared_uniform_helper_matches_actual_native_sampler(self):
        from air_selection import uniform_indices
        # 13356为用户真实失败视频帧数；同时覆盖短视频、半整数和不同预算。
        for count in [1,3,31,32,33,50,63,64,1000,1024,13356,100001]:
            for budget in [1,3,16,32,64]:
                with self.subTest(count=count,budget=budget):
                    self.assertEqual(uniform_indices(count,budget),
                                     self.native_module.select_frame_indices(count,budget))
        self.assertNotEqual(uniform_indices(13356,32),
                            self.np.linspace(0,13355,32,dtype=int).tolist())

    def test_nonuniform3_full_input_parity(self):
        for capture in [False, True]:
            with self.subTest(capture=capture):
                self._exercise([0, 70, 299], 29.97, 300, capture)

    def test_single_image_analysis_uses_native_image_path(self):
        from PIL import Image
        image = Image.fromarray(self._raw(1)[0])
        original = self.processor.video_processor
        with patch.object(self.native_module, 'extract_video_frames_to_pil', side_effect=AssertionError('no video decode')):
            values = image_inputs(self.processor, image, self.question)
        self.assertIs(self.processor.video_processor, original)
        self.assertEqual(values['image_grid_thw'].shape[0], 1)
        self.assertEqual(values['patch_positions'][:,0].unique_consecutive().tolist(), [0])
        self.assertIn('D. final-option-marker', self.processor.tokenizer.decode(values['input_ids'][0], skip_special_tokens=False))
        for key, value in values.items():
            self.assertEqual(value.device.type, 'cpu', key)


if __name__ == '__main__':
    unittest.main()
