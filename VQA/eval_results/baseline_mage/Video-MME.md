# Mage-VL × Video-MME 评测结果

> 生成时间: 2026-10-04 23:43:01
> 数据来源: `/root/autodl-tmp/VLMEvalKit/outputs/Mage-VL/Mage-VL_Video-MME-32frame_score_gpt-4o-mini.json`
> 数据集: Video-MME (32 帧子集, 238 视频 / 714 题)
> 模型: Mage-VL (codec + HEVC)
> 推理失败率 0% / 打分失败率 0%

## 总览

| 视频时长 | 准确率 |
| --- | --- |
| Short (短视频) | 75.8% |
| Medium (中等) | 63.6% |
| Long (长视频) | 52.7% |
| **Overall (总体)** | **63.7%** |

## 领域 (Domain)

| 领域 (Domain) | Short (短视频) | Medium (中等) | Long (长视频) | Overall (总体) |
| --- | --- | --- | --- | --- |
| Knowledge | 75.8% | 64.0% | 49.1% | **63.6%** |
| Film & Television | 74.4% | 60.4% | 52.4% | **62.0%** |
| Sports Competition | 71.8% | 53.8% | 51.3% | **59.0%** |
| Artistic Performance | 81.5% | 69.4% | 51.3% | **65.7%** |
| Life Record | 82.1% | 64.4% | 57.9% | **66.7%** |
| Multilingual | 55.6% | 80.0% | 66.7% | **70.4%** |

## 子类别 (Sub-category)

| 子类别 (Sub-category) | Short (短视频) | Medium (中等) | Long (长视频) | Overall (总体) |
| --- | --- | --- | --- | --- |
| Humanity & History | 66.7% | 33.3% | 16.7% | **50.0%** |
| Literature & Art | 33.3% | 66.7% | 33.3% | **47.6%** |
| Biology & Medicine | 100.0% | 66.7% | 83.3% | **77.8%** |
| Finance & Commerce | 50.0% | 83.3% | 0.0% | **53.3%** |
| Astronomy | 66.7% | 50.0% | 60.0% | **58.3%** |
| Geography | 83.3% | 83.3% | 33.3% | **73.3%** |
| Law | 100.0% | 58.3% | 77.8% | **76.7%** |
| Life Tip | 77.8% | 55.6% | 66.7% | **66.7%** |
| Technology | 77.8% | 66.7% | 0.0% | **61.9%** |
| Animation | 66.7% | 50.0% | 66.7% | **61.1%** |
| Movie & TV Show | 77.8% | 58.3% | 58.3% | **63.6%** |
| Documentary | 50.0% | 41.7% | 33.3% | **40.7%** |
| News Report | 100.0% | 91.7% | 50.0% | **78.8%** |
| Esports | 58.3% | 83.3% | 16.7% | **54.2%** |
| Basketball | - | 22.2% | 33.3% | **28.6%** |
| Football | 83.3% | 33.3% | 66.7% | **58.3%** |
| Athletics | 73.3% | 100.0% | - | **77.8%** |
| Other Sports | 83.3% | 66.7% | 75.0% | **73.3%** |
| Stage Play | 100.0% | 91.7% | 50.0% | **85.2%** |
| Magic Show | - | 50.0% | 60.0% | **57.1%** |
| Variety Show | 88.9% | - | 33.3% | **61.1%** |
| Acrobatics | 55.6% | 61.1% | 55.6% | **58.3%** |
| Handicraft | 80.0% | 100.0% | 55.6% | **78.8%** |
| Food | 66.7% | 50.0% | 33.3% | **50.0%** |
| Fashion | 66.7% | 33.3% | 83.3% | **55.6%** |
| Daily Life | 83.3% | 66.7% | 58.3% | **66.7%** |
| Travel | 100.0% | 100.0% | 16.7% | **72.2%** |
| Pet & Animal | 66.7% | 58.3% | 83.3% | **66.7%** |
| Exercise | - | 66.7% | 60.0% | **61.1%** |
| Multilingual | 55.6% | 80.0% | 66.7% | **70.4%** |

## 任务类型 (Task type)

| 任务类型 (Task type) | Short (短视频) | Medium (中等) | Long (长视频) | Overall (总体) |
| --- | --- | --- | --- | --- |
| Temporal Perception | 80.0% | 62.5% | 100.0% | **71.4%** |
| Spatial Perception | 50.0% | 100.0% | - | **75.0%** |
| Attribute Perception | 91.4% | 64.0% | 50.0% | **76.5%** |
| Action Recognition | 74.2% | 62.1% | 54.5% | **64.6%** |
| Object Recognition | 79.4% | 75.7% | 69.2% | **76.2%** |
| OCR Problems | 76.9% | 40.9% | 50.0% | **53.8%** |
| Counting Problem | 58.1% | 53.3% | 46.2% | **54.1%** |
| Temporal Reasoning | 66.7% | 58.3% | 25.0% | **36.2%** |
| Spatial Reasoning | 80.0% | 75.0% | 100.0% | **81.8%** |
| Action Reasoning | 76.9% | 36.8% | 48.9% | **50.6%** |
| Object Reasoning | 59.1% | 65.9% | 51.8% | **58.0%** |
| Information Synopsis | 91.3% | 88.9% | 73.2% | **82.4%** |
