# 模型／未来流适配器契约 v1

所有输入 float32，批量 B=1（导出固定）；N/H/S/M/D 来自检查点 config。

| 名称 | 形状 | 含义 |
|---|---|---|
| video_frames | B,N,3,H,H | RGB，NHWC→NCHW，`pixel/127.5-1`，缩放并黑边补齐，不裁掉主体 |
| audio_pcm | B,N,S | 截至同一决策时刻的过去 PCM，16kHz 单声道，[-1,1] |
| modality_mask | B,N,2 | video/audio 分别是否有效，0 或 1；缺失用零，不能当正常输入 |
| camera_mask | B,N | 能否选择该视频机位；不能大于对应 video modality |
| media_age | B,N,2 | 最近视频帧／音频片段末端距当前时刻的秒数，非负；缺失填10 |
| current_program | B,N | 实际正在播出的机位 one-hot；没有已知 PGM 时全0，不是预览机位 |
| held_seconds | B,1 | 距上一次实际提交切换的秒数，非负 |
| memory | B,M,D | 第一次全0，之后用上一步 next_memory，不跨房间复用 |
| memory_mask | B,M | 第一次全0，之后用上一步 next_memory_mask |

机位槽顺序在一段序列内必须稳定，camera IDs 到槽的映射由适配器维护。当前模型不使用固定机位 ID embedding；槽重排应同步重排所有机位输入，结果对应重排。机位重连／新增／删除／时钟跳变，应更新 generation 并重置记忆；不得悄悄复用旧槽语义。

输出：`keep_value [B,1]`、`take_value [B,N]`、`switch_gain [B,N]`、`next_memory [B,M,D]`、`next_memory_mask [B,M]`。全部 finite；缺失机位屏蔽值为 -10000。排序数值不是概率。无有效机位时适配器必须保留／恢复安全画面，不能拿屏蔽值排序执行切换。

```python
from camix_director.stream import StreamingPolicy
policy = StreamingPolicy("runs/某次实验/best.pt")
# features 是上表前7项：已解码、已对齐、float32 torch.Tensor，B=1。
result = policy.step(features, timestamp=1.2, generation="room-A:sources-v1")
assert result["execute_allowed"] is False
# 不能用此示例直接操作 Camix。
```

Core ML 使用同一顺序与维度；必须从模型 metadata 的 config_json 读取参数，不要猜测。每步取回新记忆后再执行下一次预测，禁止并发乱序覆盖。当前导出最低部署目标 iOS 17，FP32 CPU 数值验证；不等于 Neural Engine 已经支持所有算子，也不等于设备上性能达标。

## 监督与泛化边界

训练损失：在当前时刻已标注、合法的动作对中，对更优候选施加 `softplus(-(value_good-value_bad))`。当前机位 TAKE 与 KEEP 去重；未标注为 NaN；不把没有导演点击看作 KEEP。不定义场景类别，但也没有凭空推断“什么是好导播”的能力。

序列 burn-in 的前几步只填充记忆，不回传梯度；后续窗口内保留记忆的反向传播。更长时间的策略需要更长训练窗口、更充分的真实导演偏好、不同场景的独立测试与反馈，不是简单把 memory_tokens 改大。
