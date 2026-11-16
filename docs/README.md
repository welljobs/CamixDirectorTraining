# CamixDirectorTraining 文档导航

本目录属于独立的 PyTorch 训练项目，不是 Camix iOS App 的运行或真机测试报告。
仓库不附带视频、数据包、模型权重、设备日志或本地会话；这些文件由使用者在本机生成。

## 安装与日常使用

- [项目介绍与运行边界](../README.md)
- [Studio 0.3 完整工作流](../STUDIO_GUIDE.md)
- [本地网页训练台](../WEB_CONSOLE.md)
- [Windows 安装与限制](../WINDOWS.md)
- [AMI 原分辨率素材下载与准备](../AMI_ORIGINALS_GUIDE.md)
- [第三方来源与数据许可](../LICENSES_AND_DATA.md)

## V1 可复现研究基线

- [模型与流适配器契约](MODEL_CONTRACT.md)：输入输出、有限记忆、监督及导出边界。
- V1 仍是研究模型；能够训练和数值导出，不等于专业节目质量或真机性能已达标。

## V2 长事件与节目偏好

- [整体改造路线](DIRECTOR_V2_REDESIGN_PLAN.md)：目标、阶段、数据与验收思路。
- [P0/P1 基础工作流](V2_FOUNDATION_GUIDE.md)：事件协议、标注审核、冻结特征、完整事件基线。
- [P2–P5 可执行研究链路](V2_STAGES_GUIDE.md)：策略训练、多尺度记忆、偏好、蒸馏、评估、导出和记录式影子。
- 感知仍以本地冻结视觉特征和音频能量为主；强音视频语义、真实导演证据、实际流接入与端侧性能不因工程测试通过而自动成立。

## 历史设计与来源快照

- [最初的 PyTorch 记忆导播提案](PYTORCH_MEMORY_DIRECTOR_PROPOSAL.md)：2026-11-11 的历史设计，不是当前实现或配置的权威说明。
- [Windows PyTorch 安装来源快照](sources/windows-pytorch-install.json)
- [Windows wheel 可用性来源快照](sources/windows-wheel-availability.json)

## 结果应如何解读

运行后产生的 `runs/`、`reports/` 和 `.cache/` 不提交到源码仓库。合成冒烟与工程测试只证明对应路径可执行；专业能力仍需要独立原始事件、真实导演评审、因果边界以及目标设备的端到端验证。所有当前研究输出均未批准真实节目控制。
