# AMI 原分辨率素材（2026-11-12）

## 这次下载的是什么

- 官方 AMI Meeting Corpus 的 `_orig.avi`，保留原文件；不是把小视频放大。
- 实测样本：压缩版 **352×288 / 25fps**；原分辨率版 **720×576 / 25fps**。
- **不是 720p、1080p、4K**；AMI 是早期会议语料，不是现代高清拍摄。
- 四场独立会议，每场四路人物近景 + 四条对应头戴话筒，**16视频 +16音轨**。
- 文件头核验总量 **6,300,005,744字节（6.30GB /5.87GiB）**。训练包/缓存和浏览器片段另占空间。
- ES2006a、ES2007a → train；ES2008a → validation；ES2009a → test。整场隔离，分段不能冒充独立场次。

## 数据、代码和启动

项目：克隆后的 CamixDirectorTraining 仓库根目录；以下命令在该目录执行。

- 原文件：`data/ami_originals_576/raw/`
- 官方页面快照：`data/ami_originals_576/sources/`
- 下载计划：`data/ami_originals_576/download-plan.json`（首次运行会自动使用仓库自带的 `examples/ami-originals-download-plan.json` 公共来源模板，不需要手写清单）
- 校验/来源：`data/ami_originals_576/source-manifest.json`（仅全部成功后出现）
- 下载续传：双击 `download_ami_originals.command`。
- 素材准备：双击 `prepare_ami_originals.command`。
- 新配置：`examples/ami-originals-224-config.json`。
- 新数据包：`data/ami_originals_576/prepared_224_4x60/`。
- 网页训练台“刷新工作区”后，选择上述数据目录与 `ami-originals-224-config`；二者需配套。
- 如已有旧权重，不应直接完整恢复：新数据/4机位/224px配置不同。选“从头训练”；需要微调时先核对模型结构兼容性。

下载使用最多4连接、合计限速6MiB/s、验证断点Range/ETag、逐文件流式SHA256及来源收据。低于5GiB保留空间会安全停止，保留`.part`，腾空间后续传；不删除旧素材腾空间。官方不提供逐视频SHA256，清单里的SHA256是**本地下载完整性/复现哈希**，不是官方签名。

停止当前独立下载/准备管线：在 `data/ami_originals_576/` 新建空文件 `STOP`（下载阶段支持；素材解码正在进行时不要强制关机）。恢复前删除自己创建的STOP再重跑下载。素材准备逐片原子提交，已完成文件不重复写。

## 训练输入不是把6GB全部塞进内存

- 原视频720×576保留；训练抽样为**224×224 RGB、保持比例加黑边**，不是64×64。
- 四路16kHz PCM，每次因果音频窗0.25秒；决策采样2Hz。
- 有限记忆Transformer：hidden64、2层、16个记忆token；是研究基线，**不是已经训练好的大模型**。
- 首轮均匀抽取每场4段×60秒，共16段/16分钟：train8分钟、validation4分钟、test4分钟。全场原文件供后续扩充；本轮没有声称把所有时长都训练了。
- 单片数组有512MiB保护；训练用磁盘映射缓存，建议MPS / batch1 / sequence8 / burn-in2作为起点，实际速度/峰值需看本机指标。
- 分辨率提高、机位变多会增加算力，不因源文件只有6GB就保证训练速度。

## 标签和预览

标签由官方单词时间推导：**四个人中恰好只有一人在说话**时，给该说话人对应镜头较高偏好；重叠发言/无人说话不标。节目当前机位按片段轮换模拟，**没有真实导演切播轨迹**。必须勾选“我确认使用弱监督标签”，不得把`human_reviewed`改成true绕过。

这仍然偏向“跟随说话人”，不等于学会构图/舞台/运动/节奏；更清晰的画面不会自动解决偏好标签不足，也不能证明模型用到了视觉或长记忆。

原AVI不能直接在浏览器稳定播放。`camix_director.ami_review`创建**720×576 /25fps H264审片副本**（CRF18有损，非逐位原视频）及未审核审片项目；训练仍读原AVI。审片项目起始没有人工偏好、不自动标成审核完成。CLI：

```bash
.venv/bin/python -m camix_director.ami_review --workspace "$PWD" --root "$PWD/data/ami_originals_576"
```

## 官方来源及许可

- AMI下载页：https://groups.inf.ed.ac.uk/ami/download/（正文说明较大版本文件名末尾使用 `_orig`，并要求限速）。
- 信号规格/机位和话筒映射：https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml
- 许可：https://groups.inf.ed.ac.uk/ami/corpus/ （signals/transcriptions：CC BY4.0，保留AMI Consortium署名和修改声明）。
- 原文件镜像：https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/
- 仅使用音视频信号和单词转写；不把其他不同许可注释混作同一许可。

**不修改 Camix App、不替换旧推荐模型、不批准真实节目使用。** 下载完成 ≠ 数据准备完成 ≠ 已训练完成 ≠ 导播质量提升；分别以来源清单、准备报告、训练报告和独立评估为准。
