# 数据来源、许可与监督限制

## AMI Meeting Corpus

来源：AMI Consortium / University of Edinburgh，AMI Meeting Corpus。
官方主页：https://groups.inf.ed.ac.uk/ami/corpus/
同步、相机／麦克风映射：https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml
许可：https://creativecommons.org/licenses/by/4.0/

官方主页明确：信号、转写及部分标注以 CC BY 4.0 发布。本项目只使用公开音视频信号及 words 转写时间。
不将该说明延伸为所有其他标注具有相同许可。用于产品、重新分发或发布权重前需重新审核实际素材与权利范围。

原始访问位置、截取范围和本地文件 SHA256 保存在 data/ami_pilot/source-report.json 和 data/ami_fresh_holdout/source-report.json。
转换：截取短区间、缩小画面／补边、音频转浮点、根据转写生成说话人镜头偏好。
不声称获得 AMI 项目对本项目或模型的认可。

**监督类型：weak_ami_transcript_speaker。**

它表示根据人工转写的词时间判断两个所选参与者谁正在说话，构造弱镜头排序目标。
它不是导演切镜头真值，不包含专业审美/节奏；当前节目历史为模拟，不是 AMI 的真实导播记录。
会议中的其他参与者、重叠发言和无所选参与者说话的时刻不强行标成“保持”。
该数据只验证一个很窄的真实媒体弱监督任务，不能证明模型支持各种场景。

## 依赖

PyTorch、TorchVision、Core ML Tools、PyAV 等依赖各自有许可证；原 wheel 内保留相应许可。
downloads/ 是本机复现依赖缓存，不代表已对再分发或商用进行了统一法律审核。

## 本工程

未擅自把 Camix 或该训练工程声明为对外开源。工程对外发布许可由项目所有者决定。
