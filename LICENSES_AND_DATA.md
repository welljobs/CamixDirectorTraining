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

## 本仓库发布的 AMI 素材附件

本次限定发布 ES2006a、ES2007a、ES2008a、ES2009a 的32个原始音视频 signals，以及16个依据word时间生成的弱监督训练包，入口：https://github.com/welljobs/CamixDirectorTraining/releases/tag/ami-originals-576-20261116

原片未重编码；派生包的采样、缩放、音频窗口与弱标签修改均在附件署名文件说明。保留AMI Meeting Corpus / AMI Consortium出处、CC BY4.0链接/文本和无背书说明。没有公开完整人工标注归档或将该许可扩展到其他注释。

Git历史不包含这些大文件；附件中的 `ami-materials-manifest.json` 和 `SHA256SUMS` 用于完整性/复现，不冒充官方逐文件签名。下载、移动或重新训练并不会把弱监督升级为真实导演审核。此素材发布不为软件代码授予额外许可，也不对所有用途的其他权利作额外保证。
