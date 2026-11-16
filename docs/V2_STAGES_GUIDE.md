# V2 P2–P5 可执行研究链路

日期：2026-11-13。本文区分 **工程能力已实现** 与 **真实专业能力已验收**。所有策略、质量模型、导出与影子结果仍为 `production_approved=false`、`execute_allowed=false`。

## 1. 本轮范围

| 阶段 | 已实现的研究能力 | 不包含/尚未验收 |
|---|---|---|
| P2 感知特征与策略 | 本地冻结视觉特征＋音频 RMS 的共享机位注意力策略；集合监督、可选辅助任务；独立语料与训练 | 更强音视频基础模型的效果、ASR/活跃说话人/跨视角身份模块未接入；不能称为已理解语义 |
| P3 节奏与记忆 | 短时摘要、实际已播镜头摘要、慢速摘要；切换/机位/意图/事件可见性头；数值风格条件 | 慢速记忆不是事实数据库，意图预测不是可靠解释；未证明长期叙事或专业节奏 |
| P4 闭环与偏好 | 自身路径回放、校验后的未审核纠正草稿、审核片段的参考相对轨迹偏好、离线片段质量学习 | 没有用模型路径自动造专家答案；没有在线 RL/直播探索；真实导演偏好与盲评不足 |
| P5 蒸馏与部署边界 | 因果教师/学生独立记忆蒸馏、TorchScript、可选 Mac Core ML 数值对齐、严格特征包建议接口与记录式影子 | 不含原片采集/解码/编码器端侧移植、实际 RTC/RTMP 适配、iPad/ANE/温升/端到端性能验收 |

现有 V1 核心和已有权重保留。没有自动下载基础权重、租用算力、修改 Camix App 或启用真实切播。

## 2. 网页入口与推荐顺序

正常启动原有 `bash start_web.command`。从 `/v2` 页顶部的 **策略训练 / 偏好 / 蒸馏 / 导出** 进入 `/v2-model`，同一浏览器会话沿用本地令牌。已有旧服务需等当前任务结束后正常重启，不能为升级页面强杀训练/下载。

1. 在审片页导入授权素材、确认原始事件 ID/划分/时钟、补标签并审核。
2. 生成“研究样本”索引，使用同一编码器、权重和预处理为各事件提取特征。旧缓存不会被改写；跨提取代码版本的特征空间可能被拒绝混用，此时应统一重提。
3. 在策略页选择这些成员，创建语料清单。训练与验证必须是独立原始事件，测试只用于锁定后的评估。
4. 选择“监督式多任务导播”，核对 `visual_dim`、机位上限和策略规模，再训练。
5. 独立测试运行完整/抹视觉/静音/清记忆消融，同时运行相同输入与保护条件的学习策略、保持、RMS 闭环。
6. 对模型自身路径建立纠正草稿，返回审片页重新标注/审核；或者补充同源片段 A/B。
7. 使用已审核 A/B 做参考相对偏好训练；离线质量模型单独训练与独立测试。
8. 选因果 V2 教师，训练较小的学生；重新评估学生，不能假定蒸馏等价。
9. 导出策略步；可选 Core ML CPU 对齐。生成特征包并做记录式影子，最后核对研究门槛。

“完整合成冒烟”按钮会生成明确标注的虚构素材/标签/自动测试审核，跑通上述工程路径。它不是专业导演数据，不能被用来批准生产。

## 3. 模型与输入契约

`model_v2.py` 的 `FeatureDirector` 使用：

- 视觉特征投影、音频对数能量、缺失掩码、媒体年龄、实际 PGM 和保持时长；
- 同时刻共享跨机位注意力，无固定机位编号类别；
- 三组显式有限记忆：每步更新的短记忆、实际已提交切换才更新的镜头记忆、按声明时间间隔更新的慢摘要；
- 切换 logit、共享候选机位 logit、意图分布、事件可见性辅助头；
- KEEP 与非当前机位 CUT 的合法动作分布，当前机位 CUT 与 KEEP 不重复。

`style.name/notes` 仅保存为元数据，程序不声称理解自由文本。可选 `pace`、`reaction_bias`、`detail_bias`（各 -1～1）通过三维条件投影进入策略，不是按场景写切播阈值。没有相应条件覆盖的数据就不能声称风格控制有效；流运行时改变风格需显式新 generation。

缺失/不确定标签不变成 KEEP。可接受切换区间不是“每一帧必须切”：区间外的非当前候选不会被强制当当前目标，也不自动发明 KEEP。多个可接受动作使用集合概率质量；切换辅助仅在保持/切换判断不歧义时启用。没有意图或事件标注，对应辅助损失不启用。

模型输入只取观察与请求条件。事件文本、偏好标签、审核姓名、未来镜头不进入策略。教师 PGM 使用决策前状态，同一时刻的示范切换不能提前成为输入。

## 4. 语料、泄漏与编码器身份

`corpus_v2.py` 的清单格式是 `camix_feature_corpus_v2`，每个成员引用一个索引与一个 cache_id。样例：

```json
{"format":"camix_feature_corpus_v2","name":"my-reviewed-corpus","entries":[
  {"index":"runs/v2/EVENT_JOB/index.json","cache_id":"EXACT_64_HEX_CACHE_ID"}
]}
```

模型必须匹配相同编码器权重、提取/预处理代码、环境及特征维度的 feature-space 身份。只看维度相同不足以复用策略。语料校验拒绝同一原始事件或相同源内容跨划分、重复成员与混用特征空间。同一源内容换成不同事件 ID，即使同划分也不能充当独立事件证据。

同一原始训练事件可加入不同已审核的纠正轨迹，但必须同源、同划分、有不同且可追溯的 replay 来源；不能用同一 PGM 的冲突标签版本充数。验证/测试不允许重复原始事件抬高样本数。父模型/教师曾训练过的事件不能再当独立验证事件；测试与所有训练/模型选择历史核对。

测试指标以原始事件为独立单位；不足三个事件不生成跨事件 bootstrap 区间，三个也不是充分统计/专业证据。冻结编码器预训练数据是否碰过评估素材还需要额外审计，项目不能仅凭权重哈希证明。

## 5. 训练、恢复与停止

`train_v2.py` 支持：

- `supervised`：集合动作监督＋有标签才启用的切换/意图/事件可见性任务；
- `preference`：从审核 A/B 学习参考模型相对的平均轨迹 log-likelihood 偏好；不是在线 RL，也不把离线质量模型的自动分数冒充人审；
- `distill`：教师和学生只看各自同一因果观察、实际 PGM，并独立滚动记忆；软分布蒸馏与原监督共同训练。

训练默认 CPU，可显式选实际可用 MPS/CUDA。只按实际测试范围宣传；没有验证过的设备/规模不等于支持其吞吐与热稳定性。未实现 V2 AMP/DDP/多机大规模训练。

每个事件连续消费分块、块间截断梯度，但保留数值记忆。单事件接口拒绝多 worker 复制事件；不能把打乱块的记忆直接拼接。

**完整恢复范围为已提交整轮边界。** `last.pt` 保存该边界的模型、优化器、RNG、历史、最优权重、数据/代码/配置身份。轮中停止不提交部分更新，恢复会从上个边界重做当前轮。它不是中间批次精确恢复。改变计划、设备、数据、配置、教师或代码须新建微调实验；完成或早停的计划不能偷偷加轮。

输出 `best.pt`、`last.pt`、`training-report.json`、`run-manifest.json` 与关键训练源码快照；不会自动执行检查点旁的代码。早停依据独立验证目标，不用最终测试集选模型。loss weights 是可追溯的研究默认值，不是已验证最优配方。

## 6. 片段偏好、质量与纠正

每个 A/B 路径分别重建决策前 PGM、保持时长和记忆，不能套用共同专家 PGM。前缀只读历史；有界片段内保留梯度。策略偏好片段最多 128 个决策步，质量模型片段最多 512 步；不能把一个长片段的偏好随意拆成多个伪标签。

质量模型使用整段被选画面、同时刻候选池化特征、声音、切换与时长、已知风格条件做离线 GRU 评分，不读事件/胜负标签作为输入。平局保留，不确定不训练。它可以看到完整比较片段，所以不能直接作为实时策略特征或非因果教师。

`corrections_v2.py` 校验索引、源版本、回放报告和逐时刻轨迹哈希、PGM 连续性、可用机位与切换列表后创建草稿。模型路径标为 `offline_policy`，不冒充真实导演记录。旧相对标签、A/B 和审核被清除；事件事实仅保留供重审。没有新动作/有效片段标注，不能把该纠正草稿提升成策略训练成员。

## 7. 导出、特征包与影子

`export_v2.py` 默认输出 `FeatureDirectorStep.pt`、`signature.json` 和数值报告。Mac 可选 `FeatureDirectorStep.mlpackage`，FP32 CPU，最低目标 iOS 17。多步检查使用各实现自己更新的短/镜头/慢记忆，含切换、慢记忆更新、单路/全路缺失；不是每步重用 PyTorch 的真实状态给另一实现。

**导出的是策略步，不是完整手机直播系统。** 部署方必须复现同一个视觉编码器、letterbox/归一化、声音特征、时钟、槽映射和实际 PGM 反馈。未实现 V2 编码器端侧一体化、解码/采集服务或 iPad 工程接入。

`StreamingDirector` 是只返回建议的 Python 边界；它检查 feature-space、时钟配置、实际 PGM、媒体内容/可用时间、掩码、generation、机位顺序、风格改变和时间倒退。过大时间间隙重置记忆。无有效视频返回人工介入；现 PGM 无效时优先建议有效源；未校准概率门槛只是实验保护。

`shadow_v2.py` 消费本地、记录式 JSONL 特征包，记录建议；不接 RTC/RTMP，不控制混流器。计时仅 CPU 适配器/策略步，不含网络/解码/编码器；P50/P95 使用前一万步，不是端到端 SLO。包里的离线可用时间来自声明的 offset/delay，不是实时采集证明。

## 8. 命令行示例

以下从项目根目录执行。用实际路径替换大写占位符。所有命令都创建新结果，不覆盖已有实验。

```bash
# MEMBERS.json 是索引/cache_id 数组或含 entries 的对象。
.venv/bin/python -m camix_director.stages_v2 --workspace . corpus \
  --entries data/MEMBERS.json --output runs/v2/MY_CORPUS/corpus.json

.venv/bin/python -m camix_director.stages_v2 --workspace . train \
  --corpus runs/v2/MY_CORPUS/corpus.json --config data/POLICY_CONFIG.json \
  --output runs/v2/MY_SFT --mode supervised --epochs 10

.venv/bin/python -m camix_director.stages_v2 --workspace . train \
  --corpus runs/v2/MY_CORPUS/corpus.json --output runs/v2/MY_PREF \
  --mode preference --teacher-checkpoint runs/v2/MY_SFT/best.pt --epochs 5

.venv/bin/python -m camix_director.stages_v2 --workspace . train \
  --corpus runs/v2/MY_CORPUS/corpus.json --config data/SMALL_STUDENT_CONFIG.json \
  --output runs/v2/MY_STUDENT --mode distill \
  --teacher-checkpoint runs/v2/MY_PREF/best.pt --epochs 5

.venv/bin/python -m camix_director.stages_v2 --workspace . evaluate \
  --checkpoint runs/v2/MY_STUDENT/best.pt --corpus runs/v2/MY_CORPUS/corpus.json \
  --output runs/v2/MY_EVALUATION --split test

.venv/bin/python -m camix_director.stages_v2 --workspace . quality-train \
  --corpus runs/v2/MY_CORPUS/corpus.json --output runs/v2/MY_QUALITY

.venv/bin/python -m camix_director.stages_v2 --workspace . export \
  --checkpoint runs/v2/MY_STUDENT/best.pt --output runs/v2/MY_EXPORT --coreml
```

合成流程必须显式加 `--allow-synthetic`。CLI 的 `--help` 列出 replay、correct、quality-evaluate、packets、shadow、gate 和 smoke；网页提供同样入口。教师/学生机位槽上限需一致，学生可减少隐藏宽度/层数/记忆。

## 9. 晋级门槛与下一步

研究门槛绑定同一模型与实际报告/轨迹；检查独立真实审核事件、视觉/记忆消融贡献、完整回放、导出、记录式影子、同特征空间的独立质量测试。项目的样本数/正差门槛不是专业行业标准。合成全链路完成后门槛预期仍为 `blocked`。

即使研究项全部通过，只能成为 `offline_research_candidate`，仍缺：

1. 独立合格导演的盲评与分歧分析；
2. 基础编码器/数据授权与预训练重叠审计、多场景/OOD 证据；
3. 独立真实事件上的概率校准；
4. 编码器＋策略＋采集适配器的目标设备时延/内存/温升/故障测试；
5. 实际 RTC 影子、操作员接管、回滚验收。

下一轮真正的模型质量工作，应优先补真实授权多机位数据、专业示范/偏好与更强的视觉/声画语义表征，再根据独立消融决定扩大模型。不能把这次工程交付重命名成通用专业导播成品。


## 10. 已保存的验收样例

工程样例位于 `runs/v2/20261113_123638_2528fcd4/`，原片/特征及审核修订也已按原相对路径保存，语料身份重新校验通过。所有素材、偏好与自动测试审核都是 synthetic，不是真实导演证据。可在网页中查看对应监督/偏好/学生模型与各项报告。

增强原生数值证据在该目录的 `export-final/export-report.json`，12 步 FP32 CPU 最大差约 6.26e-7；旧网页截图展示的是先前一组同模型用例。两者都只验证数值，不验证节目质量或设备性能。

本轮回归 182 项通过。汇总见 `reports/v2-stages/acceptance.json`。样例的原训练源码已按检查点预存 SHA256 校验归档；最终版本新增更严格的同划分源别名检查与自动训练源码快照，所以旧样例不能被当成跨代码版本精确续训许可。需要继续训练时选择明确的新微调计划。
