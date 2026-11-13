# Windows 安装与训练

## 支持目标和验证范围

- 兼容目标：Windows 10/11 x86-64、Python 3.11 x64；不声称支持 Windows ARM 或32位Python。
- CPU路径不要求独立显卡；NVIDIA加速通过单独选择官方CUDA版PyTorch。
- 训练、公开素材准备、音视频数据导入、评估、网页训练台不依赖CoreMLTools。
- Core ML导出与数值验证在Mac上执行。Windows界面和API均禁止该操作，并提示下载`.pt`转到Mac。
- 本轮实际运行环境为Mac；Windows锁和进程控制分支有针对性测试，提供Windows CPU CI配置。**未在Windows真机上完成安装、CPU训练和CUDA实测，不能把分支测试当作整机验收。**

## 最简单的使用方法

1. 把Windows迁移ZIP解压到一个可写目录，例如 `D:\CamixDirectorTraining`。可有空格／中文路径，启动脚本和JSON均明确使用UTF-8。
2. 安装Python 3.11 x64并启用Windows `py` launcher。终端运行 `py -3.11 --version` 应有结果。
3. 双击 `setup_windows.bat`。默认安装CPU版PyTorch及训练依赖，环境单独放在 `.venv-win`，不改变系统Python。
4. 安装成功后双击 `start_web_windows.bat`，浏览器打开本地训练台。若没有自动打开，复制终端中完整URL。
5. 示例数据已包含在迁移包时可直接选择；如果没有数据，可以先用“合成烟雾测试”检查链路，再准备真实数据。
6. 安装／启动失败时保留终端输出，先看具体错误，不要盲目复制Mac虚拟环境或安装Mac锁文件。

## 可选：NVIDIA CUDA

在CMD中进入工程目录，根据显卡与驱动情况选择安装源：

```bat
setup_windows.bat cpu
setup_windows.bat cu118
setup_windows.bat cu126
setup_windows.bat cu128
```

这些是PyTorch **2.7.0** 官方历史版本提供的安装变体；不是“每张显卡都支持”的承诺。请对照官方安装说明与NVIDIA驱动支持情况选择，不能仅看显卡显存容量。固定旧版本也不保证兼容未来的新GPU架构。

安装脚本显式重新安装选定的torch/torchvision构建，避免先装CPU版后切换CUDA时pip认为版本已满足、实际仍用CPU。

安装后检查：

```bat
.venv-win\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

- `torch.cuda.is_available()`为True后，重启网页服务，计算设备下拉框才可选择CUDA。
- 为False时不要强行选CUDA；可先用CPU完成安装／小规模训练检查，再排查GPU架构、驱动与构建是否匹配。
- 本机没有可用于验证的NVIDIA Windows GPU，因此这里不提供未经实测的速度承诺。
- 更换安装变体可能重新下载较大依赖；需要联网和额外磁盘空间。

## CLI也仍然可用

```bat
.venv-win\Scripts\python.exe -m camix_director.train --data data\ami_pilot\prepared --output runs\windows_experiment_001 --config examples\pilot-config.json --epochs 30 --sequence-steps 16 --burn-in 4 --batch-size 2 --allow-weak-supervision --balance-actions --device cpu

test_windows.bat
```

换CUDA时把 `--device cpu` 改成 `--device cuda`。正式自己的数据不要盲目加弱监督／合成开关。

## 从Windows把模型交给Mac

复制该实验的 `best.pt`、`training-report.json` 和本工程同一版本代码到Mac。检查点内包含模型配置和CPU保存的参数，Mac加载时不会重新下载预训练权重。

Mac可以在网页中选择该检查点导出，或在终端运行：

```bash
.venv/bin/python -m camix_director.export \
  --checkpoint runs/windows_experiment_001/best.pt \
  --output runs/coreml_from_windows_001 --precision fp32 --parity-steps 12
```

不要复制 `.venv-win` 到Mac，也不要把 `.venv/bin/python` 复制到Windows。

## 参考与CI

- PyTorch 2.7.0官方安装源：https://pytorch.org/get-started/previous-versions/
- Python虚拟环境：https://docs.python.org/3.11/library/venv.html
- 数据许可与监督边界：`LICENSES_AND_DATA.md`。
- `.github/workflows/windows-training.yml` 提供Windows CPU训练／网页／停止任务回归。将项目放到自己的GitHub仓库后可运行；本轮没有创建远程仓库、上传素材或宣称CI已通过。

`requirements.lock`保留为原Mac实验的依赖复现记录。Windows启动器使用`requirements-train.txt`和所选PyTorch源，不安装Mac专属导出依赖。
