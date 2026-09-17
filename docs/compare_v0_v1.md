# v0 与 V1 的统一验证

这是两个历史训练模型的比较，不是只改变 Structure Prior 的训练消融。
工具不修改两套模型、原训练配置或原权重。新旧公式、阶段数、Transformer
深度、Correction 和 readout 设置各自保留。

## 复制到 GPU 电脑

将这两个文件放到 **V1 项目** 的 `tools` 目录：

- `tools/compare_v0_v1.py`
- `tools/comparison_worker.py`

本工具复用 V1 项目已有的数据与指标模块。不要复制 V1 的模型文件覆盖 v0。
也不要把本工具单独放在不含 V1 包的目录运行。

默认路径已经按本次实验设置：

| 对象 | 默认路径 |
|---|---|
| v0 项目 | `C:\codeee\inpainting\FeatureGuidedElasticaADMMNet-main` |
| v0 权重 | 上述目录下 `outputs\fg_elastica_full_21p\best.pt` |
| v0 配置 | 同目录 `config_resolved.yaml` |
| V1 项目 | 运行脚本所在的项目根目录 |
| V1 权重 | V1 项目下 `outputs\structure_v1\r2\best.pt` |
| V1 配置 | 同目录 `runner_config.yaml` |
| 比较输出 | `C:\codeee\inpainting\comparison_v0_v1` |

训练与验证列表直接从这两份配置读取，不需要复制本地 `D:\data` 目录到 GPU。
GPU 电脑上的图片路径、两组 train/val 列表及 checkpoint 必须存在。
只加载自己训练的可信 checkpoint；它们包含配置等 Python 序列化元数据。

## 先预检

在 Anaconda Prompt（CMD）执行：

```bat
conda activate inpainting
cd /d C:\codeee\inpainting\s_inpainting\FeatureGuidedElasticaADMMNet
python tools/compare_v0_v1.py --device cuda --preflight-only
```

预检会：

1. 检查图像列表无重复、共同验证集未出现在任一训练列表中。CelebA 文件名
   也参与交集检查，避免目录迁移掩盖重叠；不做图片内容去重或身份级划分。
2. 核对两版预处理一致，并核对 checkpoint 内 `model`、`stage_hyper` 配置。
   如果不一致直接报错，不忽略权重，不自动改模型。如果 checkpoint 没有配置，
   明确记录这一限制；严格加载本身不能验证所有数值超参数。
3. 冻结 checkpoint 副本，记录代码、依赖版本、列表及配置指纹。
4. 检查 LPIPS 可用性。默认需要可用的 AlexNet LPIPS（首次可能下载权重）。
5. 为整个选定验证集准备一次固定 GT PNG、mask PNG 和清单。
6. 两个独立 Python 进程分别加载旧、新模型，评估前 16 张，保存原始 FP32 预测。
7. 使用同一份 V1 指标代码计算预检指标，生成图片。

预检报告：`preflight_report/README.md`。
输入、严格加载、非有限值或 LPIPS 检查失败时不要继续解释实验结果。
预检图片是固定列表前 16 张，包含当前验证顺序中的 sample_013。

## 正式评估及继续运行

确认预检图像、mask 和模型信息正常后：

```bat
python tools/compare_v0_v1.py --device cuda --resume
```

该命令继续完整共同验证集（当前 3000 张），自动复用已经验证完整的预测和
评分缓存。中途 Ctrl+C 或进程停止后，同一命令可以继续。中断发生在单个文件
写入期间时，该文件下次会重新计算。不要同时启动两个相同输出目录的任务。

若希望一次自动完成预检和全量，首次运行也可不带 `--preflight-only`。
两个 GPU 推理进程顺序执行，结束后才启动统一评分，不同时占用 GPU。

不要在比较中途修改源代码、配置、数据列表或替换源 `best.pt`。协议检查发现
变化会报出变化类别，防止将不同实验结果混在一起。修改后使用新输出目录。
checkpoint 快照来自比较启动时的文件，不能用仍在被训练覆盖的权重开启比较。

## 可选参数

```bat
python tools/compare_v0_v1.py --device cuda --batch-size 2 --preflight-only --output-dir C:\codeee\inpainting\comparison_v0_v1_b2
```

若预检 OOM，可从较小 batch 的新输出目录开始。之后继续时保持相同参数：

```bat
python tools/compare_v0_v1.py --device cuda --batch-size 2 --resume --output-dir C:\codeee\inpainting\comparison_v0_v1_b2
```

- `--limit 64 --output-dir ...`：独立的小规模比较，不能冒充完整验证集。
- `--visual-count 16`：固定保存前 N 张图，选择不依赖模型成绩。
- `--preflight-count 16`：预检图像数量；默认 16。
- `--no-lpips`：明确省略 LPIPS，表格不填零。应在首次运行时决定并保持一致。
- `--device cpu`：本地小规模验证用，不建议 CPU 跑完整 CelebA。
- `--old-project/--v1-project`、`--old-checkpoint/--v1-checkpoint`、
  `--old-config/--v1-config`：覆盖默认路径。
- `--val-list`：共同 held-out 子集；须来自两边验证集，使用 V1 列表的图像路径。

不需要重新训练，也不需要改变两个模型的 batch_size 配置；这里的
`--batch-size` 仅控制统一工具的推理批量。

## 输入、指标和公平性

- 共享的预处理使用配置的短边缩放和中心裁剪，默认 286 → 256。
- 优先使用 V1 的 `val_mask_list`；没有时使用 V1 的 mask 参数、验证 seed
  和原验证列表索引，按现有 V1 验证规则生成固定 mask。
- mask 是 1/白色=已知、0/黑色=缺失。两组读取完全相同的预处理 PNG，
  每次推理验证其联合哈希。保存 PNG 前已完成 8-bit 输入变换，与数据集一致。
- 模型使用 FP32，关闭 autocast 和 TF32；采用各自的原始 forward。
- 模型输出按原始 float32 保存为 NPZ；指标计算不经显示 PNG 量化。
- 两组使用**同一份 V1 指标实现**。v0 旧 CSV 的逐图归一化边缘 F1 不参与比较。
- RGB 评价基于保留已知区后的 composite；PSNR 范围为 2（对应 [-1,1]）。
  评分不加显示用截断，PNG 显示会截断到 [-1,1]。
- Hole SSIM 来自实际 SSIM map 在洞内的平均。LPIPS 是整张 composite 的值。
- 边缘尺度 0.1、阈值 0.5；严格匹配与 1 像素容差 F1 分开报告。
  方向角误差以弧度计，在 GT 梯度幅值 >0.05 的 RGB 向量上统计。
- 洞内部定义为距离已知区超过 4 像素（Chebyshev 距离），不是五官检测结果。
- 逐图指标取平均；无支持像素/方向的样本按公共指标规则排除。
- 保留各自 best.pt 的选择规则，记录选中 epoch、预算、参数量与有效 rho；
  不能因测试/验证比较结果不好临时改用另一 checkpoint 而不披露。
- 这次比较涉及数学公式修正、结构模块、训练配置和 PCG 等差异，不能将所有
  改善归因于 Structure Prior。未包含重新训练的 R0/R1 消融或 Nonlocal。

## 结果位置

```text
comparison_v0_v1/
  protocol.json                    完整实验指纹
  protocol/old_model.json           旧模型配置、epoch、参数量和来源
  protocol/v1_model.json            新模型配置、epoch、参数量和有效 rho
  protocol/*_timing.json            推理耗时、缓存数量及显存
  inputs/manifest.json              图像路径、GT/mask 标识及洞比例
  predictions/old/*.npz             旧模型浮点预测
  predictions/v1/*.npz              V1 浮点预测；前 N 张附 Gs/Es
  scores/                          可继续使用的逐图评分缓存
  preflight_report/                预检结果
  report/README.md                  总体比较表
  report/comparison.csv             总体与各 mask 桶的指标和差值
  report/old_per_image.csv          旧模型逐图指标
  report/v1_per_image.csv           新模型逐图指标
  report/per_image_deltas.csv       逐图变化
  report/results.json              汇总、改善/退步/持平样本数
  figures/sample_013/comparison.png 破损 / v0 / V1 / GT
  figures/sample_013/central_crop.png 固定中央区域放大（无五官定位）
  figures/sample_013/edges.png      两组输出边缘 / GT 边缘
  figures/sample_013/structure.png  V1 Es / 从 Gs 得到的边缘 / GT 边缘
```

总体和分桶报告都来自同一个固定样本集合。先检查 40–50%、50–60% 桶和
hole_inner 指标，再结合图片判断五官是否改善。PSNR 报告 dB 增量，不报 dB
的百分比；F1/SSIM 报告绝对差；LPIPS 等误差项提供相对下降百分比。

3000 张 256×256 图像的两组 FP32 RGB 预测约占 4.4 GiB，另有 checkpoint
快照和输入 PNG，建议输出盘留出至少 8 GiB。工具保留这些文件供复查与续跑，
不会自动删除训练输出或比较数据。推理耗时以预检实测为准。

## 本地验证记录

2026-09-10：6 项工具测试通过，覆盖独立进程加载同名包、完整评分与导图、
续跑缓存、数据交集、配置与 checkpoint 不匹配、预测完整性及指标差值方向。
另用实际 v0 和 V1 网络代码、各自阶段/Transformer 配置，在 32×32 合成图像
和随机测试权重上完成了两模型前向、浮点评分及图像导出。显示布局以 256×256
合成输入检查。未运行真实训练权重的完整 CelebA 比较，CUDA 和 LPIPS 需在
GPU 电脑上执行上述预检后确认。
