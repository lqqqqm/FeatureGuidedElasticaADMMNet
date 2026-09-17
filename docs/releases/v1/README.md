# v1：第一阶段结构先验模型基线

冻结日期：2026-09-17。Git 的 annotated tag `v1` 标识本次整理后的完整源码提交；`VERSION` 为 `v1`。

## 本版本包含什么

- 当前 Euler's Elastica / ADMM 展开模型，Learned Structure Prior（Gs/Es），共享网络，默认 K=3，u Correction 与最终 PCG。
- 服务器已采用的 SSIM 评分精度修正及其测试。只在评分局部限制精度，不改变模型前向。
- 服务器已采用的续训学习率重置入口 `--reset-scheduler` 与 `optim.schedule_start_epoch` 校验；普通恢复保持原调度。
- 本地已有的模型说明注释、历史 v0/V1 比较工具与测试；这些文件与服务器副本一致。
- `configs/metax_ablation_20260915/r0.yaml`、`r1.yaml`、`r2.yaml`：EXP-006 的冻结配置副本。

没有实现 EXP-007 的 K=6 对照或 EXP-008 的逐轮独立算法系数；没有把 EXP-010 低频约束并入正式网络；没有新增第二阶段。

## 如何理解“版本”

`v1` 是代码基线，R0/R1/R2 是配置变体，EXP 编号是实验问题，checkpoint 是某次训练的权重。一个版本可以对应多个配置和多份权重，不能仅靠文件名 `best.pt` 判断出处。

本次提交整理的是 2026-09-17 核对后的可复核工作版本。它不是对历史训练全部过程的源码追认：历史 R2 和 EXP-006 的冻结源码、原始配置及权重仍各自保留。历史 SSIM 日志也不会因代码修正自动变成校准指标。

## 配置与运行边界

| 配置 | 用途 | 状态 |
|---|---|---|
| `configs/structure_v1_r0.yaml`～`r2.yaml` | 原有 V1 配置及历史运行工具 | 保留，不改写为 MetaX 协议 |
| `configs/metax_ablation_20260915/r0.yaml` | EXP-006 R0：batch16/FP32/40轮 | 已完成，最佳 PSNR e39 |
| `configs/metax_ablation_20260915/r1.yaml` | 相同协议，结构监督但 rho=0 | 已准备，未启动 |
| `configs/metax_ablation_20260915/r2.yaml` | 相同协议，结构监督与 rho→16 | 已准备，未启动 |

MetaX 配置含服务器路径和历史 run_name，是冻结记录。未来启动新实验前，应复制成新的实验配置，使用新的 run_name/output_dir；不直接覆盖原 EXP-006 输出。运行工具不会因为创建 Git 分支自动选择这些配置。

历史 R2 的前 40 轮 runner 配置来自已有台账快照，续训 41～60 轮的 resolved 配置从服务器重新读取，均保存于本目录的 `configs/`；来源与哈希见权重索引。前者保留原 Windows 路径，不能直接当作当前服务器启动配置。[EXP-006 原始协议](exp006_protocol.json)同时记录数据清单哈希、掩膜种子和训练预算。

## 两端核对结果

核对范围为根目录源码/说明/依赖文件，以及 `fg_elastica_inpaint`、`configs`、`docs`、`tests`、`tools`、`evaluation`；未枚举数据集、权重内容、环境或大批诊断输出。

- 修改前：68 个文件逐字节一致，2 个文件不同，4 个仅服务器存在，无仅本地存在的所选源码文件。
- 两处代码差异：`train.py` 的续训调度支持；`fg_elastica_inpaint/utils/metrics.py` 的 SSIM 修正。
- 四个服务器新增文件：3 份 MetaX 配置和 `tests/test_ssim_precision.py`。
- 将上述 6 个文件按服务器副本同步到本地；原本地文件与 Git 差异已备份。
- 服务器当前源码与 EXP-006 冻结源码的 63 个共有文件逐字节一致。
- 服务器有 `.git` 元数据，但没有可用 Git 命令；本次未安装 Git、未修改其工作源码或历史输出。

详细哈希见 [source_comparison.json](source_comparison.json)，权重位置与哈希见 [checkpoints.json](checkpoints.json)，服务器环境见 [environment_server.json](environment_server.json)，冻结检查见 [verification.json](verification.json)。

原始审计和本地修改前备份位于项目的 `analysis/version_freeze_20260917`，不纳入 Git。Git 不保存大型权重、诊断 NPZ/图片或缓存；旧压缩包已在整理前被删除，本次提交记录这一删除，不从历史中清除它。

## 后续开发规则

1. `v1` 标签固定保留。新工作在 `codex/v1_1` 分支上开展；建立分支不表示新结构已实现。
2. 先 EXP-007，单独检查展开深度；再在同一 K 下做 EXP-008。两个实验分别提交，分别记录配置与结果。
3. 需要同协议 K=3/R2 对照；历史混合训练 R2 不替代公平基线。若引入可学习与逐轮独立两个因素，保留可学习共享对照。
4. 启动正式实验前先提交代码，记录完整 commit、配置、数据/掩膜协议、恢复点、预算；服务器使用该提交的独立源码副本。
5. 所选实现与检查完成后再设置 `v1_1` 标签；版本编号不保证效果优于 `v1`。

本次只核对、冻结和建立开发分支。实验 7/8 未启动；没有推送 GitHub。

外部实验总账：`D:/code/inpainting/experiment-ledger`。本地 Git 管代码，服务器保留权重与运行产物，台账连接两者。
