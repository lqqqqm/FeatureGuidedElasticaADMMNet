# V1 一键训练与评估

入口为 `tools/run_structure_v1.py`。它调用项目现有的 `train.py` 和 `evaluate.py`，
按顺序完成每组训练及测试集评估，默认选择 `best.pt`，不修改模型和 ADMM 公式。

## 在 GPU 电脑运行

先激活已安装 CUDA PyTorch 和项目依赖的 Python 环境。确认三份
`configs/structure_v1_r0.yaml`、`r1.yaml`、`r2.yaml` 中的数据路径有效，然后运行：

```powershell
python tools/run_structure_v1.py --runs r0 r1 r2
```

只运行完整 V1：

```powershell
python tools/run_structure_v1.py --runs r2
```

也可以统一从命令行传入另一台电脑上的文件列表，无须修改三份源配置：

```powershell
python tools/run_structure_v1.py --runs r0 r1 r2 --train-list D:/datasets/celeba/train.txt --val-list D:/datasets/celeba/val.txt --test-list D:/datasets/celeba/test.txt --output-dir outputs/celeba_v1
```

三个 txt 使用 UTF-8 无 BOM 编码，每行一个完整图片绝对路径（不能用 `~`），
且训练/验证/测试图片互不重叠，以匹配现有数据读取器。
已有划分应保持不变。脚本检查列表中的文件是否存在，不读取图片内容做重新划分。
固定 mask 列表可用 `--val-mask-list` 和 `--test-mask-list` 统一指定；否则保留
模板中的 null，使用模型项目已有的固定验证/测试 mask 生成方式。

默认要求 CUDA 可用，检查不通过会退出，避免误在 CPU 上进行长训练。
`--device cpu` 仅供明确需要的本地小规模检查。脚本不自动安装依赖。
LPIPS 按现有配置启用，若不可用会保留 `lpips_available=0`，不会伪造 LPIPS=0。

脚本可通过绝对路径从其他目录调用；相对参数路径统一以项目根目录为基准。
子进程沿用启动脚本的 Python 环境，路径带空格也通过参数列表传递。

## 先查看命令

```powershell
python tools/run_structure_v1.py --runs r0 r1 r2 --dry-run
```

`--dry-run` 显示数据路径、生成配置的位置以及训练/评估命令，不写文件，
不访问数据集或 GPU，也不启动任何训练进程；它不是数据有效性检查。

## 统一改变实验预算

```powershell
python tools/run_structure_v1.py --runs r0 r1 r2 --batch-size 4 --epochs 40 --num-workers 4
```

`--batch-size` 同时设置 train/val/test batch。三组沿用各自模板的结构头和 rho，
其余模型、求解、数据和训练配置必须一致；R1/R2 结构监督权重也必须一致。
所有修改只写入输出目录中的配置副本，源 YAML 保持原样。
自定义模板目录可通过 `--config-dir` 指定，文件名仍为 `structure_v1_r0.yaml` 等。

## 断点恢复

在原命令后追加 `--resume`，保留原来的数据、输出目录、batch、epochs 等参数：

```powershell
python tools/run_structure_v1.py --runs r0 r1 r2 --resume
```

如果首次使用了 `--train-list`、`--output-dir` 等参数，恢复时也要提供相同参数。
已完成且 checkpoint 未变、逐图结果快照仍存在的组会跳过；未完成的组从
`last.pt` 恢复，随后评估。缓存命中时，通用评估 JSON 和逐图 CSV 会恢复为
当前所选 checkpoint 的结果；缺失逐图快照时会重新评估。
恢复粒度沿用训练器的 epoch checkpoint，尚未保存的当前 epoch 会重新执行。
若失败发生在第一次 checkpoint 之前，该组会从头开始。

脚本核对保存的配置和数据列表哈希，拒绝在旧目录中混用更改后的实验设置。
需要改变训练轮数或数据划分时，选一个新的 `--output-dir`，避免恢复后学习率
调度预算与原 checkpoint 不一致。仅更换评估用 checkpoint 允许复用原训练结果。

每个输出根目录有操作系统文件锁，阻止两个脚本同时写同一组实验。
进程退出后锁自动释放，保留的 `.runner.lock` 文件本身不表示仍在运行。
脚本只接管它创建的实验目录；原来手动运行 `train.py` 的目录，继续用原入口恢复。

## 仅评估

```powershell
python tools/run_structure_v1.py --runs r0 r1 r2 --eval-only
```

使用结构指标最优的 checkpoint：

```powershell
python tools/run_structure_v1.py --runs r0 r1 r2 --eval-only --checkpoint best_structure.pt
```

仍需提供首次使用的路径与配置覆盖参数。`--eval-only` 会实际重新评估，不使用缓存。
三组比较应选择相同 checkpoint 规则，避免按测试集成绩挑选模型。

训练或评估失败会返回非零退出码，保存日志和状态，停止后续组。重新评估时，
旧的通用 `eval_test.json` 及逐图 CSV 先另存为 `eval_test_previous.json` 和
`eval_test_previous_per_image.csv`；只有新生成、具有有效核心指标及逐图 CSV
的结果才会写入本次汇总，失败不会混入旧指标。

## 可选数值预检

```powershell
python tools/run_structure_v1.py --runs r0 r1 r2 --preflight --preflight-samples 32
```

预检在任何组训练前执行一次，生成 `preflight.json` 和 `preflight.log`；
选中 R2 时使用其配置，否则使用首个选中组。预检命令失败会停止整个流程。
它仅保存参考报告，不会自动采用推荐 rho、edge scale 或求解预算，也不把
预检进程正常退出解释为当前参数已通过数值精度要求。
使用该开关时，报告生成后会继续训练；恢复命令若仍带该开关，也会重新预检。
若需要先读报告再调整参数，应先单独运行 `tools/preflight_structure_v1.py`，
完成训练集标定后再启动训练入口，详见 [V1 说明](structure_prior_v1.md)。

## 输出

默认输出目录为 `outputs/structure_v1/`：

```text
outputs/structure_v1/
  summary.csv
  summary.json
  r0/
    runner_config.yaml
    runner_state.json
    train.log
    evaluate.log
    best.pt / best_structure.pt / last.pt
    eval_test.json
    eval_test_best.json
    eval_test_best_per_image.csv
    eval_test_per_image.csv
    log.csv / bucket_metrics.csv
    diagnostics/ / per_image/
  r1/
  r2/
```

`summary.csv` 汇总每组状态、checkpoint 和现有评估器返回的所有指标，包括整体及
洞内 PSNR/SSIM、边缘、梯度、方向和先验指标。`summary.json` 保留对应结构化结果。
缺失 LPIPS 不补零；NaN/Inf 的可选指标转换为 null（CSV 中为空），失败组有明确状态。
每组完整训练日志及固定样本图仍由原训练器生成。不同 checkpoint 的汇总 JSON
和逐图 CSV 单独保存，例如 `eval_test_best_structure.json` 和
`eval_test_best_structure_per_image.csv`；成功完成或恢复后，通用文件对应当前
所选 checkpoint。

本地验证：

```powershell
python -m unittest discover -s tests -p test_structure_runner.py -v
```

其中包含一次临时合成数据上的 CPU 32×32 短训练和真实测试集评估，输出会自动清理。
该检查不替代 4090 上的 AMP、显存和完整 CelebA 实验验证。
