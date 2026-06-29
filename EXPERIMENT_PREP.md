# 跑实验前的准备工作

`tools/prepare_and_run.py` 会替你完成数据列表生成、划分和配置落盘；你只需要先准备原始图片目录，并决定是否使用固定验证/测试 mask。

## 1. 准备数据

- 把所有待划分的原始图像放在一个目录中，允许多层子目录；支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。
- 图片应尽量不小于配置中的 `data.image_size`（默认 256），以免裁剪后尺寸不足。
- 可选：准备验证集和测试集的固定 mask 目录。mask 遵循项目约定：白色为已知区域、黑色为待修复区域。未提供时，验证和测试会使用随机 mask。
- 安装依赖：`python -m pip install -r requirements.txt`。

## 2. 一键准备并运行

在项目根目录执行：

```powershell
python tools/prepare_and_run.py --images D:/datasets/places2 --run
```

这条命令会以固定种子 `42` 将图片按 `80% / 10% / 10%` 划为训练、验证、测试集，生成：

```text
prepared_data/
  train.txt
  val.txt
  test.txt
  experiment.yaml
```

随后自动调用训练，并用 `outputs/<run_name>/best.pt` 在测试集上评估。每一份列表保存的是绝对路径，因此移动项目目录不会让数据路径失效。

若有固定 mask：

```powershell
python tools/prepare_and_run.py `
  --images D:/datasets/images `
  --val-masks D:/datasets/masks/val `
  --test-masks D:/datasets/masks/test `
  --run-name exp_001 `
  --run
```

固定 mask 与样本数不必相同，数据加载器会循环使用 mask；但建议每个目录至少准备若干张，保证评估更稳定。

## 3. 常用变体

仅生成划分、检查 `prepared_data/experiment.yaml` 后再训练：

```powershell
python tools/prepare_and_run.py --images D:/datasets/places2
python train.py --config prepared_data/experiment.yaml
```

更改划分比例和随机种子：

```powershell
python tools/prepare_and_run.py --images D:/datasets/places2 --val-ratio 0.15 --test-ratio 0.15 --seed 2026 --run
```

使用完整模型配置，训练但不自动评估：

```powershell
python tools/prepare_and_run.py --images D:/datasets/places2 --config configs/full.yaml --run --skip-eval
```

## 4. 每次实验前检查

- 保持同一个 `--seed` 和原图目录，才能复现相同划分。
- 为不同实验设置不同的 `--run-name`，避免覆盖输出目录。
- 首次建议只准备列表后，将 `train_limit` / `val_limit` 暂时设小，或使用现有 `smoke_test.py` 检查环境。
- 训练完成后，主要结果在 `outputs/<run_name>/`：`best.pt`、`log.csv`、验证可视化和 `eval_test.json`。
