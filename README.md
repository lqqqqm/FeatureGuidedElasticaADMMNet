# Feature-Guided Elastica ADMM Inpainting

一个可直接运行的 PyTorch 项目骨架，用于实现你这份 **Feature-Guided Euler's Elastica ADMM Unfolding + Transformer bottleneck** 图像修复方案。默认配置采用更稳的 MVP 版：

- `K=3`
- `Transformer depth=2`
- 只默认开启 `u correction`
- `tau_u=0.05`
- 默认先不用 perceptual，方便在离线环境直接起跑

完整增强版配置也提供在 `configs/full.yaml` 中。设计依据来自你上传的规格书，并对训练稳定性做了少量工程化保守调整。

## 1. 项目结构

```text
fg_elastica_project/
  configs/
    mvp.yaml
    full.yaml
  fg_elastica_inpaint/
    data/
    losses/
    models/
    utils/
  tools/
    make_filelist.py
  train.py
  evaluate.py
  infer.py
  smoke_test.py
  requirements.txt
```

## 2. 数据准备

如果你只有一个原始图片目录，推荐先阅读 [EXPERIMENT_PREP.md](EXPERIMENT_PREP.md)，并直接使用 `tools/prepare_and_run.py` 一键完成划分、txt 生成、训练和测试：

```bash
python tools/prepare_and_run.py --images /data/places2 --run
```

约定：

- 图像路径列表放在 `/data/train.txt`、`/data/val.txt`、`/data/test.txt`
- 固定验证/测试 mask 路径列表放在 `/data/val_masks.txt`、`/data/test_masks.txt`
- 每个 txt 文件一行一个绝对路径

生成文件列表示例：

```bash
python tools/make_filelist.py --root /data/places2_train --output /data/train.txt
python tools/make_filelist.py --root /data/places2_val --output /data/val.txt
python tools/make_filelist.py --root /data/places2_test --output /data/test.txt
python tools/make_filelist.py --root /data/fixed_val_masks --output /data/val_masks.txt
python tools/make_filelist.py --root /data/fixed_test_masks --output /data/test_masks.txt
```

## 3. 安装

```bash
cd /mnt/data/fg_elastica_project
pip install -r requirements.txt
```

如果你要开启 perceptual loss 或者 LPIPS，再额外安装：

补充说明：如果你是在 **CPU-only** 环境里跑，这个项目会自动把 PyTorch 线程数压到 1，避免某些环境下出现算子卡住不返回。


```bash
pip install -r requirements_optional.txt
```

## 4. 快速自检

```bash
python smoke_test.py
```

## 5. 训练

MVP：

```bash
python train.py --config configs/mvp.yaml
```

完整版：

```bash
python train.py --config configs/full.yaml
```

## 6. 评估

```bash
python evaluate.py --config configs/mvp.yaml --checkpoint outputs/fg_elastica_mvp/best.pt --split test
```

## 7. 单图推理

注意：默认约定 **mask 白色=known，黑色=hole**。如果你的 mask 刚好相反，就加 `--invert_mask`。

```bash
python infer.py \
  --config configs/mvp.yaml \
  --checkpoint outputs/fg_elastica_mvp/best.pt \
  --image /data/example.png \
  --mask /data/example_mask.png \
  --output_dir ./infer_out
```

## 8. 当前代码和原规格书的差异

为了先保证“能跑 + 稳定”，默认配置做了这些保守处理：

1. `MVP` 默认只开 `u correction`
2. `tau_u` 默认用 `0.05`
3. `lambda1` 仍然会更新并保存在 `aux` 里，但当前主更新仍主要由 `u/p/m/n + lambda2/lambda4` 驱动
4. `perceptual loss` 默认关闭，避免离线环境下载 VGG 权重时报错

## 9. 建议的调试顺序

1. 先跑 `python smoke_test.py`
2. 再把 `configs/mvp.yaml` 中的 `train_limit` 和 `val_limit` 改成很小值，例如 `64` / `16`
3. 确认训练无 NaN 后，再切到完整数据和 `full.yaml`
4. 最后再考虑把 `loss.use_perceptual=true`

## 10. 输出内容

训练时会在 `outputs/<run_name>/` 下保存：

- `best.pt`
- `last.pt`
- `epoch_xxxx.pt`
- `config_resolved.yaml`
- `log.csv`
- `val_epoch_xxxx.png`

验证图默认是三联图：`masked | pred | gt`
