# Learned Structure Prior V1

## Approved scope

CelebA, 256x256, batch 4. Reuse the existing backbone and Correction modules;
no Nonlocal, GAN, diffusion, extra Transformer, or RGB structure generator.
The original mathematical reference is `../eulers_elastica_admm_derivation.md`.
This document defines an explicit extension of that energy, not a replacement.

## Energy and interfaces

M=1 denotes known pixels; H=1-M denotes holes. The structure decoder runs once
per input on F1 (64,H,W), F2 (128,H/2,W/2), F3 (256,H/4,W/4), and M.
It returns `gradient` (B,6,H,W), `edge_logits` (B,1,H,W), `edge` (sigmoid),
and `gradient_pyramid` ordered full, half, quarter. Gradient ordering is
[dx_R,dx_G,dx_B,dy_R,dy_G,dy_B]. Direction is derived from the gradient.

The new energy term is rho/2 * sum(H * |p-G|^2), summed per RGB vector.
c=a+b*div(n)^2+r1+lambda1 and q=grad(u)+(r1+lambda1)*m/r2-lambda2/r2
remain unchanged. With omega=rho*H, the update is
`p=shrink((r2*q+omega*G)/(r2+omega), c/(r2+omega))`.
m/n/dual equations remain those of the reference. rho=0 restores baseline.
rho is constant across stages, with a training-only warmup/ramp multiplier.

The final output is a differentiable PCG u readout using p and lambda2 after
the final stage. It solves A u = eta*M*I_m-div(r2*p+lambda2),
A=-r2*div(grad)+eta*M, with boundary-correct Jacobi diagonal. No Correction
follows readout. No-known-pixel samples are invalid for this Neumann solve.
Use FP32 ADMM under autocast, bounded n GD step from the weighted operator
norm, finite convergence handling, and report actual PCG residuals.

## Decoder

Project F3/F2/F1 to 64/32/32 channels using 1x1 convolutions. Quarter block:
65->64; half block:97->64; full block:97->32. Each block is the existing
two-convolution GroupNorm/GELU ConvBlock. Concatenate resized M at each level.
Full 3x3 heads emit six gradient and one edge-logit channels; two auxiliary
1x1 heads emit six gradient channels at half/quarter resolution. Gradients
are bounded componentwise with 2*tanh. No independent confidence or normal
head. Structure features do not enter Correction directly.

## Shared target contract

`fg_elastica_inpaint/utils/structure.py` provides:
- `edge_strength(gradient, scale=0.1)`: 1-exp(-mean_RGB(|gradient|)/scale),
  exactly zero for a constant field, using the project's vector convention.
- `structure_targets(gt, sizes, edge_scale=0.1)`: optional convenience;
  targets must be gradients of antialiased resized GT, not resized gradients.
GT is used only in losses/evaluation/oracle diagnostics, after augmentation.

Network output preserves pred/comp/stage_preds/aux and adds:
- `structure`: decoder dictionary or None;
- `readout`: dictionary with u before readout and relative residual;
- `diagnostics`: detached small per-stage scalars by default;
- `stage_states`: full tensors only when explicitly requested for tracing.

Loss constructor additions: lambda_structure_grad, lambda_structure_edge,
lambda_structure_orientation, lambda_structure_consistency, edge_scale,
orientation_threshold, structure_scale_weights, structure_known_weight.
Loss result includes existing keys plus structure_grad, structure_edge,
structure_orientation, structure_consistency, readout. `from_config(cfg)` is
the single constructor route for train/evaluate tools. Pred is readout output;
stage_preds remain the K original stage u values.

## Execution checklist

- [x] Core model: decoder, explicit proximal update, stable weighted n step,
  PCG readout, structure/diagnostic outputs, meaningful operator/gradient tests.
- [x] Loss: balanced multiscale signed gradient loss, edge BCE+Dice,
  strong-GT-edge direction loss, E/G consistency, old edge-target repair.
- [x] Data/evaluation: enforce requested mask bucket; hole, structure,
  orientation, tolerance-edge and distance metrics with fixed edge scale.
- [x] Training/experiments: rho schedule, loss logging, fixed diagnostic
  examples/raw tensors, checkpoints, honest LPIPS availability and GPU memory;
  matched R0/R1/R2 configs; calibration/oracle preflight and interventions.
- [x] Trace uses the actual model forward rather than another ADMM copy.
- [x] Verify CPU formula/dense-solve/backward checks, R0/R1/R2 forwards and
  small-batch optimizer steps, trainer/evaluator integration, clean diff.

## Experiment contract

All three runs use depth 2, K=3, u Correction only, the same readout, data,
mask distribution, optimization budget, and seed. R0: no decoder. R1:
decoder and supervision with rho=0. R2: same decoder/supervision with rho>0.
rho and edge scale are provisional until training-only calibration; do not
claim empirical optimality. Store G/E targets and predictions, shrink survival,
injection delta, primal residuals, readout residuals, correction changes,
image/mask ids, and distance-stratified quality. Oracle/shuffle/disabled-prior
and disabled-Correction are inference diagnostics, not retrained ablations.

## Work log

2026-09-07: User approved implementation. Work directly on existing clean
`improve-stage1-structure` branch. No commits/pushes or long GPU runs requested.
Independent loss and data/metric edits are delegated under the parallel-agent
skill; model, training integration, scripts, and final verification stay with
the primary agent. No shared-file edits between workers.

## V1 使用方法（CelebA / 4090）

三份配置都是 256×256、batch=4、K=3、Transformer depth=2、仅 u Correction。
R0 不建结构头，R1 建结构头并显式监督但 rho=0，R2 与 R1 相同且 rho>0。
共同权重初始化、每个 epoch 的数据随机种子、数据分布和求解预算相同。
R0/R1/R2 的差异只在结构头、相应监督和显式耦合，不能用旧 full 配置充当 R0。

1. 修改三份配置的 `data.train_list/val_list/test_list` 为 GPU 机器的文件列表。
   每行用绝对图片路径；训练、验证、测试集须分开。三份使用相同列表。
   默认 val/test mask 列表为 null，以固定 seed+样本序号生成稳定 mask；
   如已有固定列表，三份配置填写相同列表。验证 mask 不消耗训练随机状态。
2. 在训练集上运行一次数值预检。它只前向计算，不训练模型，也不会自动修改配置：

```bash
python tools/preflight_structure_v1.py --config configs/structure_v1_r2.yaml --samples 32 --device cuda --output outputs/v1_preflight.json
```

预检保存 GT 强梯度分位数、实际初始 q/c 下的 rho 激活阈值、关闭注入与 GT
oracle 的结果，以及不同前向/反向 PCG 预算与高预算参考解的差异。
GT oracle 仅用于诊断，随机权重下的好坏不代表训练效果。报告不得使用测试集标定。
只有明确指定 `--synthetic` 才使用合成图，用于本地自检：

```bash
python tools/preflight_structure_v1.py --config configs/structure_v1_r2.yaml --samples 2 --device cpu --synthetic --synthetic-size 32 --output outputs/v1_synthetic_check.json
```

默认 rho=16、edge_scale=0.1 是起始值，并非已在 CelebA 上验证的最优值。
参考报告调整 rho 后，只改 R2 的 rho；调整 edge_scale、方向阈值或 PCG 预算时，
三份配置同步修改。R1/R2 的结构监督权重必须一致。报告中的 null 推荐表示该
参考检查未通过，不能把它当作默认小预算可用。还需确保预检覆盖足够的 45–60%
连续大洞；样本不足时增加 `--samples`，而不是根据测试成绩调参。

3. 依次完成三次等预算训练，默认 40 epochs，关闭早停：

```bash
python train.py --config configs/structure_v1_r0.yaml
python train.py --config configs/structure_v1_r1.yaml
python train.py --config configs/structure_v1_r2.yaml
```

前 2 个 epoch 结构头只受显式监督/共享特征训练，随后 3 个 epoch 线性增大
耦合，之后固定 rho。三个配置共用此时间表。checkpoint 保存验证时的实际
rho_scale；evaluate/infer/trace 复用该值，避免早期 checkpoint 被换成另一种耦合。
结构头不会把 G/E 拼接给 Correction；已有 Backbone 和 Correction 网络体保持原样。

4. 用每个运行的 `config_resolved.yaml` 和同一选择规则评估测试集：

```bash
python evaluate.py --config outputs/structure_v1_r2/config_resolved.yaml --checkpoint outputs/structure_v1_r2/best.pt --split test
```

R0/R1 重复同一命令并替换目录。`best.pt` 按验证 hole_psnr 选择；
`best_structure.pt` 按验证 hole_edge_f1 选择，可作为第二组事先约定的报告，
不要按测试结果在二者之间选择。LPIPS 需要 optional dependencies 和相应权重；
未能加载时输出 availability=0，不把缺失值当成改善。LPIPS 始终计算整张
completed 图，没有伪称的“hole LPIPS”。

## 求解精度和计算预算

V1 默认 readout 前向最多 512 次、反向最多 640 次、相对停止容差 1e-7。
实际可更早停止。选择这些保守预算源于本地 256×256 连续 56% 洞的数值检查：
80 次求解的整图相对残差约 1e-4，但洞内与高预算解的 L1 差异仍约 0.04
（图像范围 [-1,1]）；不能只看整图残差就断言已收敛。
预检可在训练数据上给出更合适的预算。比较不同预算时关注洞内解差异以及
伴随解差异。仍然是有限求解，不能宣称精确收敛的 ADMM 或全局最优解。

PCG 用隐式伴随反传：A^T v=dL/du，A 是原 u 方程的对称算子。
它近似的是方程解的导数，而不是有限次 CG 程序控制流的导数；求解初值本身
不承担导数路径。前向/反向预算不足都会影响准确性，因此训练日志保存
`readout_relative_residual*` 与 `readout_backward_relative_residual*`。
停止条件相对右端项范数，不使用会吞掉小 loss 梯度的常数绝对阈值。

FP32 中，递推残差达到停止条件不保证重新代入方程的实际残差也达到 1e-7；
大洞伴随解的二阶差分对舍入尤其敏感。预检因此采用 FP64 高预算参考，除了
记录 FP32 实际残差，还比较伴随解及直接决定 p 反向梯度的 grad(伴随解)。
不能把较小的递推残差当作精度证明，也不因二阶残差单独升高就宣称 p 梯度失效。
数值容差是停止目标；生产前向/反向实际误差以报告为准。

新增结构头 230,611 个参数；V1 总可训练参数 19,150,522，增加约 1.22%
（相对无头模型）。AdamW 的权重、梯度和两个动量约多 3.52 MiB；主要新增显存
来自全分辨率解码激活。ADMM/PCG 使用 FP32，语义 Backbone/结构头可在 CUDA 下
autocast。隐式反传避免保存数百次 PCG 迭代图，但不能据此保证 batch=4 的总显存。
RTX4090 按独立显存预算，系统“共享 GPU 内存”不作为 CUDA 显存余量。
训练会保存 peak_allocated_gib、peak_reserved_gib、耗时和设备信息。

## 如何判断先验是否有效

每个 epoch 保存前 16 个固定验证样本；第 1 个和每 5 个 epoch 保存原始状态。
PNG 包含 masked/completed/GT、G/E、GT edge、带符号梯度和方向显示。
梯度显示范围固定 [-2,2]，edge 固定 [0,1]；不逐图拉伸弱边缘。
方向色图是 RGB 平均梯度的辅助视图，定量方向误差仍逐 RGB 向量计算。
raw_tensors 包括 q、c、阈值、无先验 p、p/m/n、乘子、曲率、Correction 改变量。
日志包含每阶段非零 p 比例、实际 p 注入差异、约束残差和结构头梯度范数。
结构梯度/边缘/方向指标同时评估结构头与最终 completed 图。

每图 CSV 保存图像路径、mask 哈希、洞面积、hole/boundary/inner 支持像素数。
边界层采用到已知区的 Chebyshev 距离 <=4 像素，其余为内部；空区域不计入
相应指标均值。洞内 SSIM 是实际 SSIM map 的洞内均值，不是洞图抹黑后整体 SSIM。
方向只在强 GT 梯度上计分，保留有效向量数量；容差 Edge F1 与严格 F1 分开报告。

训练后可对固定图做五种干预：预测先验、关掉注入、固定 seed 空间乱序、GT
梯度 oracle、关掉 Correction。它们是同一权重的推理诊断，不能替代 R0/R1/R2：

```bash
python tools/diagnose_structure_prior.py --config outputs/structure_v1_r2/config_resolved.yaml --checkpoint outputs/structure_v1_r2/best.pt --image /data/fixed_face.png --mask /data/fixed_mask.png --output_dir outputs/v1_interventions
python trace_infer.py --config outputs/structure_v1_r2/config_resolved.yaml --checkpoint outputs/structure_v1_r2/best.pt --image /data/fixed_face.png --mask /data/fixed_mask.png --output_dir outputs/v1_trace
```

干预工具使用完整配置 rho，适用于 ramp 完成后的 checkpoint；GT oracle 和
空间乱序都会在输出中明确标记。空间乱序保持六个梯度分量成组，破坏空间布局。
普通 trace 直接调用真实 forward，没有独立复制的 ADMM 实现。

本地验证命令：`python smoke_test.py` 和 `python -m unittest discover -s tests -v`。
这些检查公式、梯度、shape、短训练与工具接口，不证明 CelebA 修复质量。

## 本次验证记录

2026-09-07，PyTorch 2.11 CPU：基础 smoke 通过；62 项 unittest 全部通过；
46 个 Python 文件语法解析通过，git diff --check 通过。
测试覆盖 p 闭式解/零耦合恢复、PCG 稠密参考与梯度缩放、损失/指标、真实
forward trace、预检/干预/独立评估 CLI、固定 mask 和短训练集成。

32×32、batch=2、K=3 的 V1 连续优化 3 步，无 NaN/Inf/shape error，结构头
参数更新、实际 p 注入以及独立的 RGB→readout→p→G 反向路径均已确认。
256×256、batch=1、56.25% 连续洞的完整 forward/backward 也通过；最终配置
前向实际相对残差约 5.36e-7，洞内与 1024 次参考解 L1 差异约 1.06e-5，
反向实际相对残差约 2.35e-4，所有参数梯度有限。以上输入为合成数据。
RTX4090 的 CUDA AMP、256×256 batch=4 峰值显存和 CelebA 训练效果尚未实测。

原 Structure Head 已替换，旧结构头 checkpoint 不能直接严格加载为 V1。
匹配对照应从相同 seed 开始；代码未提交 commit，也未启动完整 GPU 训练。
