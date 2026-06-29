# Euler’s Elastica 图像修复模型的 ADMM 推导整理

## 1. 原始 Euler’s Elastica 能量模型

Euler’s elastica 图像修复模型通常写为：

$$
E(u)=\int_{\Omega}
\left(
a+b\left(\nabla\cdot\frac{\nabla u}{|\nabla u|}\right)^2
\right)|\nabla u|\,dx
+
\frac{\eta}{s}\int_{\Gamma}|u-u_0|^s\,dx .
$$

其中：

- $\Omega$：整幅图像区域；
- $\Gamma$：已知像素区域；
- $u$：待恢复图像；
- $u_0$：观测图像；
- $a$：控制边长项的权重；
- $b$：控制曲率项的权重；
- $\eta$：数据保真项权重；
- $s$：保真项指数，常见取值为 $s=2$；
- $\nabla u$：图像梯度；
- $\frac{\nabla u}{|\nabla u|}$：梯度方向的单位向量；
- $\nabla\cdot\frac{\nabla u}{|\nabla u|}$：曲率相关项。

该模型的核心思想是：在缺失区域中，不仅希望图像强度平滑恢复，还希望边缘、轮廓和曲线结构具有较好的几何连续性。

---

## 2. 变量分裂

由于原始能量中包含复杂的非线性曲率项：

$$
\nabla\cdot\frac{\nabla u}{|\nabla u|},
$$

直接优化比较困难，因此引入辅助变量：

$$
p=\nabla u,
$$

$$
n=\frac{\nabla u}{|\nabla u|}.
$$

进一步引入变量 $m$，使得：

$$
|m|\leq 1,
$$

$$
|p|=m\cdot p,
$$

$$
n\approx m.
$$

这些变量的含义如下：

- $p$：代替图像梯度 $\nabla u$；
- $n$：表示单位法向量，近似为 $\frac{\nabla u}{|\nabla u|}$；
- $m$：用于处理单位向量约束；
- $|m|\leq 1$：保证 $m$ 落在单位球内；
- $|p|=m\cdot p$：用于约束 $m$ 与 $p$ 的方向一致；
- $n\approx m$：使法向量变量 $n$ 和约束变量 $m$ 保持一致。

因为有约束 $|m|\leq 1$，所以根据 Cauchy-Schwarz 不等式，有：

$$
m\cdot p \leq |m||p| \leq |p|,
$$

因此：

$$
|p|-m\cdot p\geq 0.
$$

也就是说，$|p|-m\cdot p$ 本身是非负的。正因为如此，对于这一项通常不使用标准的二次 $L^2$ 惩罚，而采用线性惩罚形式。

---

## 3. 增广拉格朗日函数

经过变量分裂后，可以构造如下增广拉格朗日函数：

$$
\begin{aligned}
\mathcal{L}(u,m,p,n;\lambda_1,\lambda_2,\lambda_4)
=&
\int_{\Omega}
\left(
a+b(\nabla\cdot n)^2
\right)|p|\,dx
+
\frac{\eta}{s}\int_{\Gamma}|u-u_0|^s\,dx
\\
&+
r_1\int_{\Omega}(|p|-m\cdot p)\,dx
+
\int_{\Omega}\lambda_1(|p|-m\cdot p)\,dx
\\
&+
\frac{r_2}{2}\int_{\Omega}|p-\nabla u|^2\,dx
+
\int_{\Omega}\lambda_2\cdot(p-\nabla u)\,dx
\\
&+
\frac{r_4}{2}\int_{\Omega}|n-m|^2\,dx
+
\int_{\Omega}\lambda_4\cdot(n-m)\,dx
+
\delta_R(m).
\end{aligned}
$$

其中：

- $\lambda_1$：对应约束 $|p|=m\cdot p$；
- $\lambda_2$：对应约束 $p=\nabla u$；
- $\lambda_4$：对应约束 $n=m$；
- $r_1,r_2,r_4$：正的惩罚参数；
- $\delta_R(m)$：指示函数，用于表示约束 $|m|\leq 1$。

指示函数定义为：

$$
\delta_R(m)=
\begin{cases}
0, & |m|\leq 1,\\
+\infty, & |m|>1.
\end{cases}
$$

> 注意：你截图中有些地方写成 $\lambda_3,r_3$，后续公式又写成 $\lambda_4,r_4$。这里为了统一，全部使用 $\lambda_4,r_4$ 表示 $n=m$ 约束对应的乘子和惩罚参数。

---

# 4. ADMM 交替优化过程

ADMM 的基本思想是：固定其他变量，依次更新 $u,p,m,n$，最后更新拉格朗日乘子。

整体迭代过程为：

$$
u^{k+1}
\rightarrow
p^{k+1}
\rightarrow
m^{k+1}
\rightarrow
n^{k+1}
\rightarrow
\lambda^{k+1}.
$$

下面分别整理每个变量的更新过程。

---

## 4.1 $u$ 子问题

固定 $p,m,n,\lambda_2$，只优化 $u$。

当 $s=2$ 时，$u$ 子问题为：

$$
\mathcal{L}_u(u)
=
\frac{\eta}{2}\int_{\Gamma}(u-u_0)^2\,dx
+
\frac{r_2}{2}\int_{\Omega}|p-\nabla u|^2\,dx
+
\int_{\Omega}\lambda_2\cdot(p-\nabla u)\,dx.
$$

对 $u$ 求变分，可得欧拉-拉格朗日方程：

$$
-r_2\Delta u
+
\nabla\cdot(r_2p+\lambda_2)
+
\eta\chi_{\Gamma}(u-u_0)
=
0.
$$

整理得：

$$
(-r_2\Delta+\eta\chi_{\Gamma})u
=
\eta\chi_{\Gamma}u_0
-
\nabla\cdot(r_2p+\lambda_2).
$$

其中：

- $\Delta u$：拉普拉斯算子；
- $\chi_{\Gamma}$：已知区域 $\Gamma$ 的指示函数。

若采用五点差分格式：

$$
\Delta u_{ij}
\approx
u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{ij},
$$

则可以得到离散更新形式：

$$
u_{ij}
=
\frac{
r_2(u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1})
+
\mathrm{rhs}_{ij}
}{
4r_2+\eta\chi_{ij}
},
$$

其中：

$$
\mathrm{rhs}
=
\eta\chi_{\Gamma}u_0
-
\nabla\cdot(r_2p+\lambda_2).
$$

因此，$u$ 更新本质上是在求解一个带数据保真项的泊松型方程。

---

## 4.2 $p$ 子问题

固定 $u,m,n,\lambda_1,\lambda_2$，只优化 $p$。

$p$ 子问题为：

$$
\begin{aligned}
\mathcal{L}_p(p)
=&
\int_{\Omega}
\left(
a+b(\nabla\cdot n)^2
\right)|p|\,dx
\\
&+
r_1\int_{\Omega}(|p|-m\cdot p)\,dx
+
\int_{\Omega}\lambda_1(|p|-m\cdot p)\,dx
\\
&+
\frac{r_2}{2}\int_{\Omega}|p-\nabla u|^2\,dx
+
\int_{\Omega}\lambda_2\cdot(p-\nabla u)\,dx.
\end{aligned}
$$

将与 $p$ 有关的项整理，可以得到逐点优化问题：

$$
\min_{p(x)}
c(x)|p(x)|
+
\frac{r_2}{2}|p(x)-q(x)|^2,
$$

其中：

$$
c(x)
=
a+b(\nabla\cdot n)^2+r_1+\lambda_1(x),
$$

$$
q(x)
=
\nabla u(x)
+
\frac{r_1+\lambda_1(x)}{r_2}m(x)
-
\frac{1}{r_2}\lambda_2(x).
$$

这是一个典型的向量软阈值问题，其解为：

$$
p(x)
=
\max
\left(
0,
1-\frac{c(x)}{r_2|q(x)|}
\right)q(x).
$$

如果 $|q(x)|=0$，则取：

$$
p(x)=0.
$$

因此，$p$ 更新的作用可以理解为：在当前图像梯度 $\nabla u$ 的基础上，根据曲率权重 $c(x)$ 进行收缩，从而抑制不合理的梯度，同时保留重要结构。

---

## 4.3 $m$ 子问题

固定 $u,p,n,\lambda_1,\lambda_4$，只优化 $m$。

$m$ 子问题为：

$$
\begin{aligned}
\mathcal{L}_m(m)
=&
r_1\int_{\Omega}(|p|-m\cdot p)\,dx
+
\int_{\Omega}\lambda_1(|p|-m\cdot p)\,dx
\\
&+
\frac{r_4}{2}\int_{\Omega}|n-m|^2\,dx
+
\int_{\Omega}\lambda_4\cdot(n-m)\,dx
+
\delta_R(m).
\end{aligned}
$$

去掉与 $m$ 无关的常数项，只保留关于 $m$ 的部分：

$$
\frac{r_4}{2}|m|^2
-
\left(
r_4n+(r_1+\lambda_1)p+\lambda_4
\right)\cdot m
+
\delta_{|m|\leq 1}(m).
$$

配方可得：

$$
\frac{r_4}{2}|m-w|^2+
\delta_{|m|\leq 1}(m),
$$

其中：

$$
w
=
n
+
\frac{r_1+\lambda_1}{r_4}p
+
\frac{1}{r_4}\lambda_4.
$$

因此，$m$ 的更新就是将 $w$ 投影到单位球上：

$$
m(x)=
\begin{cases}
w(x), & |w(x)|\leq 1,\\[6pt]
\dfrac{w(x)}{|w(x)|}, & |w(x)|>1.
\end{cases}
$$

也就是说，$m$ 更新的作用是：在保持方向信息的同时，强制满足 $|m|\leq 1$ 的几何约束。

---

## 4.4 $n$ 子问题

固定 $u,p,m,\lambda_4$，只优化 $n$。

$n$ 子问题为：

$$
\mathcal{L}_n(n)
=
\int_{\Omega}
b(\nabla\cdot n)^2|p|\,dx
+
\frac{r_4}{2}\int_{\Omega}|n-m|^2\,dx
+
\int_{\Omega}\lambda_4\cdot(n-m)\,dx.
$$

令：

$$
\mu(x)=b|p(x)|.
$$

则 $n$ 子问题可以写为：

$$
\mathcal{L}_n(n)
=
\int_{\Omega}
\mu(x)(\nabla\cdot n)^2\,dx
+
\frac{r_4}{2}\int_{\Omega}|n-m|^2\,dx
+
\int_{\Omega}\lambda_4\cdot(n-m)\,dx.
$$

对 $n$ 求变分，可得：

$$
-\nabla\left(\mu(x)\nabla\cdot n(x)\right)
+
\frac{r_4}{2}n(x)
=
\frac{r_4}{2}m(x)
-
\frac{1}{2}\lambda_4(x).
$$

该方程说明：$n$ 的更新本质上是在曲率正则项和 $m$ 的一致性约束之间取得平衡。

其中：

- $\mu(x)=b|p(x)|$；
- 当 $|p|$ 较大时，说明该位置可能存在边缘或结构，曲率约束更强；
- 当 $|p|$ 较小时，曲率项影响较弱。

因此，$n$ 主要负责更新局部几何方向和曲率结构。

---

## 4.5 拉格朗日乘子更新

当 $u,p,m,n$ 都更新完成后，更新拉格朗日乘子。

对应约束分别为：

$$
|p|=m\cdot p,
$$

$$
p=\nabla u,
$$

$$
n=m.
$$

因此乘子更新为：

$$
\lambda_1^{k+1}
=
\lambda_1^k
+
r_1
\left(
|p^{k+1}|-m^{k+1}\cdot p^{k+1}
\right),
$$

$$
\lambda_2^{k+1}
=
\lambda_2^k
+
r_2
\left(
p^{k+1}-\nabla u^{k+1}
\right),
$$

$$
\lambda_4^{k+1}
=
\lambda_4^k
+
r_4
\left(
n^{k+1}-m^{k+1}
\right).
$$

这些更新的作用是逐步加强约束，使得迭代过程中辅助变量最终满足原始变量关系。

---

# 5. 整体 ADMM 迭代流程总结

完整的 ADMM 更新流程可以概括为：

1. **更新 $u$**  
   求解泊松型方程，使恢复图像同时满足数据保真和梯度一致性。

2. **更新 $p$**  
   通过软阈值操作更新梯度变量，控制边缘与结构。

3. **更新 $m$**  
   将方向变量投影到单位球内，保证 $|m|\leq 1$。

4. **更新 $n$**  
   通过曲率相关方程更新法向量，使结构更加平滑和连续。

5. **更新乘子 $\lambda_1,\lambda_2,\lambda_4$**  
   强化约束：

   $$
   |p|=m\cdot p,\quad p=\nabla u,\quad n=m.
   $$

---

# 6. 各变量的直观含义

| 变量 | 数学含义 | 直观解释 |
|---|---|---|
| $u$ | 待恢复图像 | 最终要修复的图像 |
| $p$ | $p=\nabla u$ | 图像梯度变量，描述边缘变化 |
| $n$ | $n\approx \frac{\nabla u}{|\nabla u|}$ | 单位法向量，描述结构方向 |
| $m$ | $|m|\leq 1$ | 受约束的方向变量 |
| $\lambda_1$ | 对应 $|p|=m\cdot p$ | 控制 $p$ 与 $m$ 方向一致 |
| $\lambda_2$ | 对应 $p=\nabla u$ | 控制梯度变量和图像梯度一致 |
| $\lambda_4$ | 对应 $n=m$ | 控制法向量和方向变量一致 |
| $r_1,r_2,r_4$ | 惩罚参数 | 控制各约束的强度 |
| $a$ | 弹性曲线长度项权重 | 控制边缘长度惩罚 |
| $b$ | 曲率项权重 | 控制结构弯曲程度 |
| $\eta$ | 数据保真项权重 | 控制已知区域与原图一致程度 |

---

# 7. 直观理解

Euler’s elastica 模型的目标不是简单地让图像变平滑，而是希望恢复出具有几何连续性的结构。

其中：

- $u$ 负责恢复图像本身；
- $p$ 负责恢复梯度和边缘；
- $n$ 负责恢复曲率和结构方向；
- $m$ 负责保证方向变量满足单位约束；
- 拉格朗日乘子负责让这些辅助变量逐渐满足原始约束。

因此，这个 ADMM 推导可以理解为：

> 把一个复杂的曲率驱动图像修复问题，拆分成多个相对容易求解的子问题，然后通过交替迭代逐步恢复图像结构。

在图像修复任务中，这种方法的优势是具有较强的几何解释性，尤其适合强调边缘连续、曲线延拓和结构一致性的场景。

---

# 8. 和图像修复任务的关系

对于图像修复而言，该模型特别关注以下问题：

- 大块缺失区域中的边缘是否能够自然延拓；
- 结构线条是否连续；
- 曲线是否平滑；
- 修复结果是否符合原图的几何走向；
- 是否能避免简单平滑导致的结构模糊。

因此，Euler’s elastica + ADMM 的方法更偏向于一种具有明确几何意义的结构修复方法。

如果将其与深度学习结合，可以考虑让深度网络参与以下部分：

- 学习更准确的 $p$，即结构梯度；
- 学习更可靠的 $n$，即法向量或方向场；
- 学习自适应的 $c(x)$，即曲率权重图；
- 学习不同区域的惩罚参数，例如 $r_2,r_4$；
- 将 ADMM 的每一步展开成网络中的一个 stage，从而形成算法展开模型。

这样既可以保留 Euler’s elastica 的几何解释性，又可以利用深度网络增强大破损区域的结构预测能力。
