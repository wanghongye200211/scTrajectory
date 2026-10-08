# scTrajectory：重构轨迹，再冻结模型验证速度

这是一套可独立运行的 PyTorch 代码，承接本项目的 unbalanced Stitching 实现。**第一阶段联合拟合轨迹、质量和动力学场；第二阶段冻结所有参数，用速度与生长场重新积分。** 它不是先拟合任意曲线，再把曲线求导当作独立速度验证。

核心物理模块是现有 `ustitch/src` 的逐字节快照，见 `SOURCE_MANIFEST.json`。代码不依赖旧目录、其他项目或指定硬盘。默认 MPS float32；CPU float64 用于统计与精确离散 OT。`--device cpu` 仅在显式指定时启用。这里保留我们独立的反应–输运动力学研究线。

## 与论文 Stitching 的对应关系

[Stitching 官方代码](https://github.com/BasisResearch/wasserstein-residuals)采用可优化粒子轨迹、KDE 密度和梯度流速度残差。

| 部分 | 这里的实现 |
|---|---|
| 可训练粒子轨迹 + KDE 密度 | 保留 |
| 势能 + 熵 + 对称相互作用产生速度 | 保留；相互作用使用粒子中心求积 |
| 轨迹诱导场与模型场的残差 | 保留；同时包含中心残差与 KDE 查询点残差 |
| 非守恒质量和独立局部生长 | 本项目扩展，不能称作论文原版 |
| 逐段 OT 初始化 | 初始化先验，不是已观测的细胞谱系 |
| 质量残差、ESS、轨迹支持约束 | 本项目软约束 |
| `baseline` | 自由局部生长的基本配置，不带额外 OT 速度教师/Sinkhorn/端点损失 |
| `time_rollout` | 添加 OT 位移监督、Sinkhorn、端点积分损失及平滑时间特征；是增强配置，不是严格原版复现 |

新训练入口保留同一物理实现和配置结构，但并非逐随机数重放历史训练脚本；OT 教师库和可调粒子数等接口有简化。已有 `shape_losses` 检查点可以无损导入。软件版本 1.0 不提供其他历史模型族的通用转换器。

## 方程及两种“速度”

连续模型目标为

\[
\partial_t\rho+\nabla\cdot(\rho v)=g\rho,\quad
v=-\nabla V_\theta-D\nabla\log\rho-\nabla(W_\theta*\rho),
\quad g=g_\psi(t,x).
\]

`g` 是独立的异质局部生长场，不设成 `-ημ`。输入计数约束各可见时间点的总质量；不能由总计数唯一确定每个细胞的生长率。默认质量为 `N(t)/N(t0)`，计数默认是捕获细胞数代理。

令固定带宽 KDE 为

\[
\rho_t(x)=\sum_i a_i(t)K_h(x-z_i(t)),\quad
R_i(t,x)=\frac{a_iK_h(x-z_i)}{\rho_t(x)}.
\]

轨迹诱导的欧拉场为

\[
u_\rho(t,x)=\sum_iR_i\dot z_i,\qquad
r_\rho(t,x)=\sum_iR_i\frac{d\log a_i}{dt}.
\]

在带宽固定、轨迹与 log 质量分段可微时，它们满足
`∂tρ + div(ρuρ) = rρρ`。代码的 KDE score 是解析的，无须另外训练 score matching 网络。

**粒子中心导数 `żᵢ`、混合密度诱导速度 `uρ(x)`、模型速度 `vθ(x;ρ)` 是三个不同的量。** 导出文件把它们分别保存，残差检验比较 `uρ` 与 `vθ`。中心导数仅在每段内部定义；节点处不冒充光滑导数。相互作用数值项为 `Σⱼ aⱼ ∇Wθ(x−zⱼ)`，近似连续卷积。

冻结验证时，从起点解

\[
\dot X_i=v_\theta(t,X_i;\widehat\rho_t),\qquad
\dot\ell_i=g_\psi(t,X_i),\qquad a_i=e^{\ell_i}.
\]

每个 Heun 步都用当前积分粒子重算密度和相互作用；不调用拟合的未来轨迹作驱动、不按时间点重置。这是中心粒子 KDE 近似，不等于对原高斯混合 PDE 的精确求解，因此轨迹偏差同时涉及拟合残差及表示/离散化差异。

## 环境

已在 Python 3.9.25、PyTorch 2.8.0、macOS arm64 的 MPS 上测试。

```bash
python -m pip install -r requirements.txt
export PYTHON=python
bash run.sh --help
```

默认关闭 MPS CPU fallback，设备不可用会报错。GeomLoss 导入时可能输出 KeOps/CUDA/OpenMP 探测信息；本实现使用 tensorized Sinkhorn，未调用 KeOps 内核。统计和画图在 CPU 上是显式设计。

## 输入格式和留出规则

输入 NPZ：`X[N,d]`（固定坐标）、`time[N]`（物理时间）。可选 `count_times[T]` 和 `counts[T]`。未给计数时使用划分前各可见时间点的捕获细胞数。模型不在这里拟合 AE/PCA；严格留时验证需要用户提供由训练数据拟合的表示。

```python
np.savez_compressed('input.npz', X=latent_coordinates, time=physical_times)
```

`prepare` 先去掉整时间点，再在可见时间点内分训练/验证细胞，保存为 `visible.npz` 和 `test.npz`。训练函数只读前者。计数曲线、带宽、OT 初始化、教师配对及模型选择均不使用 test 的细胞或计数。物理时间映射写入输入；不重新标准化特征。

当前验证支持**两端之间的留出时间点插值**，不把外推混成插值。本工具不是原始 GEO 下载和转录组 QC/AE 预处理器；GSE75748 示例直接使用项目已有的封存 AE10 表示。

## 最小完整运行

在代码包目录运行，`OUTPUT_ROOT` 换成自己的输出目录。下面是 20 步软件测试，不是充分训练的质量结论。

```bash
export PYTHON=python
OUTPUT_ROOT=/absolute/path/to/stitching_output
bash run.sh demo --output "$OUTPUT_ROOT/data/raw_demo.npz"
bash run.sh prepare --input "$OUTPUT_ROOT/data/raw_demo.npz" \
  --output "$OUTPUT_ROOT/data/demo" --holdout 0.5
bash run.sh all --data "$OUTPUT_ROOT/data/demo" --output-root "$OUTPUT_ROOT" \
  --run demo_baseline --arm baseline --steps 20 --particles 16 --nodes 9 \
  --config configs/smoke.json --device mps
```

换自己的输入时，仅替换 `prepare` 的 NPZ 和物理留出时间。常规初始配置：`--steps 4000 --particles 96 --nodes 33`，不加 smoke config。正式比较需预先固定训练预算、随机种子、源/目标细胞以及评估协议；4000 步不是收敛保证。

## 分阶段运行：先看轨迹，再查速度

```bash
# 1. 联合训练：只使用可见时间点；按可见验证分布 SWD 选检查点。
bash run.sh train --data "$OUTPUT_ROOT/data/demo" --output-root "$OUTPUT_ROOT" \
  --run experiment_a --arm baseline --steps 4000 --device mps

CHECKPOINT="$OUTPUT_ROOT/models/experiment_a/selected.pt"

# 2. 导出重构轨迹、质量、中心导数、诱导速度和模型速度。
bash run.sh export --data "$OUTPUT_ROOT/data/demo" --output-root "$OUTPUT_ROOT" \
  --run experiment_a --checkpoint "$CHECKPOINT" --device mps

# 3. 冻结模型，重新积分，检查轨迹、质量、异常路径和留出分布。
bash run.sh validate --data "$OUTPUT_ROOT/data/demo" --output-root "$OUTPUT_ROOT" \
  --run experiment_a --checkpoint "$CHECKPOINT" --dtmax 0.0125 --device mps

# 4. 画图，不训练。
bash run.sh plot --data "$OUTPUT_ROOT/data/demo" --output-root "$OUTPUT_ROOT" --run experiment_a
```

如先完成 `export`，可以直接读取 `reconstruction.npz` 看优化轨迹；标准对照图需要完成 `validate`。使用新 run 名重新评估，现有数字结果拒绝覆盖。`plot` 可从已验证缓存重新生成草图。

## 产物和判读

| 位置 | 内容 |
|---|---|
| `models/<run>/selected.pt`, `final.pt` | 选中/末步参数、初始化、配置；未保存优化器/RNG 精确续训状态 |
| `models/<run>/source/` | 新训练时的源代码快照 |
| `results/<run>/protocol.json`, `training.json` | 数据/源码哈希、参数量、计时、选择规则 |
| `results/<run>/reconstruction.npz` | 优化路径及 log 质量、分段中心导数、诱导场与学习场；含物理时间单位版本 |
| `results/<run>/export.json` | KDE 速度/生长残差；仅模型内一致性 |
| `results/<run>/rollouts.npz`, `validation.json` | 同起点独立积分、真实起始细胞积分、留时预测、步长对照、质量/ESS/异常路径 |
| `logs/<run>/training.jsonl` | 损失、可见验证 SWD、梯度范数 |
| `figures/_drafts/<run>/reconstruction_audit.*` | PNG/PDF/SVG 草图及来源信息；全部路径保留 |

速度单位是**输入特征单位/输入物理时间单位**。例如 GSE75748 的模型时间是小时/24，导出的每小时速度和生长率为模型值/24；不自动转换成原始基因表达速度。

`same_initial_endpoint_coordinate_rmse` 是同一优化起点的逐坐标误差；`weighted_vector_rms` 先对维度求平方和，再按质量加权，二者数值不能混用。留时 W1/W2 是归一化有限经验分布的精确 OT；MMD² 是三个训练带宽的 biased RBF V-statistic。它们不检验总质量，也不证明单细胞真实速度。

异常路径同时报告质量比例和粒子比例。训练支持距离只是最近邻描述指标：两次采样之间的合理过渡也可能远离已观测细胞，不能把中间时刻的高比例直接解释成生物学错误。ESS 高也不能保证轨迹正确。

**通过的层级：** 数学/实现测试 → 冻结模型内一致性 → 留出分布泛化 → 有真实向量、谱系或扰动数据时的外部速度验证。只有合成示例附带真实速度；真实单细胞快照本身不能唯一识别速度或局部生长。

## 导入现有项目检查点

```bash
bash run.sh import-axis --axis-dir /path/to/sealed_axis --output "$OUTPUT_ROOT/data/gse75748"
bash run.sh import-checkpoint --checkpoint /path/to/old/selected.pt \
  --protocol /path/to/old/protocol.json --data "$OUTPUT_ROOT/data/gse75748" \
  --output-root "$OUTPUT_ROOT" --run frozen_model
```

随后对导入的 `models/frozen_model/selected.pt` 运行 `export/validate/plot`。导入核对原始输入哈希、观测时间、质量目标和配置，并验证张量和 CPU 场输出完全一致，不更新模型权重。仅加载可信的项目检查点。

## 验证代码包

```bash
python -m unittest discover -s tests -v
```

测试涵盖解析势能梯度、带异质生长的连续性恒等式、留时隔离、分割冲突、无未来轨迹依赖、解析速度/生长积分、检查点往返、物理时间映射及两可见时刻边界情况。`scripts/verify_delivery.py` 是本地交付审计入口：合成短训练 + 冻结 GSE75748 重放，需要已有项目输入；不属于通用用户输入的依赖。

已完成的本机交付核对见 [VERIFIED_RUN.md](../VERIFIED_RUN.md)，当前真实数据结果见 [current_results.md](current_results.md)。未恢复此前暂停的真实数据训练队列，也没有声称现有 GSE75748 模型精度变好。
