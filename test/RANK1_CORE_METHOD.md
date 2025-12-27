# Tucker 张量秩-1 核心约束方法

## 日期
2025-12-24

## 概述

实现了**参数化交替优化**方法来从 A^*(y) 提取 Tucker 张量，同时强制核心张量 G 在矩阵化后为**秩-1**。

---

## 理论基础

### 问题设定

给定测量向量 y 和算子 A，我们希望找到 Tucker 分解：

```
T = G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
```

其中：
- **Tucker 秩**: (r, r, r, r)
- **额外约束**: G 矩阵化为 G_mat (r²×r²) 后是**秩-1**的

### 参数化

秩-1 矩阵可以参数化为：

```
G_mat = q · q'    (对称情况)
```

其中 q ∈ ℝ^(r²) 是向量。

这样，G_mat 自动满足：
- rank(G_mat) = 1 ✓
- G_mat 是对称的 ✓
- 只需优化 r² 个参数（而不是 r⁴ 个）

### 优化目标

```
minimize  ||A(T(U, q)) - y||²
subject to: U_i 是正交矩阵
           G_mat = q · q'
```

---

## 交替优化算法

### 初始化

**方法 1: HOSVD 初始化（推荐）**
```matlab
% 1. 计算 H = A^*(y)
H_tensor = operator.kronecker_adjoint(y_spectral / sqrt(m));

% 2. HOSVD 分解
[G_init, U_cells_init] = HOSVD_with_factors(H_tensor, [r,r,r,r]);

% 3. 提取主特征向量作为 q
G_mat_init = reshape(G_init, [r*r, r*r]);
[V, D] = eig((G_mat_init + G_mat_init')/2);
[~, idx] = max(abs(diag(D)));
q_init = V(:, idx) * sqrt(abs(D(idx, idx)));
```

**方法 2: 随机初始化**
```matlab
U_cells = {orth(randn(d1,r)), orth(randn(d2,r)), ...};
q_init = randn(r*r, 1);
q_init = q_init / norm(q_init);
```

### 交替步骤

每次迭代包含两个子步骤：

#### 步骤 A: 固定 q，优化 U

```matlab
for u_iter = 1:n_u_steps
    % 1. 构造当前 Tucker 张量
    G_current = reshape(q * q', [r,r,r,r]);
    T_current = TuckerTensor(..., 'G', G_current, 'U', U_cells);
    
    % 2. 计算 Riemannian 梯度
    y_pred = operator.forward(T_current);
    Grad_F = operator.get_proj_grad_kronecker(T_current, y_pred, y);
    
    % 3. 更新 U（梯度下降 + 重正交化）
    for k = 1:4
        U_new = U_cells{k} - mu_u * Grad_F.Up{k};
        U_cells{k} = orth(U_new);
    end
end
```

**关键点**:
- 只更新 U，G 由 q 固定
- 使用 Riemannian 梯度保持 U 在 Stiefel 流形上
- `orth()` 重正交化确保 U'U = I

#### 步骤 B: 固定 U，优化 q

```matlab
for q_iter = 1:n_q_steps
    % 1. 构造当前张量
    G_current = reshape(q * q', [r,r,r,r]);
    T_current = TuckerTensor(..., 'G', G_current, 'U', U_cells);
    
    % 2. 计算关于 G 的梯度
    y_pred = operator.forward(T_current);
    Grad_F = operator.get_proj_grad_kronecker(T_current, y_pred, y);
    dG = Grad_F.G;  % r×r×r×r
    
    % 3. 链式法则: ∂L/∂q = ∂L/∂G : ∂G/∂q
    %    其中 G = reshape(q*q', [r,r,r,r])
    %    因此 ∂G/∂q 对应 2*G_mat*q
    dG_mat = reshape(dG, [r*r, r*r]);
    grad_q = 2 * dG_mat * q;
    
    % 4. 梯度下降更新
    q = q - mu_q * grad_q;
    
    % 5. 归一化（保持尺度）
    q = q / norm(q);
end
```

**关键点**:
- G_mat = q·q' 的导数是 2·G_mat·q
- 归一化 q 防止数值不稳定
- 步长 mu_q 通常 > mu_u（q 空间更简单）

### 收敛判断

```matlab
% 计算变化量
q_change = min(norm(q - q_old), norm(q + q_old));  % 考虑符号模糊性
u_change = norm(U_new - U_old, 'fro');

% 收敛条件
if q_change < tol && u_change < tol
    converged = true;
end
```

---

## 使用方法

### 基本调用

```matlab
% 准备数据
operator = struct('A_cells', A_cells);
y = measurements;  % 相位恢复测量

% 调用函数
[U_cells, q_opt, G_rank1, history] = tucker_rank1_core_alternating(...
    operator, y, d1, d2, r, ...
    'max_iter', 30, ...
    'verbose', true);

% 提取矩阵
T_final = TuckerTensor([d1,d2,d1,d2], r, 'G', G_rank1, 'U', U_cells);
X_recovered = extract_matrix_from_tucker_2(T_final);
```

### 参数调优

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `max_iter` | 20 | 最大外层迭代 | 通常 20-50 足够 |
| `u_iter` | 5 | U 更新内迭代 | 3-10，越大越慢但可能更准确 |
| `q_iter` | 10 | q 更新内迭代 | 5-20，q 优化通常更快 |
| `mu_u` | 0.01 | U 步长 | 0.001-0.05，太大会发散 |
| `mu_q` | 0.1 | q 步长 | 0.01-0.5，可以较大 |
| `tol` | 1e-6 | 收敛容差 | 1e-4 到 1e-8 |

**调优策略**:
1. 先用默认参数运行
2. 观察收敛曲线：
   - 如果震荡：减小步长
   - 如果收敛太慢：增大步长或迭代次数
3. 检查最终 rank(G_mat) 是否 = 1

---

## 测试示例

### 运行测试

```bash
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
matlab -batch "test_tucker_rank1_core"
```

### 预期输出

```
=== Tucker Tensor with Rank-1 Core (Alternating Optimization) ===
Dimensions: d1=20, d2=30, Tucker rank=3, Measurements: 800
Max iterations: 30, Tolerance: 1.00e-06
...

[Step 2] Alternating Optimization...
Iter | Loss      | ||ΔU||   | ||Δq||   | rank(G)
-----|-----------|----------|----------|--------
   1 | 2.3456e+00 | 1.23e-01 | 3.45e-01 | 1
   2 | 1.8765e+00 | 8.76e-02 | 2.34e-01 | 1
   ...
  15 | 5.6789e-02 | 1.23e-04 | 4.56e-05 | 1

✓ Converged at iteration 15

[Step 3] Final Results:
  Final loss: 5.6789e-02
  rank(G_mat): 1 (exact rank-1: true) ✓
  σ_2/σ_1: 1.23e-12 (should be ~0 for rank-1)
```

### 结果分析

测试比较两种方法：
1. **标准 HOSVD**: 无约束，rank(G) 通常 > 1
2. **秩-1 核心**: 强制 rank(G) = 1

**成功指标**:
- ✓ rank(G_mat) = 1（精确秩-1）
- ✓ 矩阵恢复误差与标准方法相近或更好
- ✓ 收敛平滑（无震荡）

**失败情况**:
- ✗ rank(G_mat) > 1（约束未满足）
- ✗ 矩阵恢复误差显著变差
- ✗ 不收敛或震荡

---

## 数学细节

### 为什么参数化为 q·q'？

对于 4 阶张量 T = X ⊗ X，理想情况下：

```
T = vec(X) · vec(X)'    (秩-1)
```

Tucker 分解后：
```
T ≈ (U₂⊗U₁) · G_mat · (U₄⊗U₃)'
```

如果 U₁≈U₃, U₂≈U₄（谱初始化的结果），则：
```
G_mat ≈ (U'⊗U')⁻¹ · [vec(X)·vec(X)'] · (U'⊗U')⁻¹
```

这个矩阵应该接近秩-1！

### 梯度计算

对于 G_mat = q·q'，我们有：

```
∂G_mat/∂q = q·I + I·q = 2·q·I
```

因此：
```
∂L/∂q = ∂L/∂G_mat : ∂G_mat/∂q = 2·(dG_mat)·q
```

其中 `dG_mat` 是损失关于 G_mat 的梯度。

### 收敛性

**理论保证**:
- 交替优化单调降低损失（在适当步长下）
- 每个子问题是凸的（固定其他变量）
- 收敛到局部最优（可能不是全局）

**实际性能**:
- 通常 10-30 次迭代收敛
- 对初始化敏感（HOSVD 初始化效果最好）
- 步长选择关键（太大发散，太小慢）

---

## 与其他方法比较

| 方法 | rank(G) | 计算量 | 精度 | 推荐场景 |
|------|---------|--------|------|----------|
| **标准 HOSVD** | 自由 | O(d⁴) | 基线 | 快速原型 |
| **Method 1: 投影** | 后处理=1 | O(d⁴+r⁴) | 中等 | 简单应用 |
| **Method 2: 交替** | 始终=1 | O(iter·d²r²) | 最好 | 高精度需求 |
| **Method 3: 对角投影** | 1 | O(iter·r⁴) | 好 | 相信对角结构 |

**Method 2（本方法）的优势**:
- ✓ 秩-1 约束在优化中始终满足
- ✓ 不需要后处理投影
- ✓ 可以获得最优的秩-1 近似
- ✓ 灵活调节 U 和 q 的更新策略

**劣势**:
- ⚠️ 需要多次迭代（比单次 HOSVD 慢）
- ⚠️ 需要调节步长参数
- ⚠️ 可能收敛到局部最优

---

## 实验建议

### 验证秩-1 假设

在使用此方法前，先检查标准 HOSVD 的 G：

```matlab
% 标准 HOSVD
[G_hosvd, U_hosvd] = HOSVD_with_factors(H_tensor, [r,r,r,r]);
G_mat = reshape(G_hosvd, [r*r, r*r]);

% 秩-1 近似误差
[U, S, V] = svd(G_mat);
rank1_approx = S(1,1) * (U(:,1) * V(:,1)');
error = norm(G_mat - rank1_approx, 'fro') / norm(G_mat, 'fro');

fprintf('Rank-1 approximation error: %.2f%%\n', error * 100);
```

**判断标准**:
- error < 10%: ✓ 秩-1 假设合理，使用 Method 2
- 10% < error < 30%: ~ 可尝试，可能有改进
- error > 30%: ✗ 秩-1 假设不成立，使用标准方法

### 参数搜索

如果默认参数不收敛，尝试网格搜索：

```matlab
mu_u_list = [0.001, 0.005, 0.01, 0.05];
mu_q_list = [0.01, 0.05, 0.1, 0.5];

best_error = inf;
for mu_u = mu_u_list
    for mu_q = mu_q_list
        [U, q, G, hist] = tucker_rank1_core_alternating(...
            operator, y, d1, d2, r, ...
            'mu_u', mu_u, 'mu_q', mu_q, 'verbose', false);
        
        if hist.loss(end) < best_error
            best_error = hist.loss(end);
            best_mu_u = mu_u;
            best_mu_q = mu_q;
        end
    end
end

fprintf('Best parameters: mu_u=%.3f, mu_q=%.3f\n', best_mu_u, best_mu_q);
```

---

## 扩展方向

### 1. 非对称核心

当前实现：G_mat = q·q'（对称）

可扩展为：G_mat = p·q'（非对称，两个向量）

```matlab
% 优化 p 和 q 分别
grad_p = dG_mat * q;
grad_q = dG_mat' * p;

p = p - mu_p * grad_p;
q = q - mu_q * grad_q;
```

### 2. 秩-k 核心

推广到 rank(G_mat) = k：

```matlab
% 参数化为 k 个秩-1 项的和
G_mat = sum(lambda_i * q_i * q_i')

% 交替优化每个 q_i
```

### 3. 正则化

添加正则项鼓励特定结构：

```matlab
% 目标函数
L = ||A(T) - y||² + α·R(q)

% 例如：对角正则化
R(q) = ||q - diag_project(q)||²
```

---

## 故障排查

### 问题 1: 不收敛（震荡）

**症状**: loss 上下波动，不单调下降

**解决**:
```matlab
% 减小步长
'mu_u', 0.001,  % 从 0.01 减小
'mu_q', 0.01    % 从 0.1 减小

% 增加内迭代（更保守的更新）
'u_iter', 10,   % 从 5 增加
'q_iter', 20    % 从 10 增加
```

### 问题 2: rank(G_mat) > 1

**症状**: 最终 rank 不等于 1

**可能原因**:
1. 数值误差（tolerance 设置）
2. 优化未收敛
3. 秩-1 假设不成立

**解决**:
```matlab
% 检查奇异值
[~, S, ~] = svd(G_mat);
fprintf('σ values: '); disp(diag(S)');

% 如果 σ₂/σ₁ > 1e-6，可能是数值问题
% 手动投影到秩-1
[U, S, V] = svd(G_mat);
G_mat = S(1,1) * (U(:,1) * V(:,1)');
```

### 问题 3: 矩阵恢复误差大

**症状**: X_recovered 与 X_true 相差很大

**检查清单**:
1. ✓ 测量数 m 是否足够？（通常需要 m > 10r）
2. ✓ 初始化是否合理？（尝试 HOSVD 初始化）
3. ✓ 是否收敛？（检查 loss 曲线）
4. ✓ 秩-1 假设是否合理？（做前述验证）

---

## 总结

**何时使用此方法**:
- ✓ 理论上 G 应该是秩-1（如相位恢复）
- ✓ 需要高精度恢复
- ✓ 可以接受较长计算时间
- ✓ 有充足的测量（m >> r²）

**何时不使用**:
- ✗ G 显然不是秩-1
- ✗ 需要极快速度（用标准 HOSVD）
- ✗ 测量不足（m < 5r²）
- ✗ 噪声极大

**最佳实践**:
1. 先用标准 HOSVD 验证秩-1 假设
2. 从 HOSVD 初始化开始
3. 监控收敛曲线调节参数
4. 与标准方法比较验证改进

---

**文档版本**: 1.0  
**最后更新**: 2025-12-24  
**相关文件**:
- `tucker_rank1_core_alternating.m` - 核心实现
- `test_tucker_rank1_core.m` - 测试脚本

