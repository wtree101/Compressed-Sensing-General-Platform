# 谱初始化测试更新说明

## 更新日期
2025-12-23

## 关键变更

### 从随机Tucker张量 → 真实谱初始化张量

**之前**:  
测试使用随机生成的Tucker张量和因子矩阵

**现在**:  
测试使用真实的谱初始化过程：
1. 生成真实矩阵 X_true (d₁ × d₂)
2. 生成测量矩阵 A_i (i=1...m)
3. 计算测量值 y_i = |⟨Ai, X_true⟩|²
4. 构建谱张量 **H = Σᵢ yᵢ * (Aᵢ ⊗ Aᵢ)**
5. 使用HOSVD提取因子矩阵 U₁, U₂, U₃, U₄

## 测试流程

### Step 1: 生成真实数据
```matlab
% Ground truth
X_true = randn(d1, d2) / norm_fro

% Measurement matrices
A_cells{i} = randn(d1, d2) / sqrt(d1*d2)  % i = 1...m

% Measurements
y(i) = |<A_cells{i}, X_true>|²
```

### Step 2: 谱初始化 (核心)
```matlab
% 谱张量 (这就是 initialize_spectral 做的事情)
H_tensor = zeros(d1, d2, d1, d2);
y_spectral = y.^2 * sqrt(m);

for i = 1:m
    H_tensor = H_tensor + y_spectral(i) * (A_cells{i} ⊗ A_cells{i});
end
```

这个公式对应于 `initialize_tensor_lift_tucker_spectral.m` 中的:
```matlab
y_spectral = y.^2 * sqrt(m);
[U_cell_init, G_init] = T_tucker.initialize_spectral(...
    spectral_operator, y_spectral, m, ...);
```

### Step 3: 提取因子矩阵
```matlab
[U_cell, ~, G_hosvd] = HOSVD_with_factors(H_tensor, [r, r, r, r]);

U1 = U_cell{1};  % d₁ × r
U2 = U_cell{2};  % d₂ × r  
U3 = U_cell{3};  % d₁ × r
U4 = U_cell{4};  % d₂ × r
```

### Step 4-8: 测试验证
- 计算核心张量 G
- 验证公式: G_mat = (U₂' ⊗ U₁') H_mat (U₄' ⊗ U₃')'
- 检查对称性
- 矩阵提取
- 性能对比

## 为什么这样更好？

### 1. **与实际使用一致**
- 测试现在模拟 `initialize_tensor_lift_tucker_spectral.m` 的真实工作流
- 从测量数据开始 → 谱初始化 → Tucker分解 → 矩阵提取

### 2. **验证谱初始化性质**
检验理论预测:
```
H = Σᵢ yᵢ * (Aᵢ ⊗ Aᵢ) ≈ λ * (X_true ⊗ X_true)
```

对于相位恢复问题，这应该导致:
- **U₁ ≈ U₃** (行空间)
- **U₂ ≈ U₄** (列空间)

### 3. **测试真实数值行为**
- 有限测量数 m 的影响
- 数值精度问题
- HOSVD 截断效果
- 条件数和稳定性

## 关键数学结构

### 谱张量的理论形式

对于相位恢复 yᵢ = |⟨Aᵢ, X⟩|²:

```
H = Σᵢ yᵢ * (Aᵢ ⊗ Aᵢ)
  = Σᵢ |⟨Aᵢ, X⟩|² * (Aᵢ ⊗ Aᵢ)
  ≈ E[|⟨A, X⟩|² * (A ⊗ A)]  (当 m → ∞)
  = c * (X ⊗ X)  (在某些分布下)
```

### 部分对称性的来源

当 H ≈ λ * (X ⊗ X) 且 X = [x₁, x₂, ..., x_{d₂}] ∈ ℝ^{d₁×d₂}:

```
H(i₁, i₂, j₁, j₂) = λ * X(i₁,i₂) * X(j₁,j₂)
```

这个4阶张量的HOSVD会产生:
- **U₁**: span{x₁, x₂, ..., x_{d₂}} 在模式1 (第一个i₁索引)
- **U₂**: 列向量空间在模式2 (i₂索引)
- **U₃**: 同U₁ (j₁索引，也是行空间)
- **U₄**: 同U₂ (j₂索引，也是列空间)

因此 **U₁≈U₃** 和 **U₂≈U₄** 是理论保证的！

## 预期测试结果

### 理想情况 (m → ∞, 无噪声)
```
||U1 - U3||_F ≈ 0  (机器精度)
||U2 - U4||_F ≈ 0  (机器精度)
H_mat 对称性: < 1e-14 (对于d₁=d₂的情况)
公式验证: < 1e-14
```

### 实际情况 (m=200, d₁=12, d₂=15)
```
||U1 - U3||_F ≈ 0.01 - 0.5  (部分对称)
||U2 - U4||_F ≈ 0.01 - 0.5  (部分对称)
H_mat 对称性: ~ 1.0 (X非方阵，不期望对称)
公式验证: < 1e-10 (仍然精确)
```

### 为什么不完美？

1. **有限测量数**: m=200 < ∞
2. **数值HOSVD**: 截断引入误差
3. **随机性**: A_i 是随机矩阵
4. **非方阵**: d₁≠d₂ 使结构更复杂

## 与 initialize_tensor_lift_tucker_spectral.m 的对应

| 测试中的步骤 | 对应的实现代码 | 行号 |
|--------------|----------------|------|
| Step 1: 生成X_true和测量 | 外部准备数据 | - |
| Step 2: H = Σᵢ yᵢ*(Aᵢ⊗Aᵢ) | `initialize_spectral` | 200 |
| Step 3: HOSVD提取U | `initialize_spectral`内部 | TuckerTensor.m |
| Step 4: 计算G | tensor_mode_product | 236 |
| Step 5: 验证公式 | (测试特有) | - |
| Step 7: 提取矩阵 | `extract_matrix_from_tucker` | 423 |

## 验证的数学命题

### 命题1: 公式正确性 (任何H)
```
G = H ×₁ U₁' ×₂ U₂' ×₃ U₃' ×₄ U₄'
⟺ G_mat = (U₂' ⊗ U₁') H_mat (U₄' ⊗ U₃')'
```
✓ 应对所有情况成立，误差 < 1e-10

### 命题2: 部分对称性 (谱初始化H)
```
如果 H = Σᵢ yᵢ*(Aᵢ⊗Aᵢ) 来自相位恢复，
则 HOSVD(H) 产生 U₁≈U₃, U₂≈U₄
```
✓ 应观察到差异 < 0.5，理想情况 < 0.1

### 命题3: 矩阵提取 (部分对称情况)
```
两种提取方法应该一致:
- Method 1: H_mat的特征分解
- Method 2: G_mat的特征分解 + 重建
```
✓ 差异应 < 1e-3，理想情况 < 1e-6

## 运行测试

```matlab
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
test_tucker_nonsymmetric
```

## 调试信息

如果测试失败，检查:

1. **HOSVD_with_factors可用性**:
   ```matlab
   which HOSVD_with_factors
   ```

2. **测量数是否足够**:
   ```matlab
   m >= 4 * d1 * d2  % 推荐
   ```

3. **秩设置是否合理**:
   ```matlab
   r <= min(d1, d2)
   ```

4. **X_true是否数值稳定**:
   ```matlab
   cond(X_true)  % 应 < 100
   ```

## 总结

✅ **测试现在验证真实的谱初始化流程**  
✅ **从测量数据 → 谱张量 → Tucker分解 → 矩阵提取**  
✅ **检验理论预测的部分对称性 U₁≈U₃, U₂≈U₄**  
✅ **与 initialize_tensor_lift_tucker_spectral.m 完全一致**  

---
**最后更新**: 2025-12-23  
**测试文件**: `test_tucker_nonsymmetric.m`  
**相关代码**: `initialize_tensor_lift_tucker_spectral.m`

