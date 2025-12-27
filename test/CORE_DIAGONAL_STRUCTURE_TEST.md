# 核心张量对角结构测试

## 测试日期
2025-12-23

## 测试目标

验证从谱初始化得到的核心张量 G 是否具有特殊的对角结构：

\[
G_{ijkl} = \begin{cases}
\sigma_i \sigma_k, & \text{if } i = j \text{ and } k = l \\
0, & \text{otherwise}
\end{cases}
\]

## 理论背景

### 谱初始化的理想情况

对于相位恢复问题，当 H ≈ λ(X ⊗ X) 且 X 为秩-r 矩阵：

1. **Tucker 分解**: H = (U₂ ⊗ U₁) · G_mat · (U₄ ⊗ U₃)'
2. **部分对称性**: HOSVD 产生 U₁≈U₃, U₂≈U₄
3. **核心结构**: G 应该有低维对角结构

### 为什么是对角结构？

**命题**: 如果 H = λ·vec(X)·vec(X)' (秩-1 张量)，且 X 是秩-r 矩阵，则：
- Tucker 分解的核心张量 G 是对角的
- 只有 i=j 且 k=l 的位置非零（即 G(i,i,k,k) 形式）
- 这些非零值满足 G(i,i,k,k) = σᵢ·σₖ

**证明思路**:
```
X = U·Σ·V' (秩-r SVD)
vec(X) = (V ⊗ U)·vec(Σ)

H = λ·vec(X)·vec(X)'
  = λ·(V ⊗ U)·vec(Σ)·vec(Σ)'·(V ⊗ U)'

当 HOSVD 正确对齐因子:
U₁ ≈ U, U₂ ≈ V, U₃ ≈ U, U₄ ≈ V

则: G(i,j,k,l) = (U₁'·U)ᵢ·(U₂'·V)ⱼ·Σ·Σ'·(U₃'·U)ₖ·(U₄'·V)ₗ
```

由于 U₁'·U ≈ I, U₂'·V ≈ I (HOSVD 对齐)，这导致对角结构。

## 测试流程

### Step 1: 提取对角元素

```matlab
G_diag_values(k, l) = G_tensor(k, k, l, l)  % k,l = 1...r
```

这是一个 r×r 矩阵，包含所有"对角"位置的值。

### Step 2: 计算偏离对角的能量

```matlab
G_diag_tensor(i,j,k,l) = {  G(i,i,k,k), if i=j and k=l  0,          otherwise
}

off_diagonal_norm = ||G - G_diag_tensor||_F
relative_off_diag = off_diagonal_norm / ||G||_F
```

**判断标准**:
- < 1e-6: 完美对角结构 ✓
- < 0.1: 近似对角结构 ~
- ≥ 0.1: 非对角结构 ✗

### Step 3: 测试秩-1 分解

如果 G 是对角的，进一步测试 G_diag_values 是否为秩-1:

```matlab
G_diag_values ≈ σ ⊗ σ = σ·σ'
```

使用 SVD:
```matlab
[U, S, V] = svd(G_diag_values)
σ = sqrt(s₁) · U(:,1)  % 主奇异值和向量
```

**判断标准**:
```matlab
rank1_error = ||G_diag_values - σ·σ'||_F / ||G_diag_values||_F

< 1e-3: 秩-1 结构 ✓
< 0.1:  近似秩-1 ~
≥ 0.1:  非秩-1 ✗
```

### Step 4: 验证 σ 向量

重建并验证:
```matlab
G_reconstructed(k, l) = σₖ · σₗ
verification_error = ||G_diag_values - G_reconstructed||_F
```

## 预期结果

### 理想情况 (X 秩-1, m→∞)

```
=== Step 4b: Test Core Tensor Diagonal Structure ===

Diagonal structure G(k,k,l,l):
   5.4321   -2.3456
  -2.3456    1.0123

Diagonal structure analysis:
  ||G - G_diag||_F = 1.234567e-15
  ||G||_F = 6.789012e+00
  Relative off-diagonal energy: 1.820000e-16
  ✓ Core tensor is DIAGONAL (i=j, k=l structure: G(i,i,k,k))

Testing factorization: G(k,k,l,l) = σ_k * σ_l
  Singular values of G_diag_values:
    σ_1 = 6.789012e+00
    σ_2 = 1.234567e-15

  Rank-1 approximation error: 1.820000e-16
  ✓ G_diag_values ≈ σ ⊗ σ (rank-1 structure)
  
  Extracted σ vector:
    σ_1 = 2.345678
    σ_2 = -1.012345

  Verification: ||G_diag - σ*σ^T||_F / ||G_diag||_F = 1.234567e-16
  ✓✓ Core tensor has EXACT diagonal structure: G_{ikjl} = σ_k*σ_l
```

### 实际情况 (有限 m, 噪声)

```
=== Step 4b: Test Core Tensor Diagonal Structure ===

Diagonal structure G(k,k,l,l):
   5.4321   -2.3456
  -2.3456    1.0123

Diagonal structure analysis:
  ||G - G_diag||_F = 3.456789e-02
  ||G||_F = 6.789012e+00
  Relative off-diagonal energy: 5.092000e-03
  ~ Core tensor is APPROXIMATELY diagonal (0.51% off-diagonal)
    This is expected for finite measurements m=300

Testing factorization: G(k,k,l,l) = σ_k * σ_l
  Singular values of G_diag_values:
    σ_1 = 6.789012e+00
    σ_2 = 3.456789e-02

  Rank-1 approximation error: 5.092000e-03
  ✓ G_diag_values ≈ σ ⊗ σ (rank-1 structure)
  
  Extracted σ vector:
    σ_1 = 2.345678
    σ_2 = -1.012345

  Verification: ||G_diag - σ*σ^T||_F / ||G_diag||_F = 5.092000e-03
  ~ Core tensor has approximate diagonal structure
```

### 失败情况 (不满足假设)

```
=== Step 4b: Test Core Tensor Diagonal Structure ===

Diagonal structure analysis:
  ||G - G_diag||_F = 3.456789e+00
  ||G||_F = 6.789012e+00
  Relative off-diagonal energy: 5.092000e-01
  ✗ Core tensor is NOT diagonal (50.92% off-diagonal)
    The diagonal structure G_{ijkl} = σ_i*σ_k if i=j, k=l is NOT satisfied
```

## 理论意义

### 1. 验证谱初始化理论

对角结构是谱初始化正确工作的标志：
- ✓ 对角结构 → 谱初始化对齐了子空间
- ✗ 非对角 → 可能的问题：
  - m 太小
  - 测量矩阵不够随机
  - X 不是低秩的

### 2. 计算效率

如果 G 是对角的：
- 存储: O(r²) 而非 O(r⁴)
- 前向传播: 更快
- 梯度计算: 更简单

### 3. 数值稳定性

对角 G 意味着：
- 更好的条件数
- 更稳定的矩阵提取
- 更少的数值误差累积

## 数学定理

### 定理 1: 秩-1 张量的核心对角性

**假设**: 
- H = λ·v·v' (秩-1 矩阵化张量)
- v = (V ⊗ U)·w 对于某些 U, V, w
- HOSVD(H) 产生因子 {U₁,U₂,U₃,U₄} 和核心 G

**结论**: 
如果 U₁≈U₃≈U 且 U₂≈U₄≈V，则 G 满足对角结构。

### 定理 2: σ 向量的唯一性

对于对角核心 G_{ikjl} = σₖ·σₗ：
- σ 向量在全局符号和置换下是唯一的
- σ 的模长由 ||G||_F 确定
- σ 的方向由主特征向量确定

## 实现细节

### 关键代码片段

```matlab
% 检查对角性
G_diag_tensor = zeros(r, r, r, r);
for k = 1:r
    for l = 1:r
        G_diag_tensor(k, k, l, l) = G_tensor(k, k, l, l);
    end
end
relative_off_diag = norm(G_tensor(:) - G_diag_tensor(:)) / norm(G_tensor(:));

% 提取 σ
G_diag_values = zeros(r, r);
for k = 1:r
    for l = 1:r
        G_diag_values(k, l) = G_tensor(k, k, l, l);
    end
end
[U, S, V] = svd(G_diag_values);
sigma_vec = sqrt(S(1,1)) * U(:,1);

% 验证
G_reconstructed = sigma_vec * sigma_vec';
verification_error = norm(G_diag_values - G_reconstructed, 'fro');
```

### 数值注意事项

1. **浮点精度**: 判断对角性时使用相对误差
2. **符号歧义**: σ 可能有全局符号翻转
3. **小特征值**: 当 r 小时，SVD 更稳定

## 与其他测试的关系

### test_core_tensor_structure.m
- 测试一般的 Tucker 公式
- 不假设对角结构
- 更基础的验证

### test_tucker_nonsymmetric.m (本测试)
- 专门测试对角结构
- 针对谱初始化场景
- 更深入的结构分析

### test_tucker_spectral_symmetry.m
- 测试完全对称情况 (U₁=U₂=U₃=U₄)
- 对角结构在此也应成立

## 何时期望对角结构？

| 条件 | 对角结构 | 秩-1 G_diag |
|------|----------|-------------|
| X 秩-1, m→∞ | ✓✓ (精确) | ✓✓ (精确) |
| X 秩-1, m 有限 | ✓ (近似) | ✓ (近似) |
| X 秩-r, m→∞ | ✓ (近似) | ✗ (秩-r) |
| X 秩-r, m 有限 | ~ (部分) | ✗ (秩-r) |
| X 满秩 | ✗ | ✗ |

## 运行测试

```matlab
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
test_tucker_nonsymmetric
```

在输出中查找 "Step 4b: Test Core Tensor Diagonal Structure"。

## 调试失败情况

### 如果对角性检查失败

**可能原因**:
1. m 太小 (尝试增加到 m > 5·d₁·d₂)
2. X 不是低秩 (检查 rank(X))
3. 测量矩阵相关 (检查 A_cells)
4. Tucker 秩 r 太大 (应该 r ≤ rank(X))

**调试步骤**:
```matlab
% 检查 H 的秩-1 性质
[U_h, S_h, V_h] = svd(H_mat);
diag(S_h)  % 应该有一个主导奇异值

% 检查因子对齐
norm(U1 - U3, 'fro')  % 应该小
norm(U2 - U4, 'fro')  % 应该小

% 检查 X 的秩
rank(X_true, 1e-6)  % 应该小
```

### 如果秩-1 检查失败

这可能是**正常的**，如果：
- X 的秩 > 1
- m 有限导致噪声

在这种情况下，G_diag_values 应该有 rank(X) 个显著奇异值。

## 理论参考

1. **Tucker 分解**: Kolda & Bader (2009)
2. **谱方法**: Netrapalli et al. (2013)
3. **相位恢复**: Candès et al. (2015)
4. **张量对角化**: De Lathauwer et al. (2000)

## 未来工作

1. **理论证明**: 在噪声下对角性的定量界
2. **算法利用**: 利用对角结构加速计算
3. **秩估计**: 从 G_diag_values 的奇异值估计 rank(X)
4. **稀疏利用**: 存储和操作对角 G

---

**最后更新**: 2025-12-23  
**测试文件**: `test_tucker_nonsymmetric.m`  
**相关代码**: `initialize_tensor_lift_tucker_spectral.m`  
**状态**: ✅ 已实现并测试

