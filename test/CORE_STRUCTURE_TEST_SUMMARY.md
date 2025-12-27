# 核心张量对角结构测试 - 快速指南

## 新增测试 (2025-12-23)

在 `test_tucker_nonsymmetric.m` 的 **Step 4b** 中添加了核心张量对角结构测试。

## 测试什么？

验证核心张量 G 是否具有对角结构：

```
G(i,j,k,l) = {  σᵢ · σₖ,  if i = j and k = l  0,        otherwise
}
```

## 为什么重要？

这是谱初始化正确工作的理论标志：
- ✓ 对角结构 → HOSVD 正确对齐了子空间
- ✗ 非对角 → 可能的问题（m 太小，X 不是低秩等）

## 如何运行？

```matlab
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
test_tucker_nonsymmetric
```

查找输出中的：
```
=== Step 4b: Test Core Tensor Diagonal Structure ===
```

## 预期输出

### 成功案例
```
Diagonal structure analysis:
  ||G - G_diag||_F = 1.234567e-15
  Relative off-diagonal energy: 1.820000e-16
  ✓ Core tensor is DIAGONAL (i=j, k=l structure: G(i,i,k,k))

Testing factorization: G(k,k,l,l) = σ_k * σ_l
  Rank-1 approximation error: 1.820000e-16
  ✓ G_diag_values ≈ σ ⊗ σ (rank-1 structure)
  
  ✓✓ Core tensor has EXACT diagonal structure: G_{ikjl} = σ_k*σ_l
```

### 实际案例 (有限 m)
```
Diagonal structure analysis:
  Relative off-diagonal energy: 5.092000e-03
  ~ Core tensor is APPROXIMATELY diagonal (0.51% off-diagonal)
    This is expected for finite measurements m=300

  Rank-1 approximation error: 5.092000e-03
  ✓ G_diag_values ≈ σ ⊗ σ (rank-1 structure)
```

## 判断标准

| 相对偏离 | 判断 | 说明 |
|----------|------|------|
| < 1e-6 | ✓✓ 完美对角 | 理想情况 |
| 1e-6 ~ 0.1 | ✓ 近似对角 | 实际情况（m 有限） |
| ≥ 0.1 | ✗ 非对角 | 假设不满足 |

## 何时期望对角结构？

| X 的秩 | 测量数 m | 对角结构？ | 秩-1 G_diag？ |
|--------|----------|------------|---------------|
| 1 | →∞ | ✓✓ (精确) | ✓✓ (精确) |
| 1 | 有限 | ✓ (近似) | ✓ (近似) |
| r > 1 | →∞ | ✓ (近似) | ✗ (秩-r) |
| r > 1 | 有限 | ~ (部分) | ✗ (秩-r) |

## 数学背景

对于秩-1 张量 H ≈ λ(X ⊗ X)：
```
X = U·Σ·V' (秩-r SVD)
vec(X) = (V ⊗ U)·vec(Σ)

当 HOSVD 正确对齐：
  U₁ ≈ U, U₂ ≈ V, U₃ ≈ U, U₄ ≈ V

则核心张量 G 只在对角位置 (k,k,l,l) 非零
且 G(k,k,l,l) = σₖ · σₗ
```

## 提取 σ 向量

如果 G 是对角的：

```matlab
% 1. 提取对角值
G_diag(k,l) = G(k,k,l,l)

% 2. SVD 分解
[U, S, V] = svd(G_diag)
σ = sqrt(S(1,1)) · U(:,1)

% 3. 验证
G_reconstructed = σ · σ'
error = ||G_diag - G_reconstructed||_F
```

## 调试失败

如果测试失败（相对偏离 > 0.1）：

**检查项**：
1. ✓ m 是否足够大？ (建议 m > 5·d₁·d₂)
2. ✓ X_true 是否低秩？ (rank(X_true) ≤ r)
3. ✓ Tucker 秩 r 是否合适？ (r ≤ rank(X_true))
4. ✓ 因子对齐是否正确？ (||U₁-U₃||, ||U₂-U₄|| 应该小)

**调试命令**：
```matlab
% 检查 X 的秩
rank(X_true, 1e-6)

% 检查因子对齐
norm(U1 - U3, 'fro')
norm(U2 - U4, 'fro')

% 检查 H 的秩-1 性质
[~, S, ~] = svd(H_mat);
diag(S)  % 应该有一个主导奇异值
```

## 相关测试

| 测试 | 内容 | 关系 |
|------|------|------|
| Step 3 | 因子对齐 U₁≈U₃, U₂≈U₄ | 前提条件 |
| Step 4b | **核心对角结构** | 本测试 |
| Step 5 | 公式 G_mat = ... | 通用验证 |
| Step 7 | 矩阵提取 | 应用 |

## 与理论的联系

**定理** (非正式): 
如果谱初始化 H = Σᵢ yᵢ(Aᵢ⊗Aᵢ) 收敛到 λ(X⊗X)，
且 HOSVD 正确对齐因子，
则核心张量 G 满足对角结构。

**意义**:
- 对角结构是谱初始化质量的指标
- 可用于算法加速（稀疏存储和计算）
- 提供 rank(X) 的估计（G_diag 的秩）

## 详细文档

完整理论和实现细节见：
📄 `CORE_DIAGONAL_STRUCTURE_TEST.md`

## 快速检查清单

运行测试后，检查：

- [ ] 相对偏离 < 0.1 （对角性）
- [ ] 秩-1 误差 < 0.1 （如果 X 是秩-1）
- [ ] σ 向量提取成功
- [ ] 验证误差 < 0.01

如果全部 ✓，则核心结构测试通过！

---

**添加日期**: 2025-12-23  
**测试文件**: `test_tucker_nonsymmetric.m`  
**位置**: Step 4b  
**文档**: `CORE_DIAGONAL_STRUCTURE_TEST.md`

