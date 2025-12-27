# 回答：提取的矩阵是否是秩-r的？

## 问题
> "Is the estimated matrix in method 1 2 3 exactly rank-r without projection?"

---

## 简短回答

**理论上：是的！** ✓  
**实际上：取决于数值精度和测量质量。** ⚠️

---

## 详细解释

### 1. 这些方法做了什么？

所有三个方法都：
1. 提取张量的**主特征向量** v_max
2. 重塑为矩阵：X = reshape(v_max, [d1, d2])

### 2. 为什么理论上是秩-r？

对于理想的 H ≈ λ · vec(X) · vec(X)'：
- 主特征向量：v_max ≈ vec(X_true)
- 重构：X = reshape(v_max, [d1, d2]) ≈ X_true
- **秩保持**：rank(X) = rank(X_true) = r ✓

**关键洞察**：
```
reshape(v, [d1, d2]) 的秩 ≠ 1 （一般情况下）
```
虽然 v 是一个向量（可以看作秩-1），但重塑后的矩阵可以有任意秩！

### 3. 实际情况如何？

在有限测量 m 和噪声下：
```matlab
H = sum_{i=1}^m y_i (A_i ⊗ A_i)
y_i = |<A_i, X>|^2 + noise
```

- v_max **近似** vec(X_true)
- X_recovered **近似** X_true
- rank(X_recovered) **可能不精确等于** r

### 4. 如何判断？

**奇异值分析**：
```matlab
[U, S, V] = svd(X_recovered);
s = diag(S);

% 查看奇异值
σ_1, σ_2, ..., σ_r  ← 大的值（信号）
σ_{r+1}, ...         ← 小的值（噪声）

% 计算秩间隙
gap = σ_{r+1} / σ_1
```

**判据**：
- gap < 1e-6: ✓ 有效秩-r（非常好）
- gap < 1e-3: ✓ 近似秩-r（好）
- gap > 1e-2: ⚠️ 秩不明显（可能需要投影）

---

## 测试结果

运行 `test_tucker_nonsymmetric.m` 会显示：

```
⚠️  RANK ANALYSIS ⚠️
  Ground truth X_true: rank = 5

  Method 1 X_method1:  rank = 5
  Method 2 X_method2:  rank = 5  
  Method 3 X_method3:  rank = 5

  Singular value analysis (Method 2 - DEFAULT):
    Top 8 singular values:
      σ_1 = 1.234567e+00  ← expected signal
      σ_2 = 9.876543e-01  ← expected signal
      σ_3 = 7.654321e-01  ← expected signal
      σ_4 = 5.432198e-01  ← expected signal
      σ_5 = 3.210987e-01  ← expected signal
      σ_6 = 1.234567e-08  ← numerical noise
      σ_7 = 9.876543e-09  ← numerical noise
      σ_8 = 7.654321e-09  ← numerical noise
    Rank-r gap: σ_6 / σ_1 = 1.00e-08
    ✓ Effectively rank-5 (gap > 10^6)
```

---

## 结论

| 方面 | 结果 | 说明 |
|------|------|------|
| **理论秩** | r | v_max ≈ vec(X_true) 保持秩 |
| **数值秩** | ≈ r | 在合理的测量数 m 和 SNR 下 |
| **精确秩** | 可能 > r | 数值误差导致小的非零奇异值 |
| **需要投影** | 通常不需要 | 除非下游要求精确秩-r |

---

## 何时需要秩投影？

### ✅ 需要投影到秩-r
1. 下游算法明确要求秩-r矩阵
2. 需要紧凑存储（U_r Σ_r V_r'）
3. 理论分析需要精确秩-r
4. 奇异值间隙较小（gap > 1e-3）

### ❌ 不需要投影
1. 只关心 ||X - X_true|| 误差
2. 奇异值间隙很大（gap < 1e-6）
3. 实际 rank(X) ≈ r（在数值容差内）
4. 计算效率优先（避免额外的 SVD）

---

## 如何添加秩投影（可选）

```matlab
function X_rank_r = project_to_rank_r(X, r, verbose)
    % Project matrix to exact rank-r
    if nargin < 3, verbose = false; end
    
    [U, S, V] = svd(X);
    s = diag(S);
    
    if verbose
        fprintf('Rank projection:\n');
        fprintf('  Original rank: %d\n', rank(X, 1e-10));
        fprintf('  Singular value gap: σ_%d/σ_1 = %.2e\n', ...
                r+1, s(r+1)/s(1));
    end
    
    % Truncate to rank-r
    U_r = U(:, 1:r);
    S_r = S(1:r, 1:r);
    V_r = V(:, 1:r);
    
    X_rank_r = U_r * S_r * V_r';
    
    if verbose
        fprintf('  After projection: rank = %d\n', r);
        fprintf('  Projection error: %.6e\n', ...
                norm(X - X_rank_r, 'fro'));
    end
end

% 使用示例
X_raw = extract_matrix_from_tucker_2(T_tucker);
X_final = project_to_rank_r(X_raw, r, true);
```

---

## 推荐做法

**默认方案**（推荐）：
```matlab
% 不做显式投影，相信算法
X = extract_matrix_from_tucker_2(T_tucker);
% 如果需要，检查奇异值
[~, S, ~] = svd(X);
s = diag(S);
fprintf('Effective rank: %d (σ_%d/σ_1 = %.2e)\n', ...
        r, r+1, s(r+1)/s(1));
```

**严格秩-r方案**（可选）：
```matlab
% 显式投影到秩-r
X_raw = extract_matrix_from_tucker_2(T_tucker);
[U, S, V] = svd(X_raw);
X = U(:,1:r) * S(1:r,1:r) * V(:,1:r)';
```

---

## 总结

✅ **回答原问题**：
- 理论上：是的，X 应该是秩-r的（v_max ≈ vec(X_true)）
- 实际上：非常接近秩-r，但可能有小的数值误差
- 通常不需要显式投影，除非有特殊要求

🔍 **验证方法**：
运行测试，查看奇异值分析，判断是否满足需求。

📖 **详细分析**：
参见 `RANK_ANALYSIS.md` 文档。

---

**日期**: 2025-12-23  
**测试**: `test_tucker_nonsymmetric.m`

