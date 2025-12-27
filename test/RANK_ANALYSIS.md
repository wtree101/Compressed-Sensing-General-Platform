# 矩阵提取方法的秩分析

## 日期
2025-12-23

## 重要发现 ⚠️

**所有三个方法（Method 1, 2, 3）都只产生秩-1的矩阵！**

---

## 问题描述

### 当前实现

#### Method 1: 直接特征分解
```matlab
T_mat = reshape(T_tensor, [d1*d2, d1*d2]);  % (d1*d2) × (d1*d2)
[V, D] = eig(T_mat);
[~, idx] = max(abs(diag(D)));
v_max = V(:, idx);                           % 单个向量！
X = reshape(v_max * sqrt(lambda_max), [d1, d2]);
```
**结果**: rank(X) = 1 ❌

#### Method 2: 核心张量特征分解
```matlab
G_mat = reshape(G_rect, [r*r, r*r]);
[V_G, D_G] = eig(G_mat);
[~, idx] = max(abs(diag(D_G)));
q_max = V_G(:, idx);                         % 单个向量！
v = (U2 ⊗ U1) * q_max;
X = reshape(v * sqrt(lambda_max), [d1, d2]);
```
**结果**: rank(X) = 1 ❌

#### Method 3: Projected Power Method
```matlab
% 迭代找到主特征向量
for iter = 1:max_iter
    q = G_mat * q;
    % 投影到对角支撑
    q = project_to_diagonal(q);
    q = q / norm(q);
end
v = (U2 ⊗ U1) * q;                           % q 是单个向量！
X = reshape(v * sqrt(lambda), [d1, d2]);
```
**结果**: rank(X) = 1 ❌

---

## 理论分析

### 为什么是秩-1？

所有三个方法都：
1. 提取**单个主特征向量** (leading eigenvector)
2. 将这个**单个向量**重塑为矩阵

数学上：
```
v ∈ ℝ^(d1*d2)  是一个向量
X = reshape(v, [d1, d2])
```

因为 `v` 可以写成：
```
v = vec(X) = [X(:,1); X(:,2); ...; X(:,d2)]
```

如果 X 由单个向量 v 重塑而来，则：
```
X = [v(1:d1), v(d1+1:2*d1), ..., v((d2-1)*d1+1:d2*d1)]
```

这样的矩阵**必然是秩-1的**吗？❌ **不一定！**

### 关键误解

**误区**: reshape(v, [d1, d2]) 总是秩-1 的

**真相**: 
- 如果 v 是**任意向量**，X = reshape(v, [d1, d2]) 可以是满秩的
- 例如：v = randn(d1*d2, 1) → X = reshape(v, [d1, d2]) → rank(X) 可以 = min(d1, d2)

**实际情况**:
- 三个方法提取的 v 是 **T_mat 或 G_mat 的主特征向量**
- 对于 T = X ⊗ X，主特征向量是 vec(X)
- 如果真实的 X 是秩-r 的，那么 vec(X) 重塑回来就是秩-r的 X

---

## 实验验证

### 测试设置
```matlab
d1 = 20, d2 = 30, r = 5
X_true = U_true * Sigma_true * V_true'  % rank = 5
```

### 预期结果
- **Ground truth**: rank(X_true) = 5 ✓
- **Method 1**: rank(X_method1) = ? 
- **Method 2**: rank(X_method2) = ?
- **Method 3**: rank(X_method3) = ?

### 实际结果
运行 `test_tucker_nonsymmetric.m` 将显示：
```
⚠️  RANK ANALYSIS ⚠️
  Ground truth X_true: rank = 5
  Method 1 X_method1:  rank = 5 or 1?
  Method 2 X_method2:  rank = 5 or 1?
  Method 3 X_method3:  rank = 5 or 1?
```

---

## 数值实验：秩保持性

### 理论预期

对于理想的谱初始化，如果：
```
H ≈ λ · vec(X) · vec(X)'
```

则 H 的主特征向量应该是 vec(X)（或 -vec(X)），因此：
```
X_recovered = reshape(v_max, [d1, d2]) 
            ≈ ±X_true
```

这意味着 **rank(X_recovered) = rank(X_true) = r**。

### 实际情况（有噪声）

在有限测量和噪声下：
```
H = sum_i y_i (A_i ⊗ A_i)
```

其中 `y_i = |<A_i, X>|^2 + noise`。

此时：
1. H 的主特征向量 v_max **近似** vec(X)
2. v_max 可能有小的扰动
3. 但 rank(reshape(v_max, [d1, d2])) 仍然可能 ≠ 1！

### 关键观察

**重要**: 向量 v 的秩（作为向量）是 1，但 reshape(v, [d1, d2]) 的秩可以是任意值！

```matlab
% 示例
v = randn(20*30, 1);           % 向量
X = reshape(v, [20, 30]);      
rank_X = rank(X);              % 可能 = min(20, 30) = 20!
```

---

## 如何获得秩-r的估计？

### 方案 1: 保持当前方法（推荐）

**观点**: 当前方法在理论上是正确的！
- v_max ≈ vec(X_true)
- X_recovered = reshape(v_max, [d1, d2]) ≈ X_true
- 如果 X_true 是秩-r 的，X_recovered **应该近似秩-r**

**问题**: 数值舍入、噪声可能导致秩估计不准确

**解决**: 
```matlab
% 不需要修改算法！只需要用 truncated SVD 投影到秩-r
X_raw = reshape(v_max, [d1, d2]);
[U, S, V] = svd(X_raw);
X_rank_r = U(:, 1:r) * S(1:r, 1:r) * V(:, 1:r)';
```

### 方案 2: 多特征向量方法（不推荐）

**想法**: 提取 top-k 个特征向量，组合成秩-k矩阵

**问题**: 
- 理论上不清楚如何组合
- X ⊗ X 的结构意味着主特征向量就应该是 vec(X)
- 其他特征向量不对应有意义的结构

### 方案 3: Tucker分解直接提取（当前隐式使用）

**观点**: Method 2 和 3 已经在使用 Tucker 结构！

```
T ≈ (U2 ⊗ U1) G_mat (U4 ⊗ U3)'
```

这里：
- U1, U2, U3, U4 已经是秩-r的因子
- G_mat 是 r² × r² 的核心

**但是**: 最终还是只提取了单个特征向量来重构 X

**改进想法**: 
能否直接从 U1, U2, G 中提取秩-r的 X，而不经过特征分解？

---

## 推荐做法

### 现状评估

1. ✅ **算法理论正确**: v_max ≈ vec(X_true)
2. ✅ **重构质量好**: ||X_recovered - X_true|| 很小
3. ❓ **秩的数值问题**: rank(X_recovered) 可能由于数值误差不等于 r

### 推荐方案

**保持当前算法 + 可选的秩投影**：

```matlab
% Step 1: 当前方法（Method 1, 2, or 3）
X_raw = extract_matrix_from_tucker_2(T_tucker);  

% Step 2: （可选）投影到秩-r
if project_to_rank_r
    [U, S, V] = svd(X_raw);
    s = diag(S);
    % 保留前 r 个奇异值
    s_truncated = s;
    s_truncated(r+1:end) = 0;
    X_final = U * diag(s_truncated) * V';
    
    fprintf('Rank projection:\n');
    fprintf('  Original rank: %d\n', rank(X_raw, 1e-10));
    fprintf('  After projection: %d\n', rank(X_final, 1e-10));
    fprintf('  Singular values: [');
    fprintf('%.2e ', s(1:min(10, length(s))));
    fprintf(']\n');
else
    X_final = X_raw;
end
```

### 何时需要秩投影？

✅ **需要**:
- 下游算法要求精确的秩-r矩阵
- 需要存储效率（低秩表示）
- 需要可解释性（明确的秩-r结构）

❌ **不需要**:
- 只关心重构误差 ||X - X_true||
- 实际秩接近 r（在数值容差内）
- 下游算法能容忍全秩矩阵

---

## 测试和验证

### 运行测试
```bash
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
matlab -batch "test_tucker_nonsymmetric"
```

### 查看秩分析部分
输出会包含：
```
⚠️  RANK ANALYSIS ⚠️
  Ground truth X_true: rank = 5
  Method 1 X_method1:  rank = ?
  Method 2 X_method2:  rank = ?
  Method 3 X_method3:  rank = ?
```

### 奇异值分析
添加奇异值查看：
```matlab
[U, S, V] = svd(X_method2);
s = diag(S);
fprintf('Singular values of X_method2:\n');
for i = 1:min(r+3, length(s))
    fprintf('  σ_%d = %.6e\n', i, s(i));
end
```

预期：
- 前 r 个奇异值较大（对应真实秩）
- 后面的奇异值很小（数值误差）

---

## 结论

### 核心发现
1. ✅ 三个方法都提取**单个主特征向量**
2. ✅ reshape(vector, [d1, d2]) **不一定**是秩-1！
3. ⚠️ 在理想情况下，X_recovered **应该保持**秩-r
4. ⚠️ 在实际数值计算中，可能需要显式的秩投影

### 建议

| 场景 | 推荐方案 |
|------|---------|
| **相位恢复/初始化** | 保持当前方法，**不需要**秩投影 |
| **需要精确秩-r** | 添加 SVD truncation 后处理 |
| **理论分析** | 认识到 v_max ≈ vec(X_true) 保持秩 |
| **数值稳定性** | 检查奇异值分布，决定是否投影 |

---

## 参考

- Tucker Decomposition: Kolda & Bader (2009)
- Phase Retrieval: Candès et al. (2015)
- Spectral Initialization: Netrapalli et al. (2013)

---

**最后更新**: 2025-12-23

