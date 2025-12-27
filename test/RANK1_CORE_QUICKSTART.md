# Tucker 秩-1 核心方法 - 快速开始

## 🚀 快速运行

```bash
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
matlab -batch "test_tucker_rank1_core"
```

---

## 📝 基本用法

### 最简单的调用

```matlab
% 准备数据
operator = struct('A_cells', A_cells);  % 测量矩阵
y = measurements;                        % 测量向量

% 调用（使用默认参数）
[U_cells, q_opt, G_rank1, history] = tucker_rank1_core_alternating(...
    operator, y, d1, d2, r);

% 提取矩阵
T_final = TuckerTensor([d1,d2,d1,d2], r, 'G', G_rank1, 'U', U_cells);
X_recovered = extract_matrix_from_tucker_2(T_final);
```

### 带参数调用

```matlab
[U_cells, q_opt, G_rank1, history] = tucker_rank1_core_alternating(...
    operator, y, d1, d2, r, ...
    'max_iter', 30, ...      % 最大迭代次数
    'verbose', true, ...     % 显示进度
    'init_method', 'hosvd'); % HOSVD 初始化
```

---

## 📊 检查结果

### 1. 验证秩-1 约束

```matlab
G_mat = reshape(G_rank1, [r*r, r*r]);
fprintf('rank(G_mat) = %d (should be 1)\n', rank(G_mat, 1e-10));

% 奇异值检查
[~, S, ~] = svd(G_mat);
s = diag(S);
fprintf('σ₁ = %.2e, σ₂ = %.2e\n', s(1), s(2));
fprintf('σ₂/σ₁ = %.2e (should be ~0)\n', s(2)/s(1));
```

**期望输出**:
```
rank(G_mat) = 1 (should be 1) ✓
σ₁ = 1.23e+00, σ₂ = 5.67e-12
σ₂/σ₁ = 4.61e-12 (should be ~0) ✓
```

### 2. 检查收敛

```matlab
% 绘制损失曲线
figure;
semilogy(history.loss);
xlabel('Iteration');
ylabel('Loss');
title('Convergence');
grid on;
```

**好的收敛**: 单调下降，平滑  
**坏的收敛**: 震荡，不下降

### 3. 矩阵恢复误差

```matlab
error = norm(X_recovered - X_true, 'fro') / norm(X_true, 'fro');
fprintf('Reconstruction error: %.2e\n', error);
```

**好**: error < 0.1  
**中等**: 0.1 < error < 0.5  
**差**: error > 0.5

---

## ⚙️ 参数调优

### 默认参数（适合大多数情况）

```matlab
'max_iter', 20      % 外层迭代
'u_iter', 5         % U 更新迭代
'q_iter', 10        % q 更新迭代
'mu_u', 0.01        % U 步长
'mu_q', 0.1         % q 步长
'tol', 1e-6         % 收敛容差
```

### 如果不收敛（震荡）

```matlab
% 减小步长
'mu_u', 0.001,   % 从 0.01 → 0.001
'mu_q', 0.01     % 从 0.1 → 0.01
```

### 如果收敛太慢

```matlab
% 增大步长或迭代次数
'max_iter', 50,
'mu_u', 0.05,
'mu_q', 0.5
```

### 如果精度不够

```matlab
% 增加内迭代
'u_iter', 10,
'q_iter', 20,
'tol', 1e-8
```

---

## 🔍 故障排查

### 问题 1: rank(G_mat) ≠ 1

**检查**:
```matlab
[~, S, ~] = svd(G_mat);
disp(diag(S)');  % 查看所有奇异值
```

**如果 σ₂/σ₁ > 1e-6**:
- 可能是数值误差 → 手动投影
- 可能是优化未收敛 → 增加迭代次数
- 可能是秩-1 假设不成立 → 检查理论

**手动投影**:
```matlab
[U, S, V] = svd(G_mat);
G_mat_rank1 = S(1,1) * (U(:,1) * V(:,1)');
G_rank1 = reshape(G_mat_rank1, [r,r,r,r]);
```

### 问题 2: 损失震荡

**症状**: `history.loss` 上下波动

**解决**: 减小步长
```matlab
'mu_u', 0.001,
'mu_q', 0.01
```

### 问题 3: 矩阵恢复误差大

**检查清单**:
1. 测量数是否足够？`m > 10*r` ✓
2. 初始化是否合理？用 `'init_method', 'hosvd'` ✓
3. 是否收敛？查看 `history.loss` ✓
4. 秩-1 假设是否成立？做验证测试 ✓

---

## 📈 与标准方法比较

```matlab
% 标准 HOSVD
y_spectral = y.^2 * sqrt(m);
H_tensor = operator.kronecker_adjoint(y_spectral / sqrt(m));
[G_hosvd, U_hosvd] = HOSVD_with_factors(H_tensor, [r,r,r,r]);

% 秩-1 核心
[U_rank1, q_rank1, G_rank1, ~] = tucker_rank1_core_alternating(...
    operator, y, d1, d2, r);

% 比较
G_mat_hosvd = reshape(G_hosvd, [r*r, r*r]);
G_mat_rank1 = reshape(G_rank1, [r*r, r*r]);

fprintf('Standard HOSVD: rank = %d\n', rank(G_mat_hosvd, 1e-10));
fprintf('Rank-1 Core:    rank = %d ✓\n', rank(G_mat_rank1, 1e-10));
```

---

## 💡 使用建议

### ✓ 推荐使用场景

- 相位恢复问题（理论上 G 是秩-1）
- 需要高精度恢复
- 测量充足（m > 10r²）
- 可以接受较长计算时间

### ✗ 不推荐场景

- G 明显不是秩-1
- 需要极快速度
- 测量不足（m < 5r²）
- 噪声极大

### 🔬 验证秩-1 假设

在使用前，先检查：

```matlab
% 标准 HOSVD
[G_hosvd, ~] = HOSVD_with_factors(H_tensor, [r,r,r,r]);
G_mat = reshape(G_hosvd, [r*r, r*r]);

% 秩-1 近似误差
[U, S, V] = svd(G_mat);
G_rank1_approx = S(1,1) * (U(:,1) * V(:,1)');
error = norm(G_mat - G_rank1_approx, 'fro') / norm(G_mat, 'fro');

fprintf('Rank-1 approximation error: %.2f%%\n', error * 100);
```

**判断**:
- error < 10%: ✓ 使用秩-1 方法
- 10% < error < 30%: ~ 可尝试
- error > 30%: ✗ 用标准方法

---

## 📚 更多信息

- **详细文档**: `RANK1_CORE_METHOD.md`
- **测试脚本**: `test_tucker_rank1_core.m`
- **核心函数**: `../Initialization_groundtruth/tucker_rank1_core_alternating.m`

---

## 🎯 典型工作流程

```matlab
%% 1. 准备数据
operator = struct('A_cells', A_cells);
y = measurements;

%% 2. 验证秩-1 假设（可选但推荐）
y_spectral = y.^2 * sqrt(m);
H = operator.kronecker_adjoint(y_spectral / sqrt(m));
[G_test, ~] = HOSVD_with_factors(H, [r,r,r,r]);
G_mat_test = reshape(G_test, [r*r, r*r]);
[U_t, S_t, V_t] = svd(G_mat_test);
rank1_error = norm(G_mat_test - S_t(1,1)*(U_t(:,1)*V_t(:,1)'), 'fro') / norm(G_mat_test, 'fro');
fprintf('Rank-1 assumption error: %.2f%%\n', rank1_error * 100);

if rank1_error < 0.3  % < 30%
    fprintf('✓ Rank-1 assumption is reasonable\n');
    
    %% 3. 运行秩-1 核心方法
    [U_cells, q_opt, G_rank1, history] = tucker_rank1_core_alternating(...
        operator, y, d1, d2, r, ...
        'max_iter', 30, ...
        'verbose', true);
    
    %% 4. 提取矩阵
    T_final = TuckerTensor([d1,d2,d1,d2], r, 'G', G_rank1, 'U', U_cells);
    X_recovered = extract_matrix_from_tucker_2(T_final);
    
    %% 5. 验证结果
    fprintf('\nFinal Results:\n');
    fprintf('  rank(G_mat) = %d (should be 1)\n', rank(reshape(G_rank1,[r*r,r*r]), 1e-10));
    fprintf('  Final loss = %.6e\n', history.loss(end));
    
else
    fprintf('✗ Rank-1 assumption does not hold\n');
    fprintf('  Use standard HOSVD instead\n');
end
```

---

**版本**: 1.0  
**日期**: 2025-12-24

