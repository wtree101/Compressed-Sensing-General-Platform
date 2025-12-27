# Extract Matrix From Tensor 非对称适配更新

## 概述

已将 `utilities_tensor/extract_matrix_from_tensor.m` 修改为默认支持非对称矩阵,并移除强制对称化操作。

## 主要修改

### 1. 函数签名和文档更新 (第1-23行)

**之前**:
```matlab
% Extract matrix X from 4th order tensor T
% for symmetric tensor phase retrieval problems
% Input:  T - 4th order tensor of size d x d x d x d
% Output: X - Extracted symmetric matrix of size d x d with rank <= r
% params.symmetrize - Whether to enforce symmetry (optional, default: true)
```

**之后**:
```matlab
% Extract matrix X from 4th order tensor T (Non-Symmetric)
% SUPPORTS NON-SYMMETRIC MATRICES (d1 can differ from d2)
% Input:  T - 4th order tensor of size d1 x d2 x d1 x d2
% Output: X - Extracted matrix of size d1 x d2 with rank <= r
% params.d1, params.d2 - Row/column dimensions (optional, inferred from T)
```

### 2. 移除对称性检查和强制对称化 (第25-54行)

**之前**:
```matlab
[d1, d2, d3, d4] = size(T);
if d1 ~= d2 || d2 ~= d3 || d3 ~= d4
    error('Tensor must be d x d x d x d');
end
d = d1;

% ...
if params.symmetrize  % default: true
    X = (X + X') / 2;
    X = project_symmetric_low_rank(X, r);
end
```

**之后**:
```matlab
[d1, d2, d3, d4] = size(T);
% Check tensor structure: T should be d1 x d2 x d1 x d2
if d1 ~= d3 || d2 ~= d4
    error('Tensor must be d1 x d2 x d1 x d2 (currently [%d,%d,%d,%d])', d1, d2, d3, d4);
end

% Allow d1, d2 to be provided in params (for verification)
if isfield(params, 'd1') && params.d1 ~= d1
    warning('params.d1=%d does not match tensor dimension d1=%d', params.d1, d1);
end

% NO SYMMETRIZATION - keep original non-symmetric structure
% Apply low-rank projection if needed
if rank(X) > r
    [U, S, V] = svd(X);
    X = U(:, 1:r) * S(1:r, 1:r) * V(:, 1:r)';
end
```

### 3. 更新提取方法函数签名

所有提取方法从 `(T, d, r, params)` 改为 `(T, d1, d2, r, params)`:

#### `extract_eigen_method` (第81-111行)

**之前**:
```matlab
function X = extract_eigen_method(T, d, ~, params)
    n = d * d;
    T_mat = reshape(T, [n, n]);
    T_mat = (T_mat + T_mat') / 2;  % Symmetrize
    % ...
    X = reshape(v * sqrt(abs(lambda)), [d, d]);
end
```

**之后**:
```matlab
function X = extract_eigen_method(T, d1, d2, ~, params)
    n = d1 * d2;
    T_mat = reshape(T, [n, n]);
    
    % For non-symmetric case, symmetrize only for eigendecomposition stability
    % but don't enforce symmetry on final result
    T_mat_sym = (T_mat + T_mat') / 2;
    % Use T_mat_sym for eigen, but return non-symmetric result
    
    % Reshape to matrix form (d1 x d2)
    X = reshape(v * sqrt(abs(lambda)), [d1, d2]);
end
```

**关键区别**:
- 对称化仅用于特征分解的数值稳定性
- 最终结果保持非对称结构 (d1 x d2)

#### `extract_svd_method` (第113-139行)

**之前**:
```matlab
function X = extract_svd_method(T, d, ~, params)
    n = d * d;
    % ...
    % For symmetric case, u1 ≈ v1, so use average
    x_vec = (u1 + v1) / 2 * sqrt(S(1,1));
    X = reshape(x_vec, [d, d]);
end
```

**之后**:
```matlab
function X = extract_svd_method(T, d1, d2, ~, params)
    n = d1 * d2;
    % ...
    % For non-symmetric case, average u1 and v1 for stability
    x_vec = (u1 + v1) / 2 * sqrt(S(1,1));
    X = reshape(x_vec, [d1, d2]);
end
```

#### `extract_hosvd_method` (第141-154行)

**之前**:
```matlab
function X = extract_hosvd_method(T, d, r, params)
    % HOSVD preprocessing method (for noisy tensors)
    % ...
    X = extract_eigen_method(T_clean, d, r, params);
end
```

**之后**:
```matlab
function X = extract_hosvd_method(T, d1, d2, r, params)
    % HOSVD preprocessing method (for noisy tensors, non-symmetric case)
    % ...
    X = extract_eigen_method(T_clean, d1, d2, r, params);
end
```

### 4. 移除的参数

- **`params.symmetrize`**: 不再支持,强制对称化已移除
- **依赖函数**: `project_symmetric_low_rank(X, r)` 不再调用

### 5. 新增的参数

- **`params.d1`**: 可选的行维度(用于验证)
- **`params.d2`**: 可选的列维度(用于验证)

如果提供这些参数,函数会验证它们是否与张量实际维度匹配。

## 数值稳定性说明

### 为什么在 `extract_eigen_method` 中仍然对称化?

```matlab
% For non-symmetric case, symmetrize only for eigendecomposition stability
% but don't enforce symmetry on final result
T_mat_sym = (T_mat + T_mat') / 2;
```

**原因**:
1. **数值稳定性**: 特征分解对对称矩阵更稳定,特征值保证为实数
2. **理论依据**: 对于张量 T = X ⊗ X,其矩阵化 T_mat 理论上应该是对称的
3. **不强制结果**: 对称化仅用于计算,最终结果 X 仍然是 d1×d2 的非对称矩阵

如果 T_mat 严重非对称,这可能表示:
- 张量不完全符合 T = X ⊗ X 结构
- 存在数值误差或噪声
- 需要更多迭代或更好的初始化

## 使用示例

### 非对称矩阵提取

```matlab
% 生成非对称张量
d1 = 20; d2 = 30; r = 3;
X_true = randn(d1, r) * randn(r, d2);
X_true = X_true / norm(X_true, 'fro');

% 创建张量 T = X ⊗ X
X_vec = X_true(:);  % (d1*d2) x 1
T_mat = X_vec * X_vec';  % (d1*d2) x (d1*d2)
T = reshape(T_mat, [d1, d2, d1, d2]);  % d1 x d2 x d1 x d2

% 提取参数
params = struct();
params.r = r;
params.d1 = d1;  % Optional: for verification
params.d2 = d2;  % Optional: for verification
params.method = 'eig';
params.verbose = true;

% 提取矩阵
X_extracted = extract_matrix_from_tensor(T, params);

% 检查维度和误差
fprintf('Extracted matrix size: [%d, %d]\n', size(X_extracted, 1), size(X_extracted, 2));
fprintf('Extraction error: %.6e\n', norm(X_extracted - X_true, 'fro') / norm(X_true, 'fro'));
```

### 方阵(不强制对称)

```matlab
% 方阵但不强制对称
d1 = 30; d2 = 30; r = 5;
X_true = randn(d1, r) * randn(r, d2);  % 非对称

% ... (与上述相同的流程)

% 检查是否保持非对称
fprintf('Symmetry error: %.6e\n', norm(X_extracted - X_extracted', 'fro'));
% 输出可能不为0,表示保持了非对称结构
```

## 向后兼容性

### 破坏性更改

1. **参数移除**: `params.symmetrize` 不再支持
   - **之前**: `params.symmetrize = true/false`
   - **之后**: 总是返回非对称结果

2. **函数签名**: 辅助函数签名改变
   - **之前**: `extract_*_method(T, d, r, params)`
   - **之后**: `extract_*_method(T, d1, d2, r, params)`

3. **返回值**: 不再保证对称
   - **之前**: 对于方阵,总是返回对称矩阵
   - **之后**: 返回非对称矩阵(d1 x d2)

### 迁移指南

如果需要对称结果,手动对称化:

```matlab
X = extract_matrix_from_tensor(T, params);

% 如果需要对称结果(仅适用于方阵)
if size(X, 1) == size(X, 2)
    X = (X + X') / 2;
    
    % 如果还需要低秩投影
    [U, S, V] = svd(X);
    r = params.r;
    X = U(:, 1:r) * S(1:r, 1:r) * V(:, 1:r)';
end
```

## 性能影响

- **内存**: 支持非对称矩阵,内存使用取决于 d1 × d2
- **计算**: 移除 `project_symmetric_low_rank`,计算略快
- **精度**: 不丢失非对称信息,精度可能提高

## 相关文件

- `utilities_tensor/extract_matrix_from_tensor.m` - 已更新
- `Initialization_groundtruth/initialize_tensor_lift_efficient.m` - 调用此函数
- `utilities_tensor/project_symmetric_low_rank.m` - 不再依赖

## 测试建议

```matlab
% 测试非对称情况
test_extract_nonsym();

function test_extract_nonsym()
    d1 = 15; d2 = 25; r = 3;
    
    % 生成地面真值
    X_true = randn(d1, r) * randn(r, d2);
    X_true = X_true / norm(X_true, 'fro');
    
    % 创建张量
    X_vec = X_true(:);
    T_mat = X_vec * X_vec';
    T = reshape(T_mat, [d1, d2, d1, d2]);
    
    % 提取
    params = struct('r', r, 'verbose', true);
    X_extracted = extract_matrix_from_tensor(T, params);
    
    % 验证
    error = norm(X_extracted - X_true, 'fro') / norm(X_true, 'fro');
    fprintf('Test passed: error = %.6e\n', error);
    assert(error < 1e-10, 'Extraction error too large');
end
```

## 修改历史

- 2025-12-27: 适配非对称矩阵,移除强制对称化,更新为默认非对称模式

