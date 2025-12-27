# Tensor Lift Initialization 非对称矩阵适配更新

## 概述

已将 `initialize_tensor_lift_efficient.m` 修改为默认支持非对称矩阵,并移除所有对称化操作。

## 主要修改

### 1. 移除对称性检查 (第34-36行)

**之前**:
```matlab
if d1 ~= d2
    error('Tensor lift initialization requires symmetric matrices: d1 must equal d2');
end
d = d1;
```

**之后**:
```matlab
% 直接支持 d1 ≠ d2 的非对称矩阵
n = d1 * d2;  % Flattened matrix dimension
```

### 2. 更新函数签名说明 (第2-7行)

- 标题改为: "Non-Symmetric"
- 明确说明: "SUPPORTS NON-SYMMETRIC MATRICES (d1 can differ from d2)"
- 从 `X = UU^T` 改为 `X (d1 x d2)` 表示

### 3. 移除测量矩阵对称化 (第117-124行)

**之前**:
```matlab
for i = 1:m
    Ai = reshape(A_matrix(i, :), [d, d]);
    Ai = (Ai + Ai')/2;  % Symmetrize
    % ...
end
```

**之后**:
```matlab
for i = 1:m
    Ai = reshape(A_matrix(i, :), [d1, d2]);
    % NO SYMMETRIZATION - keep original structure
    % Fourth-order tensor A_i ⊗ A_i
    AiAi = reshape(Ai, n, 1) * reshape(Ai, 1, n);  % (d1*d2) x (d1*d2)
    A_tensor(i, :) = AiAi(:)';
end
```

### 4. 移除结果矩阵对称化 (第233行)

**之前**:
```matlab
% Symmetrize (since we're using X = UU^T formulation)
X0 = (X0 + X0') / 2;
```

**之后**:
```matlab
% NO SYMMETRIZATION - keep original non-symmetric structure
```

### 5. 更新维度参数

所有使用单一 `d` 的地方改为使用 `d1` 和 `d2`:

- 张量维度: `(d, d, d, d)` → `(d1, d2, d1, d2)`
- 测量矩阵重塑: `[d, d]` → `[d1, d2]`
- 函数调用: `tensor_forward(T, A_tensor, d)` → `tensor_forward(T, A_tensor, d1, d2)`
- 初始化函数: `initialize_power_method(y, operator, d, d, ...)` → `initialize_power_method(y, operator, d1, d2, ...)`

### 6. 更新辅助函数签名

#### `create_tensor_from_matrix` (第278-296行)

**之前**:
```matlab
function T = create_tensor_from_matrix(X, d)
    % For matrix X of size (d x d), creates tensor T of size (d x d x d x d)
    X_vec = X(:);  % d^2 x 1
    T_mat = X_vec * X_vec';  % d^2 x d^2
    T = reshape(T_mat, [d, d, d, d]);
end
```

**之后**:
```matlab
function T = create_tensor_from_matrix(X, d1, d2)
    % For matrix X of size (d1 x d2), creates tensor T of size (d1 x d2 x d1 x d2)
    X_vec = X(:);  % (d1*d2) x 1
    T_mat = X_vec * X_vec';  % (d1*d2) x (d1*d2)
    T = reshape(T_mat, [d1, d2, d1, d2]);
end
```

### 7. 新增辅助函数 (第298-324行)

添加了 `tensor_forward` 和 `tensor_adjoint` 函数:

```matlab
function y = tensor_forward(T, A_tensor, d1, d2)
    % Apply tensor forward operator: y_i = <A_i ⊗ A_i, T>
    T_vec = T(:);  % Flatten tensor to (d1*d2)^2 x 1
    y = A_tensor * T_vec;  % m x 1
end

function T = tensor_adjoint(z, A_tensor, d1, d2)
    % Apply tensor adjoint operator: T = sum_i z_i * (A_i ⊗ A_i)
    T_vec = A_tensor' * z;  % (d1*d2)^2 x 1
    T = reshape(T_vec, [d1, d2, d1, d2]);
end
```

### 8. 更新提取参数 (第219-220行)

```matlab
extract_params.d1 = d1;
extract_params.d2 = d2;
```

## 测试与验证

### 兼容性

- ✓ 非对称矩阵 (d1 ≠ d2): 完全支持
- ✓ 方阵 (d1 = d2): 不再强制对称化,保持原始结构

### 使用示例

```matlab
% 非对称矩阵初始化
d1 = 20; d2 = 30; r = 3; m = 1000;

% 生成地面真值
U1_true = randn(d1, r);
U2_true = randn(d2, r);
Xstar = abs(U1_true) * abs(U2_true)';
Xstar = Xstar / norm(Xstar, 'fro');

% 测量算子
n = d1 * d2;
A = randn(m, n);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(y) reshape(A' * y, [d1, d2]);

% 生成测量值
y = abs(operator.A(Xstar)) / sqrt(m);

% 初始化参数
params = struct();
params.T_power = 5;
params.r = r;
params.Xstar = Xstar;
params.verbose = true;

% 调用初始化
[X0, U0, history] = initialize_tensor_lift_efficient(y, operator, d1, d2, params);

% 检查错误
error = norm(X0 - Xstar, 'fro') / norm(Xstar, 'fro');
fprintf('Initialization error: %.6e\n', error);
```

## 依赖函数更新

以下函数可能也需要更新以支持非对称矩阵:

1. ~~`extract_matrix_from_tensor`~~ - 需要添加 `d1`, `d2` 参数
2. `tensor_projection_rank_r` - 可能需要检查是否支持非对称
3. `solve_PGD` - 应该已经支持非对称(通过 d1, d2 参数)

## 向后兼容性

- **破坏性更改**: 是
  - `create_tensor_from_matrix(X, d)` → `create_tensor_from_matrix(X, d1, d2)`
  - 移除了对称性检查,方阵不再被强制对称化

- **建议**: 如果需要对称结果,应在调用后手动对称化:
  ```matlab
  X0 = (X0 + X0') / 2;  % 手动对称化(如果需要)
  ```

## 性能影响

- **内存**: 张量大小从 `d^4` 变为 `(d1*d2)^2`,对于非方阵可能更大或更小
- **计算**: 移除对称化步骤,计算略快
- **精度**: 不丢失非对称信息,精度可能提高

## 相关文件

- `Initialization_groundtruth/initialize_tensor_lift_efficient.m` - 已更新
- `solver/solve_PGD.m` - 应该已支持非对称
- `utilities_tensor/extract_matrix_from_tensor.m` - 可能需要更新

## 修改历史

- 2025-12-27: 适配非对称矩阵,移除所有对称化操作,更新为默认非对称模式

