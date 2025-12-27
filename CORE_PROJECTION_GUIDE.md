# Core Projection Feature: Implementation and Testing Guide

## 概述

核心投影 (Core Projection) 是一个可选功能，将 Tucker 张量的核心投影到秩-1对角结构上。这可能改善收敛性能。

## 理论基础

对于张量提升问题 T = X ⊗ X，理论上核心张量应具有特殊的对角结构。核心投影强制执行这种结构：

```
Core_proj = reshape(q*q', [r, r, r, r])
```

其中 `q` 是从核心张量 G 的矩阵化形式 G_mat (r²×r²) 提取的主特征向量。

## 实现位置

### 1. Solver: `solve_RGD_tucker_kronecker.m`

添加了新的可选参数 `use_core_projection`:

```matlab
params = struct(
    'T', 200,                      % 迭代次数
    'mu', 0.1,                     % 步长
    'Xstar', Xstar,                % 真值 (可选)
    'verbose', false,              % 详细输出
    'use_core_projection', false   % 核心投影 (默认: false)
);

[output, T_tucker] = solve_RGD_tucker_kronecker(T_tucker, [], y, tucker_op, [], [], [], m, params);
```

**默认行为**: `use_core_projection = false` (不使用投影)

### 2. TuckerTensor类: `utilities_tensor/TuckerTensor.m`

添加了 `.copy()` 方法用于深拷贝：

```matlab
T_tucker_copy = T_tucker.copy();
```

### 3. 提取函数: `extract_matrix_from_tucker_2()`

现在返回两个输出：

```matlab
[X, Core_proj] = extract_matrix_from_tucker_2(T_tucker);
% X - 提取的矩阵
% Core_proj - 投影后的核心 (秩-1对角结构)
```

## 如何测试

### 快速测试 (推荐先运行)

```matlab
cd test/
test_core_projection_quick
```

**配置**:
- 矩阵维度: d1=20, d2=30
- Tucker秩: r=3  
- 测量数: m=1100
- 迭代数: T=200

**输出**:
- 带/不带投影的最终误差和损失对比
- 收敛曲线图
- 改进百分比
- 明确的建议 (启用或禁用投影)

### 全面测试

```matlab
cd test/
test_core_projection_benefit
```

**测试场景**:
1. 标准情况 (r=3, m=1100, with abs)
2. 较少测量 (r=3, m=800, with abs)
3. 无abs情况 (r=3, m=1100, no abs)
4. 秩-1情况 (r=1, m=800, with abs)

**输出**:
- 所有测试案例的详细对比表
- 多个收敛曲线图
- 改进统计
- 综合建议

## 测试结果解读

### 成功指标

✓ **投影有益**:
```
✓ Error reduced by X.X%
Improvement: ✓ X.X%
```

✗ **投影有害**:
```
✗ Error increased by X.X%
Improvement: ✗ +X.X%
```

### 最终建议

测试会自动输出建议：

```
✓✓✓ RECOMMENDATION: ENABLE core projection (use_core_projection=true)
    Benefits observed in X% of test cases
```

或

```
✗ RECOMMENDATION: Keep core projection DISABLED (default)
    No significant benefit or potential degradation
```

## 如何在实际代码中使用

### 方式 1: 直接使用 Solver

```matlab
% WITHOUT projection (default)
params = struct('T', 200, 'mu', 0.1, 'Xstar', Xstar, 'verbose', false);
[output, T_final] = solve_RGD_tucker_kronecker(T_init, [], y, tucker_op, [], [], [], m, params);

% WITH projection
params = struct('T', 200, 'mu', 0.1, 'Xstar', Xstar, 'verbose', false, ...
                'use_core_projection', true);
[output, T_final] = solve_RGD_tucker_kronecker(T_init, [], y, tucker_op, [], [], [], m, params);
```

### 方式 2: 通过 initialize_tensor_lift_tucker_spectral

该函数内部调用 solver，可通过修改 `initialize_tensor_lift_tucker_spectral.m` 传递参数：

```matlab
% 在 initialize_tensor_lift_tucker_spectral.m 中:
solver_params.use_core_projection = true;  % 如果测试显示有益
```

## 性能考虑

### 计算开销

核心投影增加的额外计算：
- 每次迭代：特征分解 r²×r² 矩阵
- 对于 r=1: 几乎无额外开销
- 对于 r=3: 9×9 矩阵特征分解 (非常快)
- 对于 r=5: 25×25 矩阵特征分解 (仍然很快)

**结论**: 计算开销可忽略不计

### 内存开销

无显著额外内存使用。

### 何时可能有益

理论上，核心投影可能在以下情况有益：
1. **较少测量** (m 接近临界值)
2. **无 abs()** 的真值 (更难的问题)
3. **较高秩** (r > 1)
4. **噪声测量**

## 修改的文件

1. ✅ `solver/solve_RGD_tucker_kronecker.m`
   - 添加 `use_core_projection` 参数
   - 在迭代中应用投影 (如果启用)
   - 更新文档和输出

2. ✅ `utilities_tensor/TuckerTensor.m`
   - 添加 `.copy()` 方法

3. ✅ `test/test_core_projection_quick.m`
   - 快速对比测试
   - 配置: d1=20, d2=30, r=3, m=1100

4. ✅ `test/test_core_projection_benefit.m`
   - 全面测试多个场景
   - 自动生成建议

## 下一步

1. **运行快速测试**:
   ```matlab
   test_core_projection_quick
   ```

2. **查看结果**: 观察是否有改进

3. **运行全面测试** (可选):
   ```matlab
   test_core_projection_benefit
   ```

4. **根据结果决定**:
   - 如果有显著改进: 在代码中启用投影
   - 如果无改进或有害: 保持默认禁用状态

## 相关文档

- `BUG_FIX_REPORT.md` - 之前的bug修复
- `test/TEST_COMPARISON_README.md` - Spectral init + RGD测试文档

---

**作者**: Assistant  
**日期**: 2025-12-27  
**版本**: 1.0

