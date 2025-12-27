# Amplitude Solver 比较测试指南

## 概述

在 `test_core_projection_benefit.m` 中添加了一个新的测试类型,用于比较 Tucker RGD 和标准 RGD amplitude 求解器的性能。

## 新增功能

### 测试类型

现在支持两种测试类型:

1. **`'rgd_tucker'`** (默认): 使用 Tucker RGD 求解器,测试有/无核张量投影的效果
2. **`'amplitude'`**: 使用谱初始化(T=0,无 Tucker RGD)+ `solve_RGD_amplitude` 求解器

### 测试流程

#### Amplitude 测试流程

参考 `matrix_recovery/Phasediagram_tensor_nonsym.m` 的方式:

```matlab
1. 生成地面真值 Xstar 和测量值 y
2. 调用 initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params)
   - params.T = 0  (不进行 Tucker RGD 迭代)
   - 只获取谱初始化结果
3. 使用初始化结果 X_init 作为 solve_RGD_amplitude 的起点
4. 运行 solve_RGD_amplitude(X_init, [], y, operator, d1, d2, r, m, params)
   - params.T = 200 (或其他指定值)
   - params.use_preconditioner = true
```

## 配置示例

### Test Case 配置

```matlab
% Tucker RGD test (测试核投影效果)
struct('name', 'Standard r=3 (with abs)', ...
       'd1', 20, 'd2', 30, 'r', 3, 'm', 1100, ...
       'use_abs', true, 'mu', 0.1, 'T', 200, 'rng_seed', 40, ...
       'test_type', 'rgd_tucker');

% Amplitude solver test (baseline 比较)
struct('name', 'Init + RGD_amplitude (r=3)', ...
       'd1', 20, 'd2', 30, 'r', 3, 'm', 1100, ...
       'use_abs', true, 'mu', 0.1, 'T', 200, 'rng_seed', 40, ...
       'test_type', 'amplitude');
```

## 输出说明

### 结果结构

对于 amplitude 测试:
- `result.no_proj`: 包含 amplitude 求解器的结果
- `result.with_proj`: 设为 NaN (不适用)
- `result.test_type`: 标记为 `'amplitude'`

### 报告输出

Amplitude 测试在比较报告中标记为 `(Ampl.)`,不计入核投影的改进统计。

示例输出:
```
Test Case                      | No Proj Error   | With Proj Error | Improvement
-------------------------------|-----------------|-----------------|---------------
Standard r=3 (with abs)        | 1.234e-03       | 9.876e-04       | ✓ 20.1%
Init + RGD_amplitude (r=3)     | 2.345e-03       |            N/A  | N/A (Ampl.)
```

## 绘图

### 错误收敛图

- Tucker RGD 测试: 显示有/无投影的两条曲线
- Amplitude 测试: 只显示单条曲线 (Init + RGD_amplitude)

### 柱状图

- 第一个子图: 显示最终错误对比
  - Amplitude 测试的 "With Projection" 柱设为 0 (NaN)
- 第二个子图: 显示改进百分比
  - Amplitude 测试标记为 "N/A"

## 比较目的

### Amplitude 测试的作用

1. **Baseline 比较**: 提供标准 RGD amplitude 求解器的基准性能
2. **初始化质量**: 评估谱初始化的质量(T=0)
3. **算法对比**: 对比 Tucker RGD vs. 标准 RGD 的性能差异

### 典型使用场景

- 评估 Tucker 结构是否对特定问题有益
- 对比不同初始化方法 + 求解器组合的效果
- 验证 Tucker RGD 的优势在哪些情况下明显

## 运行测试

```matlab
% 在 MATLAB 中运行
cd /Users/wutong/Documents/MATLAB/GeneralPlatform
run('test/test_core_projection_benefit.m')
```

## 注意事项

1. **测量值转换**: 
   - Tucker RGD 使用 `y_spectral = y.^2 * sqrt(m)`
   - Amplitude solver 使用原始 `y`
   
2. **初始化参数**:
   - Amplitude 测试设置 `T=0` 来禁用 Tucker RGD
   - 确保 `initialize_tensor_lift_tucker_spectral` 支持 `T=0` 参数

3. **统计排除**:
   - Amplitude 测试不计入核投影改进统计
   - 推荐部分只基于 Tucker RGD 测试结果

## 相关文件

- `test/test_core_projection_benefit.m`: 主测试脚本
- `solver/solve_RGD_amplitude.m`: Amplitude 求解器
- `solver/solve_RGD_tucker_kronecker.m`: Tucker RGD 求解器
- `Initialization_groundtruth/initialize_tensor_lift_tucker_spectral.m`: 谱初始化
- `matrix_recovery/Phasediagram_tensor_nonsym.m`: 参考实现

## 修改历史

- 2025-12-27: 添加 amplitude 测试支持,参考 Phasediagram_tensor_nonsym.m 的实现方式

