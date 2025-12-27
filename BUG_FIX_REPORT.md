# Bug Fix Report: initialize_tensor_lift_tucker_spectral.m

## 🐛 Bug 描述

在 `initialize_tensor_lift_tucker_spectral.m` 中，调用 RGD solver 时传递了错误的测量向量。

## 📍 Bug 位置

**文件**: `Initialization_groundtruth/initialize_tensor_lift_tucker_spectral.m`  
**行号**: 第 224 行

### 错误代码（修复前）：
```matlab
% Line 224 (WRONG!)
[solver_output, T_tucker] = solve_RGD_tucker_kronecker(T_tucker, [], y, tucker_op, [], [], [], m, solver_params);
```

### 正确代码（修复后）：
```matlab
% Line 224 (CORRECT!)
[solver_output, T_tucker] = solve_RGD_tucker_kronecker(T_tucker, [], y_spectral, tucker_op, [], [], [], m, solver_params);
```

## 🔍 问题分析

### 根本原因
函数使用了**两种不同的测量格式**：

1. **Phase Retrieval 测量** (`y`)：
   - 格式：`y_i = |<Ai, X>| / sqrt(m)`
   - 这是输入函数的原始测量值
   
2. **Tensor 测量** (`y_spectral`)：
   - 格式：`y_spectral_i = <Ai ⊗ Ai, X ⊗ X> / sqrt(m)`
   - 转换公式：`y_spectral = y.^2 * sqrt(m)`
   - 这是张量空间中的等价测量

### Bug 的影响

函数分两个阶段工作：

#### 阶段 1：谱初始化（第 194 行）✅ 正确
```matlab
[U_cell_init, G_init] = T_tucker.initialize_spectral(spectral_operator, y_spectral, m, 'pre_func', pre_func);
```
**正确使用了** `y_spectral`

#### 阶段 2：RGD 优化（第 224 行）❌ 错误
```matlab
[solver_output, T_tucker] = solve_RGD_tucker_kronecker(T_tucker, [], y, tucker_op, [], [], [], m, solver_params);
```
**错误使用了** `y` 而不是 `y_spectral`

**结果**：
- 谱初始化针对张量问题进行优化（使用 `y_spectral`）
- RGD 却在解决不同的问题（使用原始 `y`）
- 两个阶段不一致，导致 RGD 无法从谱初始化的良好起点收敛

## 📊 测量转换的数学原理

Phase retrieval 问题：
```
y_i = |<Ai, X>| / sqrt(m)
```

转换为张量问题（X ⊗ X 恢复）：
```
T = X ⊗ X  (4阶张量)
<Ai ⊗ Ai, T> = <Ai ⊗ Ai, X ⊗ X> = (<Ai, X>)^2
```

因此：
```
y_spectral_i = (<Ai, X>)^2 / sqrt(m) 
             = (y_i * sqrt(m))^2 / sqrt(m)
             = y_i^2 * sqrt(m)
```

## ✅ 修复验证

### 修复前的症状：
- Method A（手动）和 Method B（wrapper）产生不同结果
- 初始误差可能相同，但收敛轨迹完全不同
- 最终误差差异显著

### 修复后的预期：
- Method A 和 Method B 产生**完全相同**的结果
- 收敛轨迹逐点匹配（差异 < 1e-8）
- 最终矩阵的 Frobenius 范数差异 < 1e-8

### 如何验证修复

运行测试：
```matlab
cd test/
test_spectral_init_plus_rgd
```

查看输出中的对比表，应该显示：
```
═══════════════════════════════════════════════════════════
✓✓✓ ALL TESTS PASSED - Both methods produce identical results!
═══════════════════════════════════════════════════════════
```

## 🎯 关键要点

1. **一致性至关重要**：如果谱初始化使用 `y_spectral`，RGD 也必须使用 `y_spectral`

2. **张量提升方法的约定**：
   - 对于 `X ⊗ X` 张量恢复问题
   - 始终使用 `y_spectral = y.^2 * sqrt(m)` 作为测量

3. **变量命名建议**：
   - `y` → phase retrieval 测量
   - `y_spectral` → 张量测量
   - 始终明确使用哪一个

## 📝 代码审查检查清单

在类似的初始化函数中，确保：

- [ ] 创建了 `y_spectral = y.^2 * sqrt(m)`
- [ ] 谱初始化使用 `y_spectral`
- [ ] 所有后续优化步骤使用 `y_spectral`
- [ ] 不混用 `y` 和 `y_spectral`
- [ ] 添加注释说明使用哪种测量格式

## 🔗 相关文件

- `Initialization_groundtruth/initialize_tensor_lift_tucker_spectral.m` - 修复的主文件
- `solver/solve_RGD_tucker_kronecker.m` - RGD solver
- `test/test_spectral_init_plus_rgd.m` - 验证测试
- `test/TEST_COMPARISON_README.md` - 测试文档

## 📅 修复日期

2025-12-26

## ✍️ 修复者

Assistant (Claude)

---

**总结**：这是一个微妙但关键的 bug，源于在不同阶段使用了不一致的测量格式。修复很简单（一个字符的改变），但影响重大，直接决定了算法能否正确收敛。


