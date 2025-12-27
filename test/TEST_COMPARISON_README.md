# Test Comparison: initialize_tensor_lift_tucker_spectral.m

## 目的
验证 `initialize_tensor_lift_tucker_spectral.m` 是否正确使用 solver，并产生与手动调用相同的结果。

## 测试方法

### 方法 A：手动谱初始化 + RGD (Manual Approach)
1. 使用 `TuckerTensor.initialize_spectral()` 进行谱初始化
2. 手动调用 `solve_RGD_tucker_kronecker()` 进行 RGD 优化

### 方法 B：封装函数 (Wrapper Approach)
- 直接调用 `initialize_tensor_lift_tucker_spectral()`
- 该函数内部应该：
  1. 执行谱初始化
  2. 调用 `solve_RGD_tucker_kronecker()` solver

## 预期结果

两种方法应该产生**完全相同**的结果：
- ✅ 初始误差相同
- ✅ 最终误差相同
- ✅ 最终损失相同
- ✅ 整个收敛轨迹相同（逐迭代比较）
- ✅ 最终矩阵相同

## 如何运行测试

```matlab
cd test/
test_spectral_init_plus_rgd
```

## 测试输出解读

### 1. 谱初始化阶段
- 两种方法的初始误差应该完全相同
- 这验证了谱初始化步骤的一致性

### 2. RGD 优化阶段
- 两种方法的每次迭代误差/损失应该完全相同
- 最大差异应该 < 1e-8（数值精度）

### 3. 可视化比较
测试会生成两组图表：

#### 图表组 1：基本收敛曲线
- Loss 收敛
- Error 收敛  
- 误差对比柱状图

#### 图表组 2：详细比较
- Loss 收敛对比（两条曲线应重合）
- Error 收敛对比（两条曲线应重合）
- Loss 差异（应接近机器精度 ~1e-15）
- Error 差异（应接近机器精度 ~1e-15）
- 最终误差对比
- 计算时间对比

## 如果测试失败

### 可能的问题：

1. **初始化不一致**
   - 检查 `initialize_tensor_lift_tucker_spectral` 中的谱初始化是否与手动步骤相同
   - 验证 `y_spectral = y.^2 * sqrt(m)` 转换

2. **Solver 调用参数不同**
   - 检查传递给 `solve_RGD_tucker_kronecker` 的参数
   - 验证步长 `mu`、迭代次数 `T`、测量向量 `y` 等

3. **矩阵提取不一致**
   - 确保使用相同的 `extract_matrix_from_tucker_2` 方法
   - 验证没有多余的对称化操作

4. **随机性问题**
   - 虽然不应该有随机性，但可以尝试不同的 `rng()` 种子

## 配置参数

在 `test_spectral_init_plus_rgd.m` 中可以调整：

```matlab
d1 = 30; d2 = 40;     % 矩阵维度（非方阵）
r = 1;                 % 秩
m = 800;               % 测量数
mu_rgd = 0.1;          % RGD 步长
T_rgd = 200;           % RGD 迭代次数
use_preprocessing = false;  % 预处理开关
```

## 成功标准

所有验证检查应显示 ✓ PASS：
```
Validation Check                                   | Status
---------------------------------------------------|------------------
Initial errors match                               | ✓ PASS
Final errors match                                 | ✓ PASS
Final losses match                                 | ✓ PASS
Error trajectories match (max diff)                | ✓ PASS (< 1e-8)
Loss trajectories match (max diff)                 | ✓ PASS (< 1e-8)
Final matrices match (Frobenius norm diff)         | ✓ PASS (< 1e-8)
```

如果看到：
```
═══════════════════════════════════════════════════════════
✓✓✓ ALL TESTS PASSED - Both methods produce identical results!
═══════════════════════════════════════════════════════════
```

说明 `initialize_tensor_lift_tucker_spectral.m` 正确实现了 solver 集成！

## 技术细节

### 测量转换
Phase retrieval 测量 → Tensor 测量：
```
y_i = |<Ai, X>| / sqrt(m)                    (phase retrieval)
y_spectral_i = <Ai ⊗ Ai, X ⊗ X> / sqrt(m)   (tensor)
             = (<Ai, X>)^2 / sqrt(m)
             = y_i^2 * sqrt(m)
```

### 矩阵提取
使用 `extract_matrix_from_tucker_2`：
- 基于核心张量特征分解（r² × r²）
- 比完整张量特征分解（(d1·d2) × (d1·d2)）更高效
- 支持非方阵矩阵（d1 ≠ d2）
- **不进行对称化**

## 相关文件
- `test_spectral_init_plus_rgd.m` - 主测试文件
- `initialize_tensor_lift_tucker_spectral.m` - 被测试的初始化函数
- `solve_RGD_tucker_kronecker.m` - RGD solver
- `TuckerTensor.m` - Tucker 张量类（谱初始化）
- `TuckerOperator.m` - Tucker 算子类


