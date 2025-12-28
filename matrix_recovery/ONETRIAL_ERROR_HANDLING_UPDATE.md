# onetrial_Mat 错误处理增强

## 问题描述

在运行大规模实验时(如 phase diagram 生成),当某些 trial 产生 NaN 或 Inf 值时,会导致整个实验中断:

```matlab
Error using rank
Input matrix must not contain NaN or Inf values.

Error in onetrial_Mat (line 112)
output.recovered_rank = rank(Xl, 1e-6);

Error in multipletrial (line 33)
    parfor i = 1:trial_num
```

这会导致:
- ❌ 整个 parfor 循环终止
- ❌ 已完成的实验数据丢失
- ❌ 需要重新运行整个实验

## 解决方案

添加了全面的错误处理机制,在多个关键点捕获错误并优雅地失败:

### 1. **Solver 错误捕获** (第88-119行)

```matlab
try
    % Use the modular solver
    [solver_output, Xl] = params.alg_func(Xl, Ul, y, operator, d1, d2, r, m, params);
    Error_Stand = solver_output.Error_Stand;
    Error_function = solver_output.Error_function;
catch ME
    % Catch solver errors and mark trial as failed
    if verbose >= 0
        warning('Solver failed: %s', ME.message);
    end
    % Return failure output
    output = create_failure_output(params);
    is_success = 0;
    return;
end
```

**捕获的错误**:
- 求解器内部错误
- 数值不稳定
- 内存错误
- 任何运行时异常

### 2. **NaN/Inf 值检查** (第122-130行)

```matlab
% Check for NaN or Inf in result
if any(isnan(Xl(:))) || any(isinf(Xl(:)))
    if verbose >= 0
        warning('Solution contains NaN or Inf values. Marking trial as failed.');
    end
    output = create_failure_output(params);
    is_success = 0;
    return;
end
```

**检查内容**:
- 解 `Xl` 中的所有元素
- 提前检测问题,避免后续错误

### 3. **符号修正错误捕获** (第133-142行)

```matlab
try
    [final_error, Xl_aligned] = rectify_sign_ambiguity(Xl, Xstar);
catch ME
    if verbose >= 0
        warning('Error rectification failed: %s', ME.message);
    end
    output = create_failure_output(params);
    is_success = 0;
    return;
end
```

**捕获的错误**:
- `rectify_sign_ambiguity` 函数内部错误
- 维度不匹配
- 数值问题

### 4. **最终误差验证** (第144-152行)

```matlab
% Check for NaN in final error
if isnan(final_error) || isinf(final_error)
    if verbose >= 0
        warning('Final error is NaN or Inf. Marking trial as failed.');
    end
    output = create_failure_output(params);
    is_success = 0;
    return;
end
```

### 5. **Rank 计算保护** (第154-162行)

```matlab
% Compute recovered rank safely
try
    recovered_rank = rank(Xl_aligned, 1e-6);
catch ME
    if verbose >= 0
        warning('Rank computation failed: %s. Using NaN.', ME.message);
    end
    recovered_rank = NaN;
end
```

**特点**:
- ✅ 捕获 rank 计算错误(原始错误位置)
- ✅ 使用 NaN 标记失败,而不是中断
- ✅ 继续执行,返回完整的输出结构

### 6. **失败输出生成器** (第184-201行)

```matlab
function output = create_failure_output(params)
    % CREATE_FAILURE_OUTPUT Create a failure output struct
    % Returns an output struct with NaN/Inf values indicating failure
    
    % Get number of iterations if available
    if isfield(params, 'T')
        T = params.T;
    else
        T = 100; % Default
    end
    
    output = struct();
    output.Error_Stand = inf(T, 1);  % Mark as failed convergence
    output.Error_function = inf(T, 1);
    output.Xl_final = [];  % Empty matrix
    output.final_error = inf;  % Failed
    output.recovered_rank = NaN;
end
```

**失败标记**:
- `Error_Stand = inf(T, 1)`: 标记收敛失败
- `Error_function = inf(T, 1)`: 标记损失函数失败
- `Xl_final = []`: 空矩阵
- `final_error = inf`: 无穷大误差
- `recovered_rank = NaN`: 秩未知
- `is_success = 0`: 失败标志

## 工作流程

### 成功 Trial

```
Initialize → Solve → Check NaN/Inf → Rectify Sign → Check Error → Compute Rank → Success
                                                                                    ↓
                                                                            is_success = 1
```

### 失败 Trial (在任何步骤)

```
Initialize → Solve → [ERROR] → create_failure_output() → is_success = 0 → CONTINUE
             ↓
        [CATCH ERROR]
             ↓
      warning(...)
             ↓
     return immediately
```

## 使用效果

### 之前

```matlab
% 运行到 m=128 时崩溃
m=128 (9/16): {Error using rank
Input matrix must not contain NaN or Inf values.
[整个实验中断]
```

### 之后

```matlab
% 单个 trial 失败,继续实验
m=128 (9/16): Warning: Solution contains NaN or Inf values. Marking trial as failed.
              Success: 0/3 (0.0%), Final Error: Inf
m=192 (10/16): [继续运行...]
```

## 数据处理

### 失败 Trial 的识别

在后续分析中,可以通过以下方式识别失败的 trial:

```matlab
% 检查是否失败
if isinf(results.final_error) || isnan(results.recovered_rank)
    fprintf('Trial failed\n');
end

% 统计成功率
num_success = sum(results.success_count);
num_total = length(results.m_values) * trial_num;
success_rate = num_success / num_total;
```

### Phase Diagram 绘制

失败的点会自动显示为失败(success_count = 0),不影响相位图生成。

## 警告信息

### 设置 verbose 级别

```matlab
% verbose = 0: 仍然显示警告(推荐用于实验)
% verbose = -1: 完全静默(不推荐,可能错过重要信息)
% verbose = 1: 显示收敛图和警告
```

### 警告示例

```
Warning: Solver failed: Matrix dimensions must agree.
Warning: Solution contains NaN or Inf values. Marking trial as failed.
Warning: Error rectification failed: Index exceeds matrix dimensions.
Warning: Rank computation failed: Input matrix must not contain NaN or Inf values. Using NaN.
```

## 性能影响

- ✅ **最小开销**: try-catch 仅在错误时触发
- ✅ **快速失败**: 检测到 NaN/Inf 后立即返回
- ✅ **继续实验**: parfor 循环不会中断
- ✅ **数据完整**: 成功的 trial 正常保存

## 兼容性

- ✅ 向后兼容: 成功的 trial 输出格式不变
- ✅ multipletrial: 自动处理失败的 trial
- ✅ Parallel: 在 parfor 中安全使用
- ✅ Phase Diagram: 失败点正确标记

## 调试建议

### 如果出现大量失败

1. **检查初始化**: 初始化可能产生 NaN/Inf
   ```matlab
   params.init = @initialize_power_method;  % 尝试不同的初始化
   ```

2. **降低步长**: 步长过大可能导致数值不稳定
   ```matlab
   params.mu = 0.01;  % 从 0.1 降到 0.01
   ```

3. **检查测量数**: m 太小可能导致病态问题
   ```matlab
   m_min = 2 * d1 * r;  % 确保足够的测量
   ```

4. **查看警告**: 了解失败原因
   ```matlab
   params.verbose = 0;  % 确保显示警告
   ```

## 测试

```matlab
% 测试错误处理
params = struct();
params.m = 100;
params.r = 3;
params.kappa = 2;
params.d1 = 20;
params.d2 = 30;
params.T = 100;
params.verbose = 1;

% 故意制造 NaN
params.alg_func = @(varargin) struct('Error_Stand', nan(100,1), 'Error_function', nan(100,1));

[output, is_success] = onetrial_Mat(params);

assert(is_success == 0, 'Should mark as failed');
assert(isinf(output.final_error), 'Should return inf error');
fprintf('✓ Error handling test passed\n');
```

## 总结

### 改进前
- ❌ 单个 trial 失败 → 整个实验崩溃
- ❌ 需要手动重启实验
- ❌ 丢失已完成的数据

### 改进后
- ✅ 单个 trial 失败 → 标记为失败,继续实验
- ✅ 自动处理所有错误
- ✅ 保留所有成功的数据
- ✅ 提供清晰的失败信息

## 修改历史

- 2025-12-27: 添加全面的错误处理,捕获 NaN/Inf 错误,确保实验连续性

