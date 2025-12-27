# Tucker非对称测试：变更前后对比

## 版本对比

| 特性 | 旧版本 (v1.0) | 新版本 (v2.0) |
|------|---------------|---------------|
| **矩阵维度** | d×d (方阵) | d₁×d₂ (任意) |
| **对称化** | X = (X+X')/2 | 无对称化 |
| **张量生成** | 随机Tucker张量 | **谱初始化 H=Σyᵢ(Aᵢ⊗Aᵢ)** |
| **因子来源** | 随机orth矩阵 | **HOSVD(H)提取** |
| **部分对称性** | 手动设置U₁=U₃ | **自动出现(理论)** |
| **实际场景** | 合成测试 | **相位恢复流程** |

## 测试流程对比

### v1.0: 随机Tucker张量
```matlab
% 步骤1: 随机生成因子
U1 = orth(randn(d, r));
U2 = orth(randn(d, r));
U3 = U1;  % 手动设置
U4 = U2;  % 手动设置

% 步骤2: 随机核心
G = randn(r,r,r,r);

% 步骤3: 构建张量
H = G ×₁U₁ ×₂U₂ ×₃U₃ ×₄U₄

% 步骤4: 测试公式...
```

❌ **问题**: 
- 不反映真实使用场景
- 手动强制部分对称性
- 与 `initialize_spectral` 流程不一致

### v2.0: 真实谱初始化
```matlab
% 步骤1: 真实数据
X_true = randn(d1, d2) / norm_fro
A_cells{i} = 测量矩阵 (i=1...m)
y(i) = |<A_cells{i}, X_true>|²

% 步骤2: 谱初始化 (就像 initialize_spectral!)
H = zeros(d1, d2, d1, d2);
for i = 1:m
    H = H + y_spectral(i) * (A_cells{i} ⊗ A_cells{i});
end

% 步骤3: HOSVD提取因子
[U_cell, ~, G] = HOSVD_with_factors(H, [r,r,r,r]);
U1 = U_cell{1};  % 自动提取
U2 = U_cell{2};
U3 = U_cell{3};  % 应该 ≈ U1
U4 = U_cell{4};  % 应该 ≈ U2

% 步骤4: 验证 U1≈U3, U2≈U4
% 步骤5: 测试公式...
```

✅ **优点**:
- 模拟真实相位恢复
- 部分对称性自然出现
- 与 `initialize_tensor_lift_tucker_spectral.m` 一致
- 验证理论预测

## 代码变更对比

### 张量生成

#### v1.0: 随机生成
```matlab
if strcmp(test_case, 'partial')
    X_true = randn(d, d);
    X_true = (X_true + X_true') / 2;  % 对称化
    v_true = X_true(:);
    H_mat = v_true * v_true';
    H_tensor = reshape(H_mat, [d, d, d, d]);
end
```

#### v2.0: 谱初始化
```matlab
% 真实测量
X_true = randn(d1, d2) / norm(X_true, 'fro');  % 无对称化
A_cells{i} = randn(d1, d2) / sqrt(d1*d2);
y(i) = abs(sum(sum(A_cells{i} .* X_true)))^2;

% 谱张量
H_tensor = zeros(d1, d2, d1, d2);
y_spectral = y.^2 * sqrt(m);
for i = 1:m
    Ai = A_cells{i};
    for i1 = 1:d1
        for i2 = 1:d2
            for j1 = 1:d1
                for j2 = 1:d2
                    H_tensor(i1,i2,j1,j2) = H_tensor(i1,i2,j1,j2) + ...
                        y_spectral(i) * Ai(i1,i2) * Ai(j1,j2);
                end
            end
        end
    end
end
```

### 因子矩阵获取

#### v1.0: 随机生成
```matlab
U1 = orth(randn(d, r));
U2 = orth(randn(d, r));
U3 = U1;  % 手动复制
U4 = U2;  % 手动复制
```

#### v2.0: HOSVD提取
```matlab
[U_cell, ~, G_hosvd] = HOSVD_with_factors(H_tensor, [r,r,r,r]);
U1 = U_cell{1};  % 自动从H提取
U2 = U_cell{2};
U3 = U_cell{3};  % 应≈U1 (理论预测)
U4 = U_cell{4};  % 应≈U2 (理论预测)
```

## 数学理论对比

### v1.0: 无理论背景
- 纯数值测试
- 人为设置部分对称性
- 不验证任何理论命题

### v2.0: 验证谱初始化理论

#### 理论命题1: 谱张量结构
```
H = Σᵢ yᵢ*(Aᵢ⊗Aᵢ) 
  ≈ E[|⟨A,X⟩|² * (A⊗A)]  (当m→∞)
  = λ*(X⊗X)  (在高斯测量下)
```

#### 理论命题2: 部分对称性
```
如果 H ≈ λ*(X⊗X), X ∈ ℝ^{d₁×d₂}
则 HOSVD(H) = {U₁,U₂,U₃,U₄,G} 满足:
  - U₁ ≈ U₃ (span同样的行空间)
  - U₂ ≈ U₄ (span同样的列空间)
```

#### 测试验证
```matlab
diff_U13 = norm(U1 - U3, 'fro');
diff_U24 = norm(U2 - U4, 'fro');

预期:
- 理想 (m→∞): < 1e-10
- 实际 (m=200): 0.01 ~ 0.5
- 差 (m<100): > 0.5
```

## 与主代码的对应

### initialize_tensor_lift_tucker_spectral.m

| 行号 | 代码 | 测试中的对应 |
|------|------|--------------|
| 138-156 | 提取测量矩阵 A_cells | Step 1 |
| 186-188 | y_spectral = y.^2*sqrt(m) | Step 2 |
| 200 | initialize_spectral(...) | Step 2 (完整实现) |
| - | 内部HOSVD | Step 3 |
| 236 | tucker_op.forward | - |
| 423 | extract_matrix_from_tucker | Step 7 |

### 流程完全一致！

**v1.0**: 测试 ≠ 实际代码  
**v2.0**: 测试 = 实际代码流程 ✅

## 预期结果变化

### v1.0预期
```
||U1 - U3||_F = 0.000000e+00  (手动设置)
||U2 - U4||_F = 0.000000e+00  (手动设置)
H_mat对称: 是 (X对称化)
G_mat对称: 是
```

### v2.0预期
```
||U1 - U3||_F = 1.23e-02 ~ 5.0e-01  (自然出现)
||U2 - U4||_F = 1.23e-02 ~ 5.0e-01  (自然出现)
H_mat对称: 否 (d₁≠d₂, X非对称)
G_mat对称: 否 (d₁≠d₂)
公式验证: < 1e-10  (仍然精确!)
```

### 关键区别

| 项目 | v1.0 | v2.0 |
|------|------|------|
| U₁=U₃ | 完全相等(人为) | 近似相等(自然) |
| 对称性 | 强制对称 | 允许非对称 |
| 真实性 | 合成测试 | 真实场景 |

## 运行对比

### v1.0
```matlab
test_tucker_nonsymmetric  % 测试随机张量
```
输出: 完美的数值结果，但无实际意义

### v2.0
```matlab
test_tucker_nonsymmetric  % 测试谱初始化
```
输出: 真实的数值行为，验证理论预测

## 为什么v2.0更好？

### 1. **真实性** ✨
- 测试真实的 `initialize_spectral` 流程
- 从测量数据到最终矩阵的完整链条

### 2. **理论验证** 📐
- 验证部分对称性出现的理论原因
- 检验有限样本效应 (m有限)

### 3. **调试价值** 🐛
- 如果主代码有bug，测试会发现
- 可以调整m、r等参数看影响

### 4. **文档价值** 📚
- 测试代码即文档
- 展示 `initialize_spectral` 如何工作

### 5. **数值现实** 🔢
- 不是完美的0误差
- 展示真实数值行为

## 升级建议

如果你的代码还在用v1.0风格:

```matlab
% ❌ 不要这样
U1 = orth(randn(d, r));
U3 = U1;  % 手动强制

% ✅ 改成这样
% 1. 从测量数据开始
X_true = ...
A_cells = ...
y = ...

% 2. 谱初始化
H = sum_i y_spectral(i) * (A_cells{i} ⊗ A_cells{i})

% 3. HOSVD提取
[U_cell, ~, G] = HOSVD_with_factors(H, [r,r,r,r]);

% 4. 验证部分对称性
diff_U13 = norm(U_cell{1} - U_cell{3}, 'fro');
fprintf('Partial symmetry: %.2e\n', diff_U13);
```

## 总结

| 方面 | v1.0 | v2.0 |
|------|------|------|
| 真实性 | ⭐ | ⭐⭐⭐⭐⭐ |
| 理论性 | ⭐ | ⭐⭐⭐⭐⭐ |
| 实用性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 调试价值 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**结论**: v2.0是真正有意义的测试！🎉

---
**创建日期**: 2025-12-23  
**测试文件**: `test_tucker_nonsymmetric.m`  
**当前版本**: v2.0

