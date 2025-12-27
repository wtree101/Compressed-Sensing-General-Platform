# 核心张量对角结构 - 快速参考

## ✅ 正确定义

```
G(i,j,k,l) ≈ 0  unless  i=j AND k=l
```

**含义**: 只有 `G(i,i,k,k)` 是主要非零元素

---

## 常见错误 ❌

```
G(i,j,k,l) ≈ 0  unless  i=k AND j=l  ← 错误！
```

---

## 代码示例

### 提取对角元素
```matlab
G_diag = zeros(r, r);
for i = 1:r
    for k = 1:r
        G_diag(i,k) = G(i,i,k,k);  % ✅ 正确
    end
end
```

### Projected Power Method
```matlab
% 投影到对角支撑
for k = 1:r
    idx = (k-1)*r + k;  % 对应 (k,k) 位置
    q_proj(idx) = q(idx);
end
```

---

## 记忆技巧

```
4D 张量 G(i,j,k,l):
├─ 第1,2维 (i,j): X 的第一个拷贝
└─ 第3,4维 (k,l): X 的第二个拷贝

对角结构:
├─ i=j: 第一个拷贝的对角
└─ k=l: 第二个拷贝的对角

∴ G(i,i,k,k) 是"双对角"元素
```

---

## 矩阵化索引

在 G_mat (r²×r²) 中：
- 对角支撑: `idx = (k-1)*r + k` for k=1,2,...,r
- 共 r 个对角位置（不是 r² 个！）
- 对应 G(1,1,*,*), G(2,2,*,*), ..., G(r,r,*,*)

---

**最后更新**: 2025-12-23  
**状态**: ✅ 已修正所有文档和注释

