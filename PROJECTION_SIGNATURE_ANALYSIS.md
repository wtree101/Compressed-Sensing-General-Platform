# Projection Operator Signature Analysis

## Date: October 26, 2025

## Summary
Analysis of projection operator signatures across the codebase to ensure consistency.

## Projection Function Signatures

### 1. Matrix Projection: `rank_projection`
**Location:** `utilities/rank_projection.m`

**Signature:**
```matlab
function Xl = rank_projection(Xl, r)
```

**Inputs:**
- `Xl` - Matrix to project (d1 x d2)
- `r` - Target rank

**Output:**
- `Xl` - Rank-r projected matrix

**Method:** Truncated SVD

### 2. Tensor Projection: `tensor_projection_rank_r`
**Location:** `utilities/tensor_projection_rank_r.m`

**Signature:**
```matlab
function T_proj = tensor_projection_rank_r(T, r)
```

**Inputs:**
- `T` - 4D tensor to project (d x d x d x d)
- `r` - Target rank for each mode

**Output:**
- `T_proj` - Projected tensor with rank [r, r, r, r]

**Method:** Higher-Order SVD (HOSVD)

## Usage in Code

### 1. In `onetrial_Mat.m` (Matrix Recovery)

**Setup:**
```matlab
params.projection = @(X) rank_projection(X, r);
```

**Signature:** `@(X) where X is (d1 x d2) matrix`

**Used by:**
- Initialization: `initialize_power_method` - Line 70
- Solvers: `solve_PGD`, `solve_AP`, `solve_PGD_amplitude`

**Status:** ✅ **CONSISTENT**

### 2. In `onetrial_MatTensor.m` (Tensor Recovery)

**Setup:**
```matlab
solver_params.projection = @(X) tensor_projection_rank_r(X, r);
```

**Signature:** `@(X) where X is (d x d x d x d) tensor`

**Used by:**
- Solver: `solve_PGD` (tensor version)

**Status:** ✅ **CONSISTENT**

### 3. In `test_power_method.m`

**Setup:**
```matlab
params.projection = @(X) rank_projection(X, rank_true);
```

**Local Implementation:**
```matlab
function X_proj = rank_projection(X, r)
    % ... truncated SVD implementation
end
```

**Status:** ✅ **CONSISTENT** (uses local implementation matching signature)

### 4. In `initialize_power_method.m`

**Usage:**
```matlab
if has_projection
    v_new = params.projection(v_new);  % v_new is (d1 x d2) matrix
end
```

**Expected Signature:** `@(X)` where X is matrix reshaped from vector

**Status:** ✅ **CONSISTENT**

## Signature Consistency Table

| Function | Expected Input | Expected Output | Actual Signature | Status |
|----------|---------------|-----------------|------------------|---------|
| `rank_projection` | Matrix (d1×d2) | Matrix (d1×d2) | `(Xl, r)` | ✅ Match |
| `tensor_projection_rank_r` | Tensor (d×d×d×d) | Tensor (d×d×d×d) | `(T, r)` | ✅ Match |
| Anonymous in `onetrial_Mat` | Matrix | Matrix | `@(X) rank_projection(X, r)` | ✅ Match |
| Anonymous in `onetrial_MatTensor` | Tensor | Tensor | `@(X) tensor_projection_rank_r(X, r)` | ✅ Match |
| Anonymous in tests | Matrix | Matrix | `@(X) rank_projection(X, rank_true)` | ✅ Match |

## Solver Usage Analysis

### Solvers Using Projection

1. **`solve_PGD.m`** (Line 72)
   ```matlab
   Xl = params.projection(Xl_temp);
   ```
   - Expects: `params.projection` to be function handle `@(X)`
   - Works with: Both matrix and tensor (polymorphic)

2. **`solve_AP.m`** (Line 78)
   ```matlab
   Xl = params.projection(Xl_update);
   ```
   - Expects: Same as PGD

3. **`solve_PGD_amplitude.m`** (Line 111)
   ```matlab
   Xl = params.projection(Xl_temp);
   ```
   - Expects: Same as PGD

### Solvers NOT Using Projection

1. **`solve_RGD.m`** - Riemannian Gradient Descent
   - Uses manifold optimization, no explicit projection needed

2. **`solve_GD.m`** - Basic Gradient Descent
   - No projection constraint

3. **`solve_SubGD.m`** - Subspace Gradient Descent
   - No projection constraint

4. **`solve_SGD.m`** - Stochastic Gradient Descent
   - No projection constraint

## Initialization Usage

### `initialize_power_method.m`

**Line 70:**
```matlab
if has_projection
    v_new = params.projection(v_new);  
end
```

**Context:**
- `v_new` is result of `operator.A_star(w)` which returns matrix (d1 x d2)
- Projection is applied to matrix form
- Then vectorized: `v_new = v_new(:)`

**Compatibility:**
```matlab
% Set in calling code:
params.projection = @(X) rank_projection(X, r);

% Called in power method:
v_new = params.projection(v_new);  % v_new is (d1 x d2) matrix

% Equivalent to:
v_new = rank_projection(v_new, r);  // ✅ Correct signature
```

## Wrapper Pattern

All projection calls use the **wrapper pattern** to capture the rank:

```matlab
% Matrix case
params.projection = @(X) rank_projection(X, r);
% Captures 'r' in closure, provides @(X) interface

% Tensor case
params.projection = @(X) tensor_projection_rank_r(X, r);
# Captures 'r' in closure, provides @(X) interface
```

This allows solvers and initialization to call `params.projection(X)` without knowing rank.

## Potential Issues

### ⚠️ None Found

All projection signatures are consistent:
- Matrix recovery uses matrix projection with correct signature
- Tensor recovery uses tensor projection with correct signature
- All wrappers properly capture rank parameter
- Solvers call projection with correct object type

## Recommendations

### ✅ Current Design is Good

The current design is:
1. **Polymorphic**: Same `params.projection(X)` interface for matrix and tensor
2. **Flexible**: Rank captured in closure, not passed each time
3. **Consistent**: All usage follows same pattern
4. **Type-safe**: Matrix solver gets matrix projection, tensor solver gets tensor projection

### Optional Enhancement

If you want explicit type checking, you could add:

```matlab
function Xl = rank_projection(Xl, r)
    % Validate input
    if ndims(Xl) ~= 2
        error('rank_projection expects 2D matrix, got %dD array', ndims(Xl));
    end
    
    [U, S, V] = svd(Xl);
    S_proj = S;
    S_proj(r+1:end, r+1:end) = 0;
    Xl = U * S_proj * V';
end
```

But this is **not necessary** - current code works correctly.

## Conclusion

✅ **All projection signatures are consistent and correct.**

No changes needed. The codebase properly:
- Defines projection functions with appropriate signatures
- Wraps them in anonymous functions capturing rank
- Passes them via `params.projection`
- Calls them uniformly as `params.projection(X)`

The distinction between matrix and tensor projections is handled correctly at the point where `params.projection` is set, not where it's called.
