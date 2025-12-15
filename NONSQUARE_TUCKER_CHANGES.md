# Non-Square Matrix Support for Tucker Spectral Initialization

## Overview
Updated `initialize_tensor_lift_tucker_spectral.m` to support non-square matrices (d1 ≠ d2) in addition to square matrices.

## Changes Made

### 1. **Removed Square Matrix Requirement** (Line ~43)
**Before:**
```matlab
if d1 ~= d2
    error('Tensor lift initialization requires symmetric matrices: d1 must equal d2');
end
d = d1;
```

**After:**
```matlab
m = length(y);
is_square = (d1 == d2);
```

### 2. **Updated Tensor Dimensions** (Line ~166)
**Before:**
```matlab
dims = [d, d, d, d];  % Always square: d×d×d×d
```

**After:**
```matlab
dims = [d1, d2, d1, d2];  % Non-square support: d1×d2×d1×d2
```

### 3. **Updated Tucker Rank Calculation** (Line ~55)
**Before:**
```matlab
tucker_rank = min(5, max(1, floor(d/4)));
```

**After:**
```matlab
tucker_rank = min(5, max(1, floor(min(d1, d2)/4)));
```

### 4. **Conditional Symmetrization**
Updated multiple locations to only symmetrize when `is_square = true`:

**Measurement matrices** (Line ~153):
```matlab
if is_square
    A_cells{i} = (Ai + Ai') / 2;  % Symmetrize
else
    A_cells{i} = Ai;
end
```

**Extracted matrices** (Lines ~357, ~403, ~421):
```matlab
X_current = extract_matrix_from_tucker(T_tucker, is_square);
if is_square
    X_current = (X_current + X_current') / 2;  % Symmetrize
end
```

### 5. **Updated Ground Truth Tensor Creation** (Line ~108)
**Before:**
```matlab
Tstar_full = zeros(d, d, d, d);
for i = 1:d
    for j = 1:d
        for k = 1:d
            for l = 1:d
```

**After:**
```matlab
Tstar_full = zeros(d1, d2, d1, d2);
for i = 1:d1
    for j = 1:d2
        for k = 1:d1
            for l = 1:d2
```

### 6. **Updated Measurement Matrix Extraction** (Line ~140)
**Before:**
```matlab
n = d * d;
...
E_j = reshape(e_j, [d, d]);
...
Ai = reshape(A_matrix(i, :), [d, d]);
```

**After:**
```matlab
n = d1 * d2;
...
E_j = reshape(e_j, [d1, d2]);
...
Ai = reshape(A_matrix(i, :), [d1, d2]);
```

### 7. **Updated `extract_matrix_from_tucker` Function** (Line ~460)
**Function signature:**
```matlab
function X = extract_matrix_from_tucker(T_tucker, is_square)
```

**Key changes:**
- Accepts `is_square` parameter
- Handles d1 ≠ d2 dimensions throughout
- Conditionally symmetrizes T_mat and output X only when `is_square = true`
- Reshapes output to [d1, d2] instead of [d, d]

**Before:**
```matlab
d = T_tucker.dims(1);
...
T_mat = (T_mat + T_mat') / 2;  % Always symmetrize
...
X = reshape(v_lead, [d, d]);
...
X = (X + X') / 2;  % Always symmetrize
```

**After:**
```matlab
d1 = T_tucker.dims(1);
d2 = T_tucker.dims(2);
...
if is_square
    T_mat = (T_mat + T_mat') / 2;  % Conditionally symmetrize
end
...
X = reshape(v_lead, [d1, d2]);
...
if is_square
    X = (X + X') / 2;  % Conditionally symmetrize
end
```

### 8. **Updated Documentation**
- Updated function header comments to reflect non-square support
- Added "Supports both square (d1=d2) and non-square (d1≠d2) matrices" to key advantages
- Updated tensor dimension notation from d×d×d×d to d1×d2×d1×d2

## Backward Compatibility
✓ **Fully backward compatible** - All existing code using square matrices (d1 = d2) will work exactly as before:
- Square matrices are still symmetrized
- Tucker tensor structure unchanged
- Same optimization algorithm

## Testing
Created `test_nonsquare_tucker_spectral.m` to verify:
1. Square matrices (20×20) - baseline test
2. Tall matrices (30×20) - d1 > d2
3. Wide matrices (20×30) - d1 < d2
4. Non-square with higher rank (40×25, r=3)

Test verifies:
- Correct output dimensions
- Square matrices remain symmetric
- Non-square matrices don't enforce symmetry
- Successful recovery with reasonable errors

## Usage Example

### Square Matrix (unchanged behavior)
```matlab
d1 = 20; d2 = 20;
params.r = 3;
[X0, ~, history] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params);
% Output: X0 is 20×20 and symmetric
```

### Non-Square Matrix (new capability)
```matlab
d1 = 30; d2 = 20;
params.r = 3;
[X0, ~, history] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params);
% Output: X0 is 30×20 (not symmetric)
```

## Key Implementation Details

### Tensor Formulation
- **Square case:** X ∈ ℝ^(d×d), T = X ⊗ X ∈ ℝ^(d×d×d×d)
- **Non-square case:** X ∈ ℝ^(d1×d2), T = X ⊗ X ∈ ℝ^(d1×d2×d1×d2)

### Tucker Decomposition
Both cases use: T = G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄

Where:
- G: Core tensor (r×r×r×r)
- U₁, U₃: Factor matrices (d1 × r)
- U₂, U₄: Factor matrices (d2 × r)

### Matrix Extraction
Matricization: T_mat ∈ ℝ^(d1·d2 × d1·d2)
- T_mat = (U₁ ⊗ U₂) G_mat (U₃ ⊗ U₄)^T
- Extract leading eigenvector
- Reshape to d1 × d2

## Files Modified
1. `/Initialization_groundtruth/initialize_tensor_lift_tucker_spectral.m` - Main changes
2. `/test/test_nonsquare_tucker_spectral.m` - New test file

## Next Steps
To use non-square matrices in phase diagram experiments, update:
- `Phasediagram_tensor_nonsym.m` - Remove or update d1 = d2 constraint
- Ground truth generation - Use U ∈ ℝ^(d1×r), V ∈ ℝ^(d2×r), Xstar = UV^T

## Status
✓ Implementation complete
✓ Test framework created
⏳ Ready for testing with `test_nonsquare_tucker_spectral.m`
