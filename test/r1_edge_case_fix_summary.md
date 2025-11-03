# r=1 Edge Case Handling - Summary

## Problem Description

When Tucker rank r=1, MATLAB automatically collapses singleton dimensions, which can cause issues in tensor operations:

- A 4D tensor of size [1,1,1,1] may be stored as a scalar
- `size()` returns [1,1] instead of [1,1,1,1]
- `ndims()` returns 2 instead of 4
- Operations expecting 4D tensors fail

## Critical Areas Fixed

### 1. TuckerOperator.m - forward_kronecker()

**Issue**: Line 142 - Reshaping G to matrix form
```matlab
G_mat = reshape(permute(G, [1,2,3,4]), [r*r, r*r]);
```

**Fix**: Added explicit handling for r=1 case
```matlab
if r == 1
    G_mat = G(1);  % Ensure scalar
else
    G_mat = reshape(permute(G, [1,2,3,4]), [r*r, r*r]);
end
```

### 2. TuckerOperator.m - get_proj_grad_kronecker()

**Issue**: Line 221 - Reshaping scalar to 4D tensor
```matlab
dG = dG + residual(i) * reshape(B1(:) * B2(:)', [r, r, r, r]);
```

**Fix**: Special handling for r=1 to maintain 4D structure
```matlab
if r == 1
    dG_increment = B1(1) * B2(1);  % Scalar
    % Ensure dG stays as proper 4D array
    dG(1,1,1,1) = dG(1,1,1,1) + residual(i) * dG_increment;
else
    dG = dG + residual(i) * reshape(B1(:) * B2(:)', [r, r, r, r]);
end
```

### 3. TuckerTensor.m - tensor_mode_product()

**Issue**: Dimension collapse causes mode operations to fail
- When r=1, `size(T)` returns [1,1] instead of [1,1,1,1]
- Check `length(sz) < mode` fails for modes 3,4

**Fix**: Already implemented in previous fix
```matlab
% Handle scalar/low-dimensional tensors
if isscalar(T)
    sz = ones(1, mode);
end

% Ensure sz has at least 'mode' dimensions
if length(sz) < mode
    sz = [sz, ones(1, mode - length(sz))];
end
```

### 4. TuckerTensor.m - mode_k_unfold()

**Issue**: Permutation and reshape operations may fail with collapsed dimensions

**Fix**: Added robust handling
```matlab
% Handle scalar case
if isscalar(T)
    T_mat = reshape(T, dims(k), []);
    return;
end

% Ensure T has correct dimensions (MATLAB drops trailing 1s)
sz = size(T);
if length(sz) < n_modes
    % Pad with ones to match expected dimensions
    T = reshape(T, [sz, ones(1, n_modes - length(sz))]);
end
```

### 5. TuckerTensor.m - initialize_core()

**Issue**: `zeros([1,1,1,1])` might collapse to scalar

**Fix**: Added explicit dimension enforcement
```matlab
% Ensure proper dimensions (MATLAB may collapse for r=1)
if ndims(G_init) < obj.order
    G_init = reshape(G_init, obj.tucker_ranks);
end
```

## Test Coverage

Created `test_r1_case.m` with comprehensive tests:

1. **Tucker Tensor Creation**: Verify r=1 tensor initializes correctly
2. **Full Tensor Reconstruction**: Check dimension preservation in `full()`
3. **Forward Operators**: Compare Kronecker vs General with r=1
4. **Gradient Computation**: Test Riemannian gradient with r=1
5. **Retraction**: Verify retraction maintains rank and orthogonality

## Key Principles Applied

1. **Explicit Scalar Detection**: Use `isscalar()` to catch collapsed tensors
2. **Dimension Padding**: Pad `size()` results with ones when needed
3. **Direct Indexing**: Use `T(1,1,1,1)` instead of `T(:)` for r=1
4. **Safe Permutation**: Calculate order as `max(ndims(T), mode)`
5. **Reshape Guards**: Always verify dimensions before reshape

## Testing Recommendations

Run both test files to verify correctness:

```matlab
% Test general Kronecker operator
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
test_forward_kronecker

% Test r=1 edge case specifically  
test_r1_case
```

Expected output:
- All relative errors < 1e-10
- All dimension checks pass
- No "Invalid mode for tensor" errors
- Proper 4D structure maintained throughout

## Mathematical Correctness

Even with r=1, all operations remain mathematically correct:

- **Tucker decomposition**: T = G ×₁ U ×₂ U ×₃ U ×₄ U where G is 1×1×1×1
- **Forward operator**: y_i = <A_i ⊗ A_i, T> = scalar · vec(U'AU) · vec(U'AU)
- **Gradient**: Still projects onto tangent space, just lower dimensional
- **Retraction**: Extended core is 2×2×2×2, truncated back to 1×1×1×1

All algorithms work correctly; we just ensure MATLAB doesn't lose track of dimensions.
