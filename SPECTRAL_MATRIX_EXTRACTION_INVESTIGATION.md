# Spectral Initialization Investigation: Matrix Extraction from Tensors

## Research Question
**Can the 4th-order tensor T (before and after HOSVD) be extracted to a symmetric matrix?**

For a tensor T ∈ ℝ^(d1×d2×d1×d2), we investigate:
1. Whether matricizing T as (d1·d2 × d1·d2) produces a **symmetric matrix**
2. Whether we can extract the ground truth matrix X from this matricization
3. How HOSVD affects both symmetry and extraction quality

---

## Background

### Tensor Construction (Spectral Initialization)
```
T = Σᵢ yᵢ * (Aᵢ ⊗ Aᵢ)
```

Where:
- Aᵢ ∈ ℝ^(d1×d2) are measurement matrices
- yᵢ are measurements: yᵢ = |⟨Aᵢ, Xstar⟩| / √m
- Xstar ∈ ℝ^(d1×d2) is the ground truth matrix

### Matricization
The 4th-order tensor T can be reshaped to a matrix:
```
T_mat = reshape(T, [d1*d2, d1*d2])
```

### Matrix Extraction
If T_mat is symmetric, extract X via leading eigenvector:
```matlab
[V, D] = eig(T_mat);
[~, idx] = max(abs(diag(D)));
v_lead = V(:, idx);
X_extracted = reshape(v_lead, [d1, d2]);
```

---

## Test Investigation Steps

### For Each Test Case (square/tall/wide matrices):

**1. Form T_before_HOSVD (Spectral Initialization)**
```matlab
tucker_op = TuckerOperator(A_cells);
T_before = tucker_op.kronecker_adjoint(y / sqrt(m));
```

**2. Matricize and Check Symmetry**
```matlab
n = d1 * d2;
T_mat = reshape(permute(T_before, [1,2,3,4]), [n, n]);
symmetry_error = norm(T_mat - T_mat', 'fro') / norm(T_mat, 'fro');
```

**3. Extract Matrix via Leading Eigenvector**
```matlab
[V, D] = eig(T_mat);
v_lead = V(:, max_eigenvalue_index);
X_extracted = reshape(v_lead, [d1, d2]);
error = norm(X_extracted - Xstar, 'fro') / norm(Xstar, 'fro');
```

**4. Apply HOSVD and Repeat**
```matlab
[T_after, U_cells] = HOSVD_with_factors(T_before, rank_vec);
% Repeat matricization, symmetry check, and extraction
```

**5. Compare Results**
- Symmetry before vs after HOSVD
- Extraction error before vs after HOSVD
- Eigenvalue spectrum comparison

---

## Expected Results

### Hypothesis 1: Symmetry Preservation
**Claim:** T_mat should be symmetric because:
```
T[i,j,k,l] = Σᵢ yᵢ * Aᵢ[i,j] * Aᵢ[k,l]
```
When matricized as T_mat[i·d2+j, k·d2+l], the structure should preserve symmetry.

**Test:** Check if `symmetry_error < 1e-10`

### Hypothesis 2: HOSVD Preserves Symmetry
**Claim:** HOSVD projection preserves symmetry because it's a best rank-r approximation.

**Test:** Compare symmetry_error_before vs symmetry_error_after

### Hypothesis 3: Extraction Quality
**Claim:** Leading eigenvector of T_mat should recover Xstar (up to sign).

**Test:** Measure `norm(X_extracted - Xstar) / norm(Xstar)`

### Hypothesis 4: HOSVD Improves Extraction
**Claim:** HOSVD denoising should reduce extraction error.

**Test:** Compare error_before vs error_after

---

## Test Output Interpretation

### Example Output:
```
Test 1 (Square matrix):
  [Matrix Extraction Investigation]
    T_before_HOSVD matricization (400×400):
      Symmetry error: 2.34e-16 ✓ SYMMETRIC
      Leading eigenvector extraction error: 1.23e-01
    T_after_HOSVD matricization (400×400):
      Symmetry error: 3.45e-16 ✓ SYMMETRIC
      Leading eigenvector extraction error: 4.56e-02
    Eigenvalue comparison (top 5):
      Before HOSVD: [8.45e+02, 5.23e+01, 4.12e+01, 3.98e+01, 3.67e+01]
      After HOSVD:  [8.44e+02, 5.21e+01, 4.10e+01, 3.95e+01, 3.65e+01]
      Ratio (after/before): 0.9988
```

### Interpretation:

1. **Symmetry Error ~1e-16**: ✓ Matricized tensor is symmetric (numerical precision)

2. **Extraction Error**: 
   - Before HOSVD: 12.3% error
   - After HOSVD: 4.56% error
   - **3x improvement** from HOSVD denoising

3. **Eigenvalue Spectrum**:
   - Leading eigenvalue slightly reduced (0.999x) - minimal change
   - Spectrum preserved - HOSVD doesn't destroy signal

---

## Success Criteria

### ✓ Success (All Claims Verified)
- Symmetry error < 1e-10 before and after HOSVD
- Extraction error decreases after HOSVD
- Works for square, tall, and wide matrices

### ⚠ Partial Success
- Symmetric but extraction doesn't improve
- Non-symmetric but extraction still works

### ✗ Failure
- Matricized tensor is NOT symmetric
- HOSVD breaks symmetry
- Extraction fails (high error)

---

## Theoretical Justification

### Why Should T_mat Be Symmetric?

For T = Σᵢ yᵢ * (Aᵢ ⊗ Aᵢ):
```
T[i,j,k,l] = Σₙ yₙ * Aₙ[i,j] * Aₙ[k,l]
```

Matricized as T_mat with index mapping: `idx(i,j) = i*d2 + j`
```
T_mat[idx(i,j), idx(k,l)] = T[i,j,k,l]
T_mat[idx(k,l), idx(i,j)] = T[k,l,i,j]
```

**Symmetry condition:**
```
T[i,j,k,l] = T[k,l,i,j]  (Kronecker product property)
```

This holds because:
```
Aᵢ ⊗ Aᵢ has structure: [i,j,k,l] = Aᵢ[i,j] * Aᵢ[k,l]
                                   = Aᵢ[k,l] * Aᵢ[i,j]
                                   = [k,l,i,j]
```

Therefore, T_mat should be symmetric! ✓

---

## Practical Implications

### If T_mat Is Symmetric:
1. **Efficient Extraction**: Use eigenvector method (O(n³) vs iterative methods)
2. **Theoretical Guarantee**: Leading eigenvector gives best rank-1 approximation
3. **Numerical Stability**: Symmetric eigenvalue problems are well-conditioned

### If HOSVD Improves Extraction:
1. **Denoising Effect**: HOSVD acts as rank-r projection filter
2. **Better Initialization**: For iterative methods
3. **Trade-off**: Computational cost vs improved accuracy

### For Non-Square Matrices (d1 ≠ d2):
1. **Size**: T_mat is (d1·d2 × d1·d2) - larger than d1 or d2 alone
2. **Structure**: Extraction reshapes to (d1 × d2) correctly
3. **Symmetry**: Should hold regardless of d1 ≠ d2

---

## Running the Test

```bash
cd test
matlab -batch "test_nonsquare_tucker_spectral"
```

The test will output:
1. Basic functionality checks (dimensions, symmetry of output X0)
2. **Matrix Extraction Investigation** for each test case
3. Summary comparing before/after HOSVD
4. Overall findings on symmetry and extraction quality

---

## Files Modified

- `test/test_nonsquare_tucker_spectral.m`: Added investigation section

---

## Key Findings to Look For

1. ✓ **Symmetry**: All matricized tensors should be symmetric
2. ✓ **Square vs Non-Square**: Both should produce symmetric T_mat
3. ✓ **HOSVD Effect**: Should improve extraction error
4. ✓ **Eigenvalue Ratio**: Leading eigenvalue preserved (ratio ≈ 1)

This investigation validates the theoretical foundation of spectral initialization!
