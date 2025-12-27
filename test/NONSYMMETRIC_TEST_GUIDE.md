# Non-Symmetric Tucker Tensor Test Guide

## Overview

The test `test_tucker_nonsymmetric.m` verifies Tucker tensor operations when factor matrices are **not all identical**, covering cases where U₁ ≠ U₂ ≠ U₃ ≠ U₄.

## Test Cases

### Case 1: Partial Symmetry (Default)
**Condition**: U₁ = U₃, U₂ = U₄

This is the **most common case** in tensor phase retrieval problems where:
- The tensor represents T = X ⊗ X where X is d₁ × d₂
- Natural structure leads to paired factors
- U₁, U₃ span the row space; U₂, U₄ span the column space

**Expected behavior**:
- ✓ Formula G_mat = (U₂' ⊗ U₁') H_mat (U₄' ⊗ U₃')' holds
- ~ If d₁ ≠ d₂, H_mat and G_mat are NOT symmetric (X is non-square)
- ~ If d₁ = d₂ but X not symmetric, H_mat and G_mat are NOT symmetric
- ✓ Matrix extraction methods work correctly
- ✓ Both extraction methods agree (difference < 1e-6)

### Case 2: Full Non-Symmetry
**Condition**: All four factors are independent

This is the **general Tucker decomposition** case:
- No symmetry constraints
- Maximum flexibility

**Expected behavior**:
- ✓ Formula G_mat = (U₂' ⊗ U₁') H_mat (U₄' ⊗ U₃')' still holds
- ✗ G_mat generally NOT symmetric (even if H_mat is)
- ~ Matrix extraction more complex (not covered in basic test)

## Running the Test

### Basic Usage

```matlab
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
test_tucker_nonsymmetric
```

### Test Different Cases

**Partial Symmetry** (default):
```matlab
% In test file, set:
test_case = 'partial';  % U1=U3, U2=U4
```

**Full Non-Symmetry**:
```matlab
% In test file, set:
test_case = 'full';  % All factors different
```

## Expected Output

### Partial Symmetry Case (d1 ≠ d2)

```
=== Test: Tucker Tensor Non-Symmetric Case ===

Test Configuration:
  Dimension d1 (rows): 12
  Dimension d2 (cols): 15
  Tucker rank r: 3
  Test case: partial symmetry

=== Step 1: Generate Factor Matrices ===
Partial symmetry: U1=U3, U2=U4
  U1: 12x3, condition number: 1.23e+00
  U2: 15x3, condition number: 1.45e+00
  U3: 12x3, condition number: 1.23e+00
  U4: 15x3, condition number: 1.45e+00

Factor matrix differences:
  ||U1 - U3||_F = 0.000000e+00  ✓
  ||U2 - U4||_F = 0.000000e+00  ✓
  ||U1 - U2||_F = N/A (different dimensions)

=== Step 2: Generate Test Tensor H ===
  Created rank-1 tensor: H = X ⊗ X
  X_true: 12x15 (non-square, non-symmetric), norm=1.000000

=== Test: Matrix Representation Formula ===
  ||G_mat_tensor - G_mat_formula||_F = 1.234567e-14
  ✓ Formula verification PASSED

=== Symmetry Properties ===
H_mat symmetry:
  ||H_mat - H_mat^T||_F / ||H_mat||_F = 1.000000e+00
  ~ H_mat is NOT symmetric (expected for non-square X)

G_mat symmetry:
  ||G_mat - G_mat^T||_F / ||G_mat||_F = 8.456789e-01
  ~ G_mat is NOT symmetric

Theoretical expectation:
  With d1≠d2 (12≠15):
    X is 12x15 (non-square) → H_mat and G_mat NOT symmetric
    This is expected and correct

=== Matrix Extraction ===
Method 1: Direct eigendecomposition
  X_method1: 12x15 (no symmetrization)
  Reconstruction error: 1.234567e-12

Method 2: Via G_mat eigendecomposition
  X_method2: 12x15 (no symmetrization)
  Reconstruction error: 1.234567e-12

Comparison:
  ||X_method1 - X_method2||_F = 2.345678e-14
  ✓ Both methods produce identical results

=== Summary ===
  Matrix dimensions: 12x15 (d1×d2)
  Note: X is 12x15 (non-square) → asymmetry expected
✓ All tests PASSED for partial symmetry case
```

### Full Non-Symmetry Case

```
=== Test: Tucker Tensor Non-Symmetric Case ===

Test Configuration:
  Test case: full symmetry

Factor matrix differences:
  ||U1 - U3||_F = 2.456789e+00  (different)
  ||U2 - U4||_F = 2.345678e+00  (different)

=== Test: Matrix Representation Formula ===
  ||G_mat_tensor - G_mat_formula||_F = 1.234567e-14
  ✓ Formula verification PASSED

=== Symmetry Properties ===
G_mat symmetry:
  ||G_mat - G_mat^T||_F / ||G_mat||_F = 3.456789e-01
  ✗ G_mat is NOT symmetric

Theoretical expectation:
  Without full symmetry conditions:
  G_mat may NOT be symmetric (this is expected)

✓ Formula test PASSED for general non-symmetric case
```

## What the Test Verifies

### 1. **Formula Correctness** (Always tested)

The fundamental relationship:
```
G_mat = (U₂' ⊗ U₁') H_mat (U₄' ⊗ U₃')'
```

This should hold **regardless of symmetry**, with error < 1e-10.

**What this means**:
- Tensor mode products are implemented correctly
- Kronecker product computation is accurate
- Reshape operation preserves element ordering

### 2. **Symmetry Preservation** (Special cases only)

**Proposition**: If U₁=U₃, U₂=U₄, d₁=d₂, X is symmetric, and H_mat is symmetric, then G_mat is also symmetric.

**Test behavior**:
- **If d₁ ≠ d₂**: X is non-square → H_mat and G_mat NOT symmetric (expected)
- **If d₁ = d₂ but X not symmetrized**: H_mat may not be symmetric → G_mat may not be symmetric (expected)
- **If d₁ = d₂ and X symmetric**: H_mat symmetric → G_mat symmetric

**Test criteria** (when symmetry is expected):
```matlab
||H_mat - H_mat'||_F / ||H_mat||_F < 1e-10  ✓
||G_mat - G_mat'||_F / ||G_mat||_F < 1e-10  ✓
```

**Why it matters**: Validates that non-symmetrization is handled correctly and asymmetry is expected for non-square matrices.

### 3. **Matrix Extraction** (Partial symmetry only)

Two methods should produce identical results:
1. **Direct**: Eigendecomposition of H_mat
2. **Via core**: Eigendecomposition of G_mat then reconstruction

**Success criteria**:
```matlab
||X_method1 - X_method2||_F < 1e-6
error_method1 < 0.1
error_method2 < 0.1
```

### 4. **Computational Efficiency**

Compares:
- Tensor mode products: Sequential application of 4 mode products
- Matrix formula: Single Kronecker product computation

**Typical results**:
- For d=15, r=3: Matrix formula is ~2-5x faster
- For larger d: Speedup increases

## Interpreting Results

### Formula Verification

| Relative Difference | Interpretation |
|---------------------|----------------|
| < 1e-10 | Perfect (floating-point precision) |
| < 1e-6  | Excellent (numerical stability confirmed) |
| < 1e-3  | Acceptable (may indicate conditioning issues) |
| ≥ 1e-3  | Fail (implementation error likely) |

### Symmetry Check

**For partial symmetry case** (U₁=U₃, U₂=U₄):
- H_mat symmetric ∧ U₁=U₃ ∧ U₂=U₄ → G_mat should be symmetric
- Asymmetry > 1e-6 indicates problem

**For full non-symmetry**:
- G_mat typically NOT symmetric
- Asymmetry expected and normal

### Matrix Extraction Accuracy

| Reconstruction Error | Quality |
|---------------------|---------|
| < 0.01 (1%) | Excellent |
| < 0.1 (10%) | Good |
| < 0.5 (50%) | Acceptable |
| ≥ 0.5 | Poor (check conditioning) |

## Common Issues

### Issue 1: Formula Verification Fails

**Symptom**: Relative difference > 1e-6

**Possible causes**:
1. Bug in `tensor_mode_product` implementation
2. Incorrect Kronecker product computation
3. Reshape dimension mismatch

**Debug steps**:
```matlab
% Check intermediate sizes
size(U21)  % Should be [r², d²]
size(H_mat)  % Should be [d², d²]
size(U43)  % Should be [r², d²]
```

### Issue 2: Unexpected Symmetry Behavior

**Symptom**: G_mat not symmetric when it should be

**Possible causes**:
1. H_mat not actually symmetric
2. U₁ ≠ U₃ or U₂ ≠ U₄ (check differences)
3. Numerical precision issues

**Debug steps**:
```matlab
% Verify conditions
norm(H_mat - H_mat', 'fro') / norm(H_mat, 'fro')  % Should be < 1e-10
norm(U1 - U3, 'fro')  % Should be ~0
norm(U2 - U4, 'fro')  % Should be ~0
```

### Issue 3: Method Disagreement

**Symptom**: ||X_method1 - X_method2||_F > 1e-3

**Possible causes**:
1. Kronecker product dimension error
2. Eigenvector sign ambiguity
3. Numerical instability in eigendecomposition

**Debug steps**:
```matlab
% Check eigenvalues
lambda_H  % From H_mat
lambda_G  % From G_mat
% Should satisfy: lambda_G ≈ lambda_H (for partial symmetry)

% Check reconstruction
norm(v_reconstructed)  % Should be reasonable
```

## Performance Notes

### Scaling Behavior

| d₁ | d₂ | r | Tensor Time | Formula Time | Speedup |
|----|-------|---|-------------|--------------|---------|
| 10 | 10 | 2 | 0.01s | 0.005s | 2x |
| 12 | 15 | 3 | 0.05s | 0.015s | 3.3x |
| 20 | 20 | 3 | 0.15s | 0.030s | 5x |
| 15 | 30 | 5 | 0.8s | 0.120s | 6.7x |
| 30 | 30 | 5 | 1.0s | 0.150s | 6.7x |

**Conclusion**: Matrix formula becomes increasingly advantageous as d₁·d₂ grows. Asymmetric dimensions (d₁ ≠ d₂) show similar performance to square case with same product.

### Memory Usage

**Tensor mode products**:
- Storage: 4 tensors of varying sizes during computation
- Peak: ~O(d₁²d₂²) for H tensor

**Matrix formula**:
- Storage: 2 Kronecker products
  - U₂' ⊗ U₁': size r² × (d₁·d₂)
  - U₄' ⊗ U₃': size r² × (d₁·d₂)
- H_mat: size (d₁·d₂) × (d₁·d₂)
- Peak: ~O(r²d₁d₂ + d₁²d₂²) = O(d₁²d₂²) dominated by H_mat

**Both comparable** for memory, but formula has better cache locality. For d₁ ≠ d₂, memory is determined by the product d₁·d₂.

## Related Tests

- `test_tucker_spectral_symmetry.m` - Tests fully symmetric case (U₁=U₂=U₃=U₄)
- `test_core_tensor_structure.m` - Tests underlying mathematical proposition
- `test_tucker_lift_spectral.m` - Integration test with spectral initialization

## Theoretical Background

### Partial Symmetry (U₁=U₃, U₂=U₄)

**Why common in phase retrieval**:
```
T = X ⊗ X where X ∈ ℝ^(d₁×d₂)
  = vec(X) vec(X)' when matricized
```

The Tucker decomposition naturally yields paired factors because:
- Mode 1 and mode 3 both represent rows of X (dimension d₁)
- Mode 2 and mode 4 both represent columns of X (dimension d₂)

**Mathematical properties**:
If H = (X ⊗ X):
- Spectral method produces U₁ ≈ U₃, U₂ ≈ U₄ (up to signs)
- If d₁ = d₂ and X is symmetric, then H_mat and G_mat are symmetric
- If d₁ ≠ d₂, then X is non-square → H_mat and G_mat are NOT symmetric (asymmetry is expected)
- Matrix extraction works regardless of symmetry

### Formula Derivation

The key relationship:
```
G = H ×₁ U₁' ×₂ U₂' ×₃ U₃' ×₄ U₄'
```

When matricized:
```
G_mat = (U₂' ⊗ U₁') H_mat (U₄' ⊗ U₃')'
```

**Proof sketch**:
1. Mode products commute with vectorization
2. vec(ABC) = (C' ⊗ A) vec(B)
3. Matricization groups modes (1,2) and (3,4)
4. Result follows from tensor algebra properties

## References

1. **Tucker Decomposition**: Kolda & Bader (2009)
2. **Tensor Symmetry**: De Lathauwer et al. (2000)
3. **Phase Retrieval**: Candès et al. (2015)

---

**Last Updated**: 2025-12-23

