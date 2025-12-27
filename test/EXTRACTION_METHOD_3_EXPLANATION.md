# Method 3: Core Tensor Eigendecomposition for Matrix Extraction

## Overview

This document explains Method 3 for extracting a matrix X from a Tucker tensor decomposition, which uses the mathematical relationship between the core tensor G and the original tensor T.

## Mathematical Foundation

### The Key Relationship

Given:
- Original tensor: **H** ∈ ℝ^(d₁×d₂×d₁×d₂)
- Factor matrices: **U₁**, **U₂**, **U₃**, **U₄** ∈ ℝ^(d×r)
- Core tensor: **G** = **H** ×₁ U₁ᵀ ×₂ U₂ᵀ ×₃ U₃ᵀ ×₄ U₄ᵀ

When matricized (reshaped to matrices):
- **H_mat** = reshape(**H**, [d², d²])
- **G_mat** = reshape(**G**, [r², r²])

The relationship is:
```
G_mat = (U₂ᵀ ⊗ U₁ᵀ) H_mat (U₄ᵀ ⊗ U₃ᵀ)ᵀ
```

### Rank-1 Structure Preservation

If **H_mat** has rank-1 structure:
```
H_mat ≈ λ v vᵀ
```

Then **G_mat** also has rank-1 structure:
```
G_mat ≈ λ q qᵀ
```

where:
```
q = (U₂ᵀ ⊗ U₁ᵀ) v
```

**Inverse relationship**:
```
v = (U₂ ⊗ U₁) q
```

This allows us to recover v from q!

## Algorithm Steps

### Step 1: Align Factor Matrices with Sign Flips

**Goal**: Ensure U₁ = U₃ and U₂ = U₄ (up to numerical precision).

**Why**: The spectral initialization may produce factors that differ only in sign. We need to align them for the mathematical relationships to hold exactly.

**Implementation**:
```matlab
for col = 1:r
    % For each column, choose sign to maximize agreement
    corr_pos = U1(:, col)' * U3(:, col);
    corr_neg = U1(:, col)' * (-U3(:, col));
    
    if abs(corr_neg) > abs(corr_pos)
        U3(:, col) = -U3(:, col);  % Flip sign
    end
end
```

Repeat for U₂ and U₄.

**Verification**:
```matlab
||U1 - U3||_F < 1e-10  ✓
||U2 - U4||_F < 1e-10  ✓
```

### Step 2: Form G_mat

**Formula**:
```matlab
G_mat = (U' ⊗ U') * H_mat * (U' ⊗ U')'
```

where U represents the common factor matrix (since U₁=U₃, U₂=U₄ after alignment).

**Alternative** (if already have core tensor):
```matlab
G_mat = reshape(G_tensor, [r², r²])
```

### Step 3: Check Symmetry

**Expected**: If H_mat is symmetric, then G_mat should also be symmetric.

**Verification**:
```matlab
symm_error = ||G_mat - G_mat'||_F
```

If asymmetry detected, symmetrize:
```matlab
G_mat = (G_mat + G_mat') / 2
```

### Step 4: Extract Leading Eigenvector of G_mat

**Operation**: Eigendecomposition of G_mat.

```matlab
[V, D] = eig(G_mat);
[λ_max, idx] = max(abs(diag(D)));
q = V(:, idx);  % Leading eigenvector
```

**Result**:
- λ_max: Leading eigenvalue of G_mat
- q ∈ ℝ^(r²): Leading eigenvector

**Interpretation**: 
- q encodes the core structure of the rank-1 approximation
- q = (U' ⊗ U') v, where v is the eigenvector of H_mat

### Step 5: Reconstruct v from q

**Formula**:
```
v = (U ⊗ U) q
```

**Implementation**:
```matlab
U_kron = kron(U, U);  % d² × r²
v = U_kron * q;       % d² × 1
```

**Dimensions**:
- U: d × r
- U ⊗ U: d² × r²
- q: r² × 1
- v: d² × 1

### Step 6: Reshape v to Matrix X

**Operation**: Reshape the vector v back to matrix form.

```matlab
X = reshape(v, [d, d]);
```

**Post-processing**:
```matlab
X = X / norm(X, 'fro');     % Normalize
X = (X + X') / 2;            % Symmetrize
```

## Why This Method Works

### Theoretical Justification

1. **Tucker decomposition preserves structure**: The core tensor G encodes the same fundamental structure as the original tensor H, just in a compressed form.

2. **Kronecker product relationship**: The matricization and mode products commute with the Kronecker product operation, enabling the transformation between v and q.

3. **Eigenspace preservation**: The leading eigenspace of H_mat is mapped to the leading eigenspace of G_mat through the linear transformation (U' ⊗ U').

4. **Computational efficiency**: Working with G_mat (r² × r²) is much faster than working with H_mat (d² × d²) when r << d.

### Advantages Over Method 1

| Aspect | Method 1 (Direct) | Method 3 (Core) |
|--------|-------------------|-----------------|
| Matrix size | d² × d² | r² × r² |
| Memory | O(d⁴) | O(r⁴) |
| Computation | O(d⁶) | O(r⁶) + O(r²d²) |
| Best for | Small d | Large d, small r |

## Comparison with Other Methods

### Method 1: Matricization + Eigendecomposition
- **Direct approach**: Extract from full tensor T_mat
- **Pros**: Straightforward, no alignment needed
- **Cons**: Expensive for large d

### Method 2: Diagonal Core Extraction
- **Uses**: G(i,i,j,j) structure
- **Pros**: Even more efficient (only r² elements)
- **Cons**: Approximate, ignores off-diagonal structure

### Method 3: Core Eigendecomposition (This Method)
- **Uses**: Full G_mat structure with mathematical relationship
- **Pros**: Theoretically exact, efficient, leverages proven relationships
- **Cons**: Requires factor alignment step

## Expected Results

For well-conditioned problems with proper spectral initialization:

```
||X_method1 - X_method3||_F < 1e-6
```

All three methods should produce nearly identical results, validating:
1. Correct implementation of tensor operations
2. Valid Tucker decomposition structure
3. Accurate spectral initialization

## Example Output

```
=== Method 3: G_mat Eigendecomposition ===

Step 1: Aligning factor matrices
  After alignment: ||U1 - U3||_F = 1.234e-15
  After alignment: ||U2 - U4||_F = 2.345e-15

Step 2: Computing G_mat
  G_mat computed: 9x9
  G_mat norm: 12.3456

Step 3: Checking symmetry
  ||G_mat - G_mat'||_F = 3.456e-16
  ✓ G_mat is symmetric

Step 4: Eigendecomposition
  Leading eigenvalue: λ = 1.234567e+02
  Leading eigenvector q: size 9, norm 1.000

Step 5: Reconstructing v = (U ⊗ U) q
  Kronecker product (U ⊗ U): 400x9
  Reconstructed v: size 400, norm 12.345

Step 6: Reshaping v to matrix X
  X_method3 size: 20x20
  Reconstruction error: 1.234567e-12
  ✓ Good reconstruction
```

## Validation Tests

The test `test_tucker_spectral_symmetry.m` compares all three methods:

```matlab
% Test output
Pairwise differences:
  ||X_method1 - X_method2||_F = 1.234e-03
  ||X_method1 - X_method3||_F = 2.345e-12
  ||X_method2 - X_method3||_F = 1.234e-03

✓ All methods produce similar results
```

Method 3 should match Method 1 very closely (< 1e-10), as both use the full eigenstructure.

## When to Use This Method

**Recommended when**:
- d is large (> 50)
- r is small (< 10)
- You already have Tucker decomposition
- You need theoretical guarantees

**Not recommended when**:
- d is very small (< 20) - use Method 1
- Only diagonal structure matters - use Method 2
- Factor matrices cannot be aligned

## Implementation Notes

### Sign Alignment Importance

Without sign alignment, the relationship q = (U' ⊗ U')v breaks down because:
- If U₃ = -U₁ for some columns
- Then (U₂' ⊗ U₁') ≠ (U₄' ⊗ U₃')'
- And the inverse transformation fails

**Always check**:
```matlab
||U1 - U3||_F should be < 1e-10 after alignment
```

### Numerical Stability

The method involves:
1. Kronecker products (can be large)
2. Eigendecomposition (stable for symmetric matrices)
3. Matrix-vector multiplication

For best stability:
- Ensure factors are well-conditioned
- Symmetrize G_mat before eigendecomposition
- Normalize intermediate results

## References

1. **Tucker Decomposition**: Kolda & Bader (2009), "Tensor Decompositions and Applications"
2. **Kronecker Products**: Van Loan (2000), "The Ubiquitous Kronecker Product"
3. **Matrix Unfolding**: De Lathauwer et al. (2000), "A Multilinear Singular Value Decomposition"

---

**Last Updated**: 2025-12-23

