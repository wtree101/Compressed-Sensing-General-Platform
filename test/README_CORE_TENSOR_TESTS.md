# Core Tensor Structure Tests

This directory contains tests for verifying the mathematical properties of Tucker tensor decomposition and core tensor structure from spectral initialization.

## Files

### 1. `test_core_tensor_structure.m`
**Purpose**: Verify the mathematical proposition about core tensor structure from spectral initialization.

**What it tests**:
- **(i) Matrix Representation Formula**: 
  ```
  G_mat = (U2' ⊗ U1') * H_mat * (U4' ⊗ U3')'
  ```
  where G is the core tensor computed via tensor mode products.

- **(ii) Symmetry Preservation**: 
  If U1=U3, U2=U4, and H_mat is symmetric, then G_mat is also symmetric.

- **(iii) Rank-1 Structure Preservation**: 
  If H_mat ≈ λv·v', then G_mat ≈ λq·q' where q = (U2' ⊗ U1')v.

**Usage**:
```matlab
cd test
test_core_tensor_structure
```

**Expected Output**:
```
=== Test (i): Matrix Representation Formula ===
  ||G_mat - G_mat_formula||_F = X.XXXXXe-XX
  ✓ TEST (i) PASSED: Formula is correct

=== Test (ii): Symmetry Preservation ===
  ||G_mat - G_mat^T||_F = X.XXXXXe-XX
  ✓ TEST (ii) PASSED: G_mat is symmetric

=== Test (iii): Rank-1 Structure Preservation ===
  ||G_mat - λqq^T||_F = X.XXXXXe-XX
  ✓ TEST (iii) PASSED: Rank-1 structure preserved
```

### 2. `MATLAB_RESHAPE_DEFINITION.tex`
**Purpose**: Provide a rigorous mathematical definition of MATLAB's `reshape` operation.

**Contents**:
- Column-major ordering definition
- Mathematical formulation of reshape
- Relationship to Kronecker products
- Matrix unfolding interpretation
- Proof of the Kronecker product relationship

**Compile**:
```bash
pdflatex MATLAB_RESHAPE_DEFINITION.tex
```

## Mathematical Background

### Tensor Mode Product

The mode-$k$ product of a tensor $\mathcal{T} \in \mathbb{R}^{n_1 \times \cdots \times n_N}$ with a matrix $\bm{M} \in \mathbb{R}^{m \times n_k}$ is defined as:

$$(\mathcal{T} \times_k \bm{M})_{i_1,\ldots,i_{k-1},j,i_{k+1},\ldots,i_N} = \sum_{i_k=1}^{n_k} \mathcal{T}_{i_1,\ldots,i_N} M_{j,i_k}$$

### Core Tensor Computation

Given a 4th-order tensor $\mathcal{H}$ and factor matrices $\bm{U}_1, \bm{U}_2, \bm{U}_3, \bm{U}_4$:

$$\mathcal{G} = \mathcal{H} \times_1 \bm{U}_1^\top \times_2 \bm{U}_2^\top \times_3 \bm{U}_3^\top \times_4 \bm{U}_4^\top$$

### Matricization

For a 4th-order tensor $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times d_1 \times d_2}$, the mode-(1,2) matricization is:

$$\bm{T}_{\text{mat}} = \text{reshape}(\mathcal{T}, [d_1 d_2, d_1 d_2])$$

This creates a matrix where:
- Rows correspond to modes 1 and 2 (vectorized)
- Columns correspond to modes 3 and 4 (vectorized)

### Key Proposition

**Theorem**: The matricized core tensor satisfies:

$$\bm{G}_{\text{mat}} = (\bm{U}_2^\top \otimes \bm{U}_1^\top) \bm{H}_{\text{mat}} (\bm{U}_4^\top \otimes \bm{U}_3^\top)^\top$$

This is the fundamental relationship tested in `test_core_tensor_structure.m`.

## Why This Matters

### 1. **Efficiency**
Instead of computing the core tensor via four sequential mode products (expensive for large tensors), we can:
- Matricize H once
- Apply two Kronecker products
- This is often faster for moderately sized problems

### 2. **Theoretical Understanding**
The formula shows that:
- Core tensor computation is essentially a change of basis
- The Kronecker product structure explains why Tucker decomposition works
- Symmetry properties are preserved through the computation

### 3. **Numerical Verification**
These tests ensure that:
- `tensor_mode_product` is implemented correctly
- MATLAB's `reshape` behaves as expected
- The mathematical theory matches the implementation

## Test Parameters

### Default Configuration
```matlab
d1 = 10;             % First dimension
d2 = 10;             % Second dimension
r = 3;               % Tucker rank
test_symmetric = true; % Test symmetric case
```

### Tolerance Levels
- Formula verification: `1e-10` (relative error)
- Symmetry check: `1e-10` (Frobenius norm)
- Rank-1 approximation: `1e-6` (relative error)

## Interpreting Results

### Test (i) - Formula Correctness
- **Pass**: Relative difference < 1e-10
- **Fail**: Indicates implementation error in `tensor_mode_product` or incorrect Kronecker product computation

### Test (ii) - Symmetry
- **Pass**: ||G_mat - G_mat'||_F < 1e-10
- **Fail**: May indicate:
  - Numerical precision issues
  - Factor matrices not exactly equal (U1≠U3 or U2≠U4)
  - Input tensor H not symmetric

### Test (iii) - Rank-1 Structure
- **Pass**: Relative difference < 1e-6
- **Approximate**: 1e-6 < difference < 1e-3 (still reasonable)
- **Fail**: difference > 1e-3 (indicates numerical issues)

## Common Issues

### 1. Dimension Mismatch
**Error**: "Matrix dimensions must agree"
**Solution**: Ensure d1*d2 matches the total size of H

### 2. Numerical Precision
**Warning**: "Relative difference: 1.234e-09"
**Solution**: This is acceptable; floating-point arithmetic has inherent precision limits

### 3. Non-Symmetric Results in Symmetric Case
**Error**: "G_mat is NOT symmetric"
**Possible causes**:
- H_mat not actually symmetric (check construction)
- Factor matrices differ (verify U1==U3, U2==U4)
- Numerical errors accumulated (check condition numbers)

## References

1. **Tucker Decomposition**: 
   - Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. SIAM review.

2. **Mode Products**:
   - De Lathauwer, L., De Moor, B., & Vandewalle, J. (2000). A multilinear singular value decomposition.

3. **Kronecker Products**:
   - Van Loan, C. F. (2000). The ubiquitous Kronecker product. Journal of computational and applied mathematics.

## Contact

For questions or issues with these tests, please refer to the main documentation or open an issue in the repository.

---

**Last Updated**: 2025-12-23

