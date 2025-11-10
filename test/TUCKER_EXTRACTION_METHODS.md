# Tucker Tensor Matrix Extraction Methods

This document describes the two methods implemented in `test_tucker_spectral_symmetry.m` for extracting a matrix X from a Tucker tensor representation.

## Problem Setup

Given a 4th-order Tucker tensor:
```
T = G ×₁ U ×₂ U ×₃ U ×₄ U
```
where all factor matrices are identical (symmetric case), we want to extract matrix X such that `T ≈ X ⊗ X`.

## Method 1: Matricization + Eigendecomposition

**Approach**: Convert the tensor to matrix form and extract the leading eigenvector.

### Steps:
1. **Form matricized tensor**: `T_mat = (U₁ ⊗ U₂) * G_mat * (U₃ ⊗ U₄)'`
   - For scalar core G: `T_mat = G * (U_left * U_right')`
   - For general core: Use matricized core `G_mat` (r² × r²)

2. **Symmetrize**: `T_mat = (T_mat + T_mat') / 2`

3. **Eigendecomposition**: Extract leading eigenvector `v`
   ```matlab
   [V, D] = eig(T_mat);
   [~, idx] = max(abs(diag(D)));
   v_lead = V(:, idx);
   ```

4. **Reshape to matrix**: `X = reshape(v_lead, [d, d])`

5. **Normalize and symmetrize**: 
   ```matlab
   X = X / norm(X, 'fro');
   X = (X + X') / 2;
   ```

### Properties:
- ✓ Most stable for symmetric tensors
- ✓ Well-established eigendecomposition theory
- ✓ Direct extraction from full tensor representation
- ✗ Requires forming large Kronecker products (d² × d²)

---

## Method 2: Tucker Decomposition Structure (U * C_root * U^T)

**Approach**: Exploit the Tucker decomposition structure assuming `X = U * C_root * U^T`.

### Theory:
If `X = U * C_root * U^T`, then:
```
T = X ⊗ X = (U * C_root * U^T) ⊗ (U * C_root * U^T)
  = (C_root ⊗ C_root) ×₁ U ×₂ U ×₃ U ×₄ U
```

Therefore, the core tensor `G ≈ C_root ⊗ C_root`.

### Steps:

#### For scalar core (r=1):
1. `C_root = sqrt(|G|)`
2. `X = U * C_root * U^T`

#### For general core (r>1):
1. **Matricize core**: `G_mat = reshape(G, [r², r²])`

2. **Extract leading structure**:
   ```matlab
   [V_g, D_g] = eig(G_mat);
   v_g = leading eigenvector
   lambda_g = leading eigenvalue
   ```

3. **Reshape to intermediate form**:
   ```matlab
   C_temp = reshape(v_g * sqrt(|lambda_g|), [r, r])
   C_temp = (C_temp + C_temp') / 2
   ```

4. **Extract C_root via eigendecomposition**:
   ```matlab
   [V_c, D_c] = eig(C_temp);
   D_c_root = diag(nthroot(|diag(D_c)|, 4))
   C_root = V_c * D_c_root * V_c'
   ```

5. **Form matrix**: `X = U * C_root * U^T`

6. **Normalize and symmetrize**:
   ```matlab
   X = X / norm(X, 'fro');
   X = (X + X') / 2;
   ```

### Properties:
- ✓ Exploits Tucker structure explicitly
- ✓ Works directly with core tensor (r × r)
- ✓ More memory efficient (no d² × d² matrices)
- ✓ Natural interpretation: X has eigen-structure from U
- ✗ More complex extraction procedure
- ✗ Requires multiple square root operations

---

## Comparison

| Aspect | Method 1 (Matricization) | Method 2 (Tucker Structure) |
|--------|-------------------------|----------------------------|
| **Complexity** | O(d⁴) | O(r⁴) |
| **Memory** | O(d⁴) | O(r²) |
| **Stability** | High | Medium |
| **Interpretation** | Leading eigenvector of T_mat | Eigen-structure from U and G |
| **Best for** | Small d, stable extraction | Large d, low rank r |

## Expected Results

For symmetric Tucker tensors with `U{1} = U{2} = U{3} = U{4}`:
- Both methods should produce **similar results** (difference < 1e-3)
- Both should have **good reconstruction error** (< 0.1 for well-conditioned problems)
- Method consistency indicates correct Tucker structure

## Test Command

```bash
cd test
matlab -batch "test_tucker_spectral_symmetry"
```

## Example Output

```
=== Checking Factor Matrix Symmetry ===
  ||U{1} - U{2}||_F = 0.000000e+00
  ||U{1} - U{3}||_F = 0.000000e+00
  ||U{1} - U{4}||_F = 0.000000e+00
  ✓ PASS: All factor matrices are identical

--- Method 1: Matricization + Eigendecomposition ---
  Reconstruction error: 1.234567e-02
  ✓ Good reconstruction

--- Method 2: Tucker Decomposition (U * C_root * U^T) ---
  C_root extracted: 3x3 matrix, norm=1.234
  Reconstruction error: 1.234789e-02
  ✓ Good reconstruction

--- Comparison of Extraction Methods ---
  Difference between methods: ||X1 - X2||_F = 1.234e-05
  ✓ Both methods produce identical results
```

## References

1. T. G. Kolda and B. W. Bader, "Tensor Decompositions and Applications," SIAM Review, 2009.
2. L. De Lathauwer et al., "A Multilinear Singular Value Decomposition," SIAM J. Matrix Anal. Appl., 2000.

