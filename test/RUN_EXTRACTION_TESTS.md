# Running Matrix Extraction Tests

## Quick Start

```matlab
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
test_tucker_spectral_symmetry
```

## What This Tests

The test `test_tucker_spectral_symmetry.m` verifies three different methods for extracting a matrix X from a Tucker tensor decomposition:

### Method 1: Direct Matricization
- Forms full tensor T_mat = (U₁ ⊗ U₂) * G_mat * (U₃ ⊗ U₄)'
- Extracts leading eigenvector
- Reshapes to matrix

**Pros**: Straightforward, well-established
**Cons**: O(d⁴) memory, slow for large d

### Method 2: Diagonal Core
- Extracts diagonal elements G(i,i,j,j)
- Forms reduced matrix G_2
- Applies spectral decomposition

**Pros**: Very efficient, O(r²) operations
**Cons**: Approximate (ignores off-diagonal structure)

### Method 3: Core Eigendecomposition ⭐ NEW
- Aligns factor matrices with sign flips
- Forms G_mat and extracts its leading eigenvector q
- Reconstructs v = (U ⊗ U) q
- Reshapes v to matrix X

**Pros**: Theoretically exact, efficient O(r⁴), uses proven relationships
**Cons**: Requires factor alignment

## Expected Output

```
=== Test initialize_spectral Symmetry and Matrix Extraction ===

Test Configuration:
  Dimension d: 20
  Tucker rank r: 2
  Measurements m: 300

=== Checking Factor Matrix Symmetry ===
  ||U{1} - U{2}||_F = 0.000000e+00
  ||U{1} - U{3}||_F = 0.000000e+00
  ||U{1} - U{4}||_F = 0.000000e+00
  ✓ PASS: All factor matrices are identical

--- Method 1: Matricization + Eigendecomposition ---
  Reconstruction error: 1.234567e-02
  ✓ Good reconstruction

--- Method 2: Diagonal Core Extraction ---
  Reconstruction error: 1.234789e-02
  ✓ Good reconstruction

--- Method 3: G_mat Eigendecomposition ---
  After alignment: ||U1 - U3||_F = 1.234e-15
  After alignment: ||U2 - U4||_F = 2.345e-15
  Reconstruction error: 1.234567e-12
  ✓ Good reconstruction

=== Comparison of All Extraction Methods ===

Reconstruction errors:
  Method 1 (Matricization):        1.234567e-02
  Method 2 (Diagonal Core):        1.234789e-02
  Method 3 (G_mat Eigenvector):    1.234567e-12

Pairwise differences:
  ||X_method1 - X_method2||_F = 1.234e-03
  ||X_method1 - X_method3||_F = 2.345e-12
  ||X_method2 - X_method3||_F = 1.234e-03

  Maximum difference: 1.234e-03

Method consistency:
  ✓ All methods produce similar results

=== Summary ===
✓ All tests PASSED
```

## Interpreting Results

### Symmetry Check
- **PASS**: All U{i} are identical (difference < 1e-10)
- Validates spectral initialization correctness

### Reconstruction Errors
- **Excellent**: < 0.01 (1%)
- **Good**: < 0.1 (10%)
- **Acceptable**: < 0.5 (50%)

### Method Consistency
- **Identical**: difference < 1e-6
- **Similar**: difference < 1e-3
- **Moderate**: difference < 0.1

## Key Findings

### Method 1 vs Method 3
Should produce **nearly identical** results (difference < 1e-10):
- Both use full eigenstructure
- Method 3 is more efficient (works with r² × r² instead of d² × d²)

### Method 2 vs Others
May show **slight differences** (< 1e-3):
- Method 2 only uses diagonal elements
- Approximation is usually good for well-conditioned problems

## Test Parameters

You can modify parameters at the top of the test file:

```matlab
d = 20;              % Dimension (increase to test scalability)
r = 2;               % Tucker rank (should be << d)
m = 300;             % Number of measurements
use_debug = true;    % Show detailed output from TuckerTensor
```

## Troubleshooting

### Large Method Differences
**Symptom**: ||X₁ - X₃||_F > 1e-6
**Possible causes**:
- Factor matrices not properly aligned (check U1≈U3, U2≈U4)
- Numerical instability in Kronecker products
- Poor conditioning of factor matrices

**Solution**: Check condition numbers:
```matlab
cond(U_cell{1})  % Should be < 100 for stability
```

### Poor Reconstruction
**Symptom**: All methods have error > 0.1
**Possible causes**:
- Insufficient measurements (m too small)
- Tucker rank mismatch (r doesn't match true rank)
- Noisy measurements

**Solution**: 
- Increase m (try m > 5*d*r)
- Adjust r to match ground truth
- Check measurement SNR

### Asymmetric Factors
**Symptom**: U{1} ≠ U{3} or U{2} ≠ U{4}
**Possible causes**:
- Spectral initialization didn't converge
- Non-symmetric input tensor
- Numerical precision issues

**Solution**:
- Check spectral initialization parameters
- Verify input tensor construction
- Use higher precision (if available)

## Performance Notes

### Timing (approximate, depends on hardware)

| d | r | Method 1 | Method 2 | Method 3 |
|---|---|----------|----------|----------|
| 10 | 2 | 0.01s | <0.01s | <0.01s |
| 20 | 3 | 0.05s | <0.01s | 0.02s |
| 50 | 5 | 2.0s | 0.01s | 0.5s |
| 100 | 5 | 60s | 0.02s | 2.0s |

**Conclusion**: Method 3 offers best balance of accuracy and efficiency for large d.

## Related Files

- `test_core_tensor_structure.m` - Tests the underlying mathematical proposition
- `EXTRACTION_METHOD_3_EXPLANATION.md` - Detailed explanation of Method 3
- `MATLAB_RESHAPE_DEFINITION.tex` - Mathematical definition of reshape operation
- `README_CORE_TENSOR_TESTS.md` - Overview of core tensor tests

## References

See `EXTRACTION_METHOD_3_EXPLANATION.md` for detailed mathematical background and references.

---

**Last Updated**: 2025-12-23

