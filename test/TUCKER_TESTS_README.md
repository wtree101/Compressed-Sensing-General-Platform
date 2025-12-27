# Tucker Tensor Tests - Complete Guide

This directory contains a comprehensive test suite for Tucker tensor decomposition, spectral initialization, and matrix extraction methods.

## Quick Start

### Run All Tests
```matlab
cd /Users/wutong/Documents/MATLAB/GeneralPlatform/test
run_all_tucker_tests
```

### Run Individual Tests
```matlab
test_core_tensor_structure         % Mathematical formula verification
test_tucker_spectral_symmetry      % Symmetric case + 3 extraction methods
test_tucker_nonsymmetric          % Non-symmetric cases
```

## Test Suite Overview

| Test File | Purpose | Key Validation | Time |
|-----------|---------|----------------|------|
| `test_core_tensor_structure.m` | Verify mathematical proposition (i)-(iii) | Formula correctness | ~0.1s |
| `test_tucker_spectral_symmetry.m` | Test symmetric factors + extraction methods | 3 methods agree | ~2-5s |
| `test_tucker_nonsymmetric.m` | Test partial/full non-symmetry | Formula + symmetry preservation | ~0.2s |

## Test Descriptions

### 1. Core Tensor Structure Test

**File**: `test_core_tensor_structure.m`

**What it tests**:
- **(i)** Matrix formula: G_mat = (U₂' ⊗ U₁') H_mat (U₄' ⊗ U₃')'
- **(ii)** Symmetry: U₁=U₃, U₂=U₄, H_mat symmetric → G_mat symmetric
- **(iii)** Rank-1 structure: H_mat ≈ λvv' → G_mat ≈ λqq' where q = (U₂' ⊗ U₁')v

**Expected result**: All three properties verified with relative error < 1e-10

**Documentation**: `README_CORE_TENSOR_TESTS.md`

---

### 2. Symmetric Spectral Initialization Test

**File**: `test_tucker_spectral_symmetry.m`

**What it tests**:
- Factor symmetry: U{1} = U{2} = U{3} = U{4}
- **Method 1**: Direct matricization + eigendecomposition
- **Method 2**: Diagonal core extraction (Tucker structure)
- **Method 3**: Core eigendecomposition with sign alignment

**Expected result**: 
- All factors identical (difference < 1e-10)
- All methods produce similar results (difference < 1e-3)
- Method 1 and Method 3 nearly identical (difference < 1e-10)

**Documentation**: 
- `EXTRACTION_METHOD_3_EXPLANATION.md` - Detailed Method 3 explanation
- `RUN_EXTRACTION_TESTS.md` - Usage guide

---

### 3. Non-Symmetric Test

**File**: `test_tucker_nonsymmetric.m`

**What it tests**:
- **Partial symmetry**: U₁=U₃, U₂=U₄ (common in phase retrieval)
- **Full non-symmetry**: All factors independent
- **Non-square matrices**: X is d₁ × d₂ where d₁ ≠ d₂
- Formula correctness for both cases
- Matrix extraction for partial case (no symmetrization)

**Expected result**:
- Formula correct regardless of symmetry (error < 1e-10)
- For d₁ ≠ d₂: H_mat and G_mat NOT symmetric (expected)
- Full non-symmetry: G_mat generally not symmetric
- Matrix extraction works without symmetrization

**Documentation**: `NONSYMMETRIC_TEST_GUIDE.md`

---

## Test Parameters

### Configurable Parameters

Each test has adjustable parameters at the top:

```matlab
% Common parameters
d = 15-20;           % Dimension (for symmetric tests)
d1 = 12-20;          % Row dimension (for non-symmetric tests)
d2 = 15-25;          % Column dimension (for non-symmetric tests)
r = 2-3;             % Tucker rank
m = 200-400;         % Number of measurements (if applicable)
use_debug = true;    % Detailed output from TuckerTensor class
```

### Recommended Settings

**For quick testing**:
```matlab
d = 10; r = 2; m = 100;        % Symmetric tests
d1 = 10; d2 = 12; r = 2;       % Non-symmetric tests
```

**For thorough validation**:
```matlab
d = 20; r = 3; m = 400;        % Symmetric tests
d1 = 15; d2 = 20; r = 3;       % Non-symmetric tests
```

**For scalability testing**:
```matlab
d = 50; r = 5; m = 1000;       % Symmetric tests
d1 = 40; d2 = 50; r = 5;       % Non-symmetric tests
```

## Understanding Test Results

### Pass/Fail Criteria

| Test | Pass Criteria | Typical Value |
|------|--------------|---------------|
| Formula correctness | Relative error < 1e-10 | 1e-14 to 1e-16 |
| Factor symmetry | ||U{i} - U{j}||_F < 1e-10 | 0 (exactly) |
| Method consistency | ||X_i - X_j||_F < 1e-3 | 1e-10 to 1e-6 |
| Reconstruction | Error < 0.1 | 1e-3 to 1e-2 |
| Matrix symmetry | Relative asymmetry < 1e-10 | 1e-15 |

### Interpreting Failures

#### Formula Verification Fails
**Symptoms**: G_mat formula error > 1e-6

**Likely causes**:
1. Bug in `tensor_mode_product`
2. Incorrect Kronecker product dimensions
3. MATLAB reshape behavior mismatch

**Action**: Check implementation against `MATLAB_RESHAPE_DEFINITION.tex`

#### Symmetry Issues
**Symptoms**: Expected symmetry not observed

**Likely causes**:
1. Numerical precision accumulation
2. Factors not actually equal (check differences)
3. Input tensor not symmetric

**Action**: Verify preconditions, check condition numbers

#### Method Disagreement
**Symptoms**: Extraction methods produce different X

**Likely causes**:
1. Sign ambiguity in eigenvectors
2. Numerical instability (check condition numbers)
3. Implementation bug in one method

**Action**: Compare intermediate results (eigenvalues, eigenvectors)

## Performance Benchmarks

### Expected Runtimes (Apple M1, MATLAB R2023a)

| Test | Small | Medium | Large |
|------|-------|--------|-------|
| Core structure | 0.05s (d=10, r=2) | 0.10s (d=20, r=3) | 2.0s (d=50, r=5) |
| Symmetric extraction | 0.5s (d=10, r=2) | 2.0s (d=20, r=3) | 20s (d=50, r=5) |
| Non-symmetric | 0.05s (d₁=10, d₂=12, r=2) | 0.20s (d₁=15, d₂=20, r=3) | 3.0s (d₁=40, d₂=50, r=5) |
| **Total** | **0.6s** | **2.3s** | **25s** |

### Scalability Notes

**Memory usage scales as**:
- Tensor operations: O(d₁²d₂²) for non-symmetric, O(d⁴) for symmetric
- Matrix formula: O(d₁²d₂²) but better cache locality
- Extraction methods: O(d₁d₂) to O(d₁²d₂²)

**For large problems** (d₁·d₂ > 10000):
- Use partial symmetry case when applicable
- Consider sparse tensor operations (not implemented)
- Method 3 (core eigendecomposition) most efficient
- Non-square matrices (d₁ ≠ d₂) may have better conditioning

## File Organization

```
test/
├── run_all_tucker_tests.m           # Run all tests with summary
│
├── test_core_tensor_structure.m     # Mathematical proposition test
├── test_tucker_spectral_symmetry.m  # Symmetric + extraction methods
├── test_tucker_nonsymmetric.m       # Non-symmetric cases
│
├── README_CORE_TENSOR_TESTS.md      # Core test documentation
├── RUN_EXTRACTION_TESTS.md          # Extraction methods guide
├── NONSYMMETRIC_TEST_GUIDE.md       # Non-symmetric guide
├── EXTRACTION_METHOD_3_EXPLANATION.md # Method 3 detailed explanation
├── MATLAB_RESHAPE_DEFINITION.tex    # Mathematical definitions
└── TUCKER_TESTS_README.md          # This file
```

## Theory References

### Key Mathematical Results

1. **Matricization Formula**:
   ```
   G_mat = (U₂' ⊗ U₁') H_mat (U₄' ⊗ U₃')'
   ```
   Proven in: `test_core_tensor_structure.m`

2. **Symmetry Preservation**:
   ```
   U₁=U₃ ∧ U₂=U₄ ∧ H_mat symmetric → G_mat symmetric
   ```
   Verified in: `test_tucker_nonsymmetric.m` (partial case)

3. **Rank-1 Structure**:
   ```
   H_mat ≈ λvv' → G_mat ≈ λqq' where q = (U₂' ⊗ U₁')v
   ```
   Tested in: `test_core_tensor_structure.m`

### Papers

1. **Tucker Decomposition**: 
   - Kolda, T. G., & Bader, B. W. (2009). "Tensor decompositions and applications." SIAM review, 51(3), 455-500.

2. **Mode Products**:
   - De Lathauwer, L., De Moor, B., & Vandewalle, J. (2000). "A multilinear singular value decomposition." SIAM journal on Matrix Analysis and Applications, 21(4), 1253-1278.

3. **Kronecker Products**:
   - Van Loan, C. F. (2000). "The ubiquitous Kronecker product." Journal of computational and applied mathematics, 123(1-2), 85-100.

## Troubleshooting

### Common Issues

#### Issue: Tests fail with dimension mismatch
**Solution**: Check that all helper functions use correct dimensions. Verify d₁=d₂ for symmetric cases.

#### Issue: Numerical precision warnings
**Solution**: This is normal. Floating-point arithmetic has ~1e-16 precision. Results with error < 1e-10 are excellent.

#### Issue: Memory errors for large d
**Solution**: Reduce d or r. Tensor operations require O(d⁴) memory. For d=100, this is ~40GB.

#### Issue: Tests pass individually but fail in batch
**Solution**: Clear workspace between tests. Use `run_all_tucker_tests.m` which handles this.

### Debug Mode

Enable detailed output in any test:
```matlab
use_debug = true;    % In test parameters
params.debug = true; % For TuckerTensor operations
```

This shows:
- Memory usage at each step
- Intermediate tensor norms
- Eigenvalue spectra
- Detailed convergence info

## Integration with Main Code

These tests validate components used in:
- `initialize_tensor_lift_tucker_spectral.m` - Spectral initialization
- `TuckerTensor.m` - Tucker tensor class
- `solve_RGD_tucker.m` - Riemannian gradient descent
- Phase diagram generation scripts

**Before modifying core tensor code**, run:
```matlab
run_all_tucker_tests  % Ensure nothing breaks
```

## Contributing

When adding new Tucker tests:

1. Follow naming convention: `test_tucker_*.m`
2. Include detailed comments and documentation
3. Add to `run_all_tucker_tests.m`
4. Update this README
5. Verify all existing tests still pass

## Support

For questions or issues:
1. Check relevant documentation (see File Organization above)
2. Review test output carefully (error messages are detailed)
3. Enable debug mode for more information
4. Check mathematical definitions in `.tex` files

## License

These tests are part of the GeneralPlatform MATLAB project.

---

**Last Updated**: 2025-12-23  
**MATLAB Version**: R2023a or later  
**Dependencies**: Statistics Toolbox (for `orth` function)

