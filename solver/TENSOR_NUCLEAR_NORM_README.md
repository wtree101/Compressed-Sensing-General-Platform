# Tensor Nuclear Norm Initialization

This directory contains implementation of tensor nuclear norm minimization for low-rank matrix recovery initialization.

## Files

### Core Solver
- **`solver/solve_tensor_nuclear_norm.m`**: Main convex optimization solver
  - Minimizes: sum of nuclear norms of all 4 tensor unfoldings
  - Subject to: measurement constraints y_i = ⟨A_i ⊗ A_i, T⟩
  - Algorithm: ADMM with conjugate gradient for T-update
  - Output: Extracted rank-r matrix from tensor

### Initialization Function
- **`Initialization_groundtruth/initialize_tensor_nuclear_norm.m`**: Initialization wrapper
  - Runs tensor nuclear norm solver for few iterations (default: 10)
  - Returns initial estimate X0 for local refinement
  - Can be used as drop-in replacement for other initialization methods

### Test Scripts
- **`test/test_tensor_nuclear_norm.m`**: Basic solver test
  - Tests tensor nuclear norm minimization standalone
  - Visualizes convergence and reconstruction

- **`test/test_nuclear_init_plus_refine.m`**: Complete pipeline test
  - Stage 1: Tensor nuclear norm initialization (10 iterations)
  - Stage 2: Local refinement with gradient descent (200 iterations)
  - Compares with random initialization
  - Shows convergence curves and timing

### Phase Diagram Integration
- **`matrix_recovery/Phasediagram_nuclear_init.m`**: Phase diagram script
  - Uses tensor nuclear norm for initialization
  - Follows with local refinement (solve_PGD_amplitude)
  - Compatible with existing phase diagram framework

## Usage

### As Standalone Solver
```matlab
% Create operator with A_cells
operator.A_cells = {A_1, A_2, ..., A_m};

% Run solver
[X_recovered, info] = solve_tensor_nuclear_norm(operator, y, d, ...
    'rank', 2, 'max_iter', 100, 'lambda', 1.0, 'rho', 0.1);
```

### As Initialization for Local Refinement
```matlab
% Initialize
X0 = initialize_tensor_nuclear_norm(operator, y, d, ...
    'rank', 2, 'max_iter', 10, 'verbose', 1);

% Refine with gradient descent
[X_final, output] = solve_PGD_amplitude(operator, y, r, d, d, nonlinear_func, ...
    'X0', X0, 'T', 200, 'mu', 0.01, 'Xstar', Xstar);
```

### In Phase Diagram Framework
```matlab
% In Phasediagram_nuclear_init.m or similar:
trial_func = @onetrial_Mat;
alg_func = @solve_PGD_amplitude;
init_method = @initialize_tensor_nuclear_norm;  % Use TNN init
nonlinear_func = @(y) abs(y);

% Run experiments...
```

## Algorithm Details

### Tensor Nuclear Norm Minimization
The solver minimizes:
```
min_T  sum_{k=1}^4 w_k * ||T_(k)||_*
s.t.   y_i = ⟨A_i ⊗ A_i, T⟩, for all i
```

Where:
- T is a 4th-order tensor (d×d×d×d)
- T_(k) is the mode-k unfolding (matricization)
- ||·||_* is the nuclear norm (sum of singular values)
- A_i ⊗ A_i are 4th-order measurement tensors

### ADMM Algorithm
1. **T-update**: Solve least squares with measurement constraints
   - Uses conjugate gradient for efficiency
   - Balances ADMM consensus and measurement fit

2. **Z-update**: Apply nuclear norm proximal operator
   - Singular value soft-thresholding on each unfolding
   - Separately for each mode k=1,2,3,4

3. **U-update**: Update dual variables (Lagrange multipliers)

### Matrix Extraction
From tensor T, extract matrix X using:
- Mode-(1,2) matricization: T → T_mat (d²×d²)
- Eigendecomposition: extract leading eigenvector
- Reshape and project to rank-r symmetric space
- Uses `utilities_tensor/extract_matrix_from_tensor.m`

## Parameters

### Solver Parameters
- `max_iter`: Maximum iterations (default: 1000)
- `tol`: Convergence tolerance (default: 1e-6)
- `lambda`: Measurement fit weight (default: 1.0)
- `rho`: ADMM penalty parameter (default: 1.0)
- `nuclear_weight`: Weights for mode nuclear norms [w1,w2,w3,w4] (default: [1,1,1,1])
- `rank`: Target rank for extraction (default: from X_true or 5)

### Initialization Parameters
- `max_iter`: Iterations for init (default: 10, enough for good warm start)
- `lambda`: Regularization (default: 1.0)
- `rho`: ADMM penalty (default: 0.1, lighter penalty for initialization)
- `normalize`: Normalize output (default: true)

## Performance Notes

### Computational Cost
- **Per iteration**: O(m·d⁴) for measurement terms + O(d⁴) for SVD on unfoldings
- **Memory**: O(d⁴) for tensor T + O(d·d³) for each unfolding
- **Initialization**: ~10 iterations sufficient (vs 200+ for full convergence)

### When to Use
- **Good for**: Warm starting local refinement algorithms
- **Better than random**: Provides structured initialization from convex relaxation
- **Trade-off**: More expensive than spectral methods, but more principled

### Typical Behavior
- Iterations 1-5: Rapid improvement from zero initialization
- Iterations 6-10: Refinement to reasonable initialization
- Iterations 10+: Slow convergence to convex optimum (not needed for init)

## Examples

See test scripts for complete examples:
- `test/test_tensor_nuclear_norm.m`: Basic usage
- `test/test_nuclear_init_plus_refine.m`: Initialization + refinement pipeline
- `matrix_recovery/Phasediagram_nuclear_init.m`: Phase diagram generation

## References

This implementation is based on convex relaxation approaches for low-rank matrix recovery:
- Tensor nuclear norm as convex surrogate for tensor rank
- ADMM for efficient optimization with splitting
- Phase retrieval via lifting to 4th-order tensor space
