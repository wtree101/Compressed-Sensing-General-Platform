# Tensor Lift Initialization Method

## Overview

The **Tensor Lift Initialization** method provides a powerful initialization strategy for symmetric low-rank matrix recovery by lifting the problem to a fourth-order tensor space, solving a tensor power method, and extracting the matrix.

## Motivation

For symmetric matrix recovery problems with structure X = UU^T, lifting to the tensor space T = X ⊗ X can provide better initialization by:
1. **Exploiting Structure**: Captures the self-Kronecker product structure inherent in phase retrieval
2. **Improved Spectral Properties**: Tensor power method can have better convergence for certain problem instances
3. **Rank Flexibility**: Natural handling of symmetric low-rank structure

## Mathematical Formulation

### Matrix Problem
Given measurements: y_i = |⟨A_i, X⟩| where X ∈ ℝ^(d×d) is symmetric, rank-r

### Tensor Lifting
1. Lift to tensor: T = X ⊗ X ∈ ℝ^(d×d×d×d)
2. Lift operators: A_i → A_i ⊗ A_i
3. Measurements remain: y_i = |⟨A_i ⊗ A_i, T⟩|
4. Solve tensor power method iterations
5. Extract matrix from final tensor

### Algorithm

```
Input: y (measurements), operator (A, A*), d, params
Output: X0 (initialized matrix), U0 (factor), history

1. Lift operators to tensor space:
   For each i: A_tensor[i] = A_i ⊗ A_i
   
2. Initialize random tensor: T^(0) ← random, normalize

3. Tensor power iterations (t = 1 to T_tensor):
   T^(t) = A_tensor^* (y^2 .* A_tensor(T^(t-1)))
   T^(t) ← T^(t) / ||T^(t)||
   
4. Extract matrix: X0 = extract_matrix_from_tensor(T^(T_tensor))

5. Symmetrize: X0 ← (X0 + X0^T) / 2

6. Apply projection if provided

Return X0, U0, history
```

## Function Signature

```matlab
function [X0, U0, history] = initialize_tensor_lift(y, operator, d1, d2, params)
```

### Inputs

- `y` - Measurement vector (m × 1)
- `operator` - Struct with fields:
  - `.A`: Forward operator @(X) A*X(:)
  - `.A_star`: Adjoint operator @(y) reshape(A'*y, [d1,d2])
- `d1, d2` - Matrix dimensions (must be equal: d1 = d2)
- `params` - Struct with fields:
  - `.T_tensor`: Number of tensor power iterations (default: 5)
  - `.r`: Target rank for extracted matrix
  - `.Xstar`: (optional) Ground truth for error tracking
  - `.projection`: (optional) Projection function @(X)
  - `.verbose`: (optional) Print progress (default: false)

### Outputs

- `X0` - Initialized symmetric matrix (d × d)
- `U0` - Factor matrix from SVD of X0
- `history` - Struct with:
  - `.method`: 'tensor_lift'
  - `.tensor_norms`: Tensor norms at each iteration
  - `.tensor_errors`: Tensor errors (if Xstar provided)
  - `.matrix_errors`: Matrix errors after extraction (if Xstar provided)
  - `.iterations`: Number of iterations performed
  - `.final_error`: Final matrix error (if Xstar provided)

## Usage Examples

### Example 1: Basic Usage

```matlab
% Setup problem
d = 20; m = 400; r = 2;
A = randn(m, d*d);
operator.A = @(X) A * X(:);
operator.A_star = @(z) reshape(A' * z, [d, d]);
y = abs(operator.A(Xstar)) / sqrt(m);

% Initialize with tensor lift
params = struct();
params.r = r;
params.T_tensor = 5;  % 5 tensor iterations

[X0, U0, history] = initialize_tensor_lift(y, operator, d, d, params);
```

### Example 2: With Projection and Ground Truth

```matlab
params = struct();
params.r = r;
params.T_tensor = 10;
params.Xstar = Xstar;  % For error tracking
params.projection = @(X) rank_projection(X, r);
params.verbose = true;

[X0, U0, history] = initialize_tensor_lift(y, operator, d, d, params);

% Check convergence
figure;
semilogy(history.matrix_errors);
xlabel('Tensor Iteration');
ylabel('Matrix Error');
title('Tensor Lift Convergence');
```

### Example 3: In onetrial_Mat.m

```matlab
% In main script
params.init = @initialize_tensor_lift;
params.r = 3;
params.T_tensor = 5;
params.m = m;
params.Xstar = Xstar;

% Will be called automatically in onetrial_Mat
[output, is_success] = onetrial_Mat(params);
```

### Example 4: Via set_init

```matlab
% Get tensor lift initialization
[init_name, init_handle] = set_init(3);  % 3 = Tensor Lift
fprintf('Using: %s\n', init_name);  % Prints: Tensor_Lift_Init

% Setup params
params.r = r;
params.T_tensor = 5;
params.Xstar = Xstar;

% Call initialization
[X0, U0, history] = init_handle(y, operator, d, d, params);
```

## Comparison with Other Methods

### vs. Standard Power Method

| Feature | Power Method | Tensor Lift |
|---------|-------------|-------------|
| Space | Matrix (d×d) | Tensor (d×d×d×d) |
| Iterations | 20-50 typical | 5-10 typical |
| Exploits Structure | No | Yes (X ⊗ X) |
| Symmetric Only | No | Yes |
| Memory | O(d²) | O(d⁴) |
| Best For | General matrices | Symmetric, low d |

### Performance Guidelines

**When to use Tensor Lift:**
- ✅ Symmetric matrices (X = X^T)
- ✅ Small to medium dimensions (d ≤ 30)
- ✅ Low-rank structure (r ≪ d)
- ✅ When structure X = UU^T is known
- ✅ Phase retrieval problems

**When to use Power Method:**
- ✅ Large dimensions (d > 30)
- ✅ Non-symmetric matrices
- ✅ Memory constrained
- ✅ Faster initialization needed

## Complexity Analysis

### Time Complexity
- Operator lifting: O(m × d⁴)
- Per tensor iteration: O(m × d⁴)
- Total: O(T_tensor × m × d⁴)

### Space Complexity
- Tensor storage: O(d⁴)
- Operator storage: O(m × d⁴)
- Total: O(m × d⁴)

**Note:** Exponential in dimension! Use only for d ≤ 30.

## Parameter Tuning

### T_tensor (Number of Iterations)

- **Default: 5**
- Fewer iterations (3-5): Fast, good for easy problems
- More iterations (10-20): Better convergence, harder problems
- Diminishing returns after 10-15 iterations typically

**Rule of thumb:** Start with T_tensor = 5, increase if initialization error > 0.1

### r (Target Rank)

Must match true rank of ground truth for best results.

### projection

Optional rank projection for regularization:
```matlab
params.projection = @(X) rank_projection(X, r);
```

## Implementation Details

### Operator Lifting Process

The function automatically lifts matrix operators to tensor space:

1. **Extract measurement matrices** from operator structure
2. **Symmetrize** each A_i: Ã_i = (A_i + A_i^T)/2
3. **Kronecker product**: A_i ⊗ A_i ∈ ℝ^(d²×d²)
4. **Flatten** and store as rows of A_tensor

### Matrix Extraction

Uses eigenvalue-based extraction:
1. Matricize tensor: T → T_mat ∈ ℝ^(d²×d²)
2. Symmetrize: T_mat ← (T_mat + T_mat^T)/2
3. Compute leading eigenvector
4. Reshape to matrix: v → X ∈ ℝ^(d×d)
5. Project to rank-r

## Limitations

1. **Dimension Constraint**: d ≤ 30 due to O(d⁴) complexity
2. **Symmetry Required**: Only works for symmetric matrices
3. **Memory Intensive**: Requires O(m × d⁴) memory
4. **Non-convex**: May not always improve over power method

## Testing

Run the comprehensive test:

```matlab
cd test
test_tensor_lift_initialization
```

This will:
- Compare tensor lift (T=5, T=10) vs power method (T=20)
- Show convergence curves
- Display tensor vs matrix errors
- Visualize recovered matrices
- Save results to .mat file

## References

This method is based on tensor lifting techniques for phase retrieval:
- Fourth-order tensor formulation for symmetric matrices
- Power method on tensor manifold
- Matrix extraction via eigendecomposition

## Integration with Existing Code

### Unified Signature ✅

Follows the same signature as all other initialization methods:
```matlab
[X0, U0, history] = initialize_tensor_lift(y, operator, d1, d2, params)
```

### Compatible with:
- ✅ `onetrial_Mat.m` - via `params.init`
- ✅ `set_init(3)` - via init_flag
- ✅ All solvers - outputs standard X0, U0
- ✅ Error tracking - via `params.Xstar`

### Example Integration:

```matlab
% In Phasediagram or main script
params.init = @initialize_tensor_lift;
params.init_flag = 3;  % For set_init
params.T_tensor = 5;
params.r = rank_true;

% Run experiment
[output, is_success] = onetrial_Mat(params);
```

## Summary

The tensor lift initialization:
- ✅ Provides alternative initialization via tensor space
- ✅ Can improve convergence for symmetric low-rank problems
- ✅ Follows unified signature convention
- ✅ Includes comprehensive error tracking
- ⚠️ Limited to moderate dimensions (d ≤ 30)
- ⚠️ Requires symmetric matrices
- 📊 Best with T_tensor = 5-10 iterations

Use when: symmetric structure + moderate dimension + low rank!
