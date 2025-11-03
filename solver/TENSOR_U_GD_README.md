# Tensor U Gradient Descent Solver

## Overview
This solver performs gradient descent directly on the factor matrix **U ∈ ℝ^(d×r)** for the tensor-lifted matrix recovery problem.

### Formulation
- **Matrix**: X = UU^T (d × d symmetric, rank r)
- **Tensor**: T = X ⊗ X (d×d×d×d 4th-order tensor)
- **Measurements**: y_i = ⟨A_i ⊗ A_i, T⟩ = trace(A_iXA_iX)
- **Loss**: f(U) = ½||y - A(T)||²

### Gradient
The gradient with respect to U is:
```
∇_U f = 4 ∑_i residual_i · (A_iXA_iX)U
```
where `residual_i = y_i - ⟨A_i⊗A_i, X⊗X⟩` and `X = UU^T`.

## Files Created

### 1. `solve_tensor_U_GD.m` (Core Solver)
Main optimization algorithm:
- Input: measurements y, operators A_cells, initial U0
- Output: final U, X = UU^T, convergence history
- Performs vanilla gradient descent with fixed step size

### 2. `onetrial_tensor_U_GD.m` (Wrapper)
Interface compatible with phase diagram experiments:
- Generates measurements
- Handles initialization
- Calls solver and returns error history

### 3. `test_tensor_U_GD.m` (Test Script)
Standalone test with visualization:
- Simple synthetic problem
- Convergence plots
- Comparison with ground truth

## Usage

### Standalone Test
```bash
cd test
matlab -batch "test_tensor_U_GD"
```

### With Phase Diagram
Edit `Phasediagram_tensor.m`:
```matlab
% Change trial function
trial_func = @onetrial_tensor_U_GD;

% Optional: set step size
trial_params.mu = 0.01;
```

Then run:
```bash
cd matrix_recovery
matlab -batch "Phasediagram_tensor"
```

## Key Features

✓ **Memory Efficient**: Never forms full d⁴ tensor
✓ **Simple Gradient**: Direct gradient on U (no Tucker decomposition)
✓ **Fast Forward**: y_i = trace(A_iXA_iX) computed efficiently
✓ **Symmetric X**: Automatically maintained via X = UU^T

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T` | 100 | Number of iterations |
| `mu` | 0.01 | Step size for gradient descent |
| `verbose` | false | Print progress |
| `Xstar` | - | Ground truth for error tracking |

## Comparison with Other Methods

| Method | Variables | Manifold | Memory |
|--------|-----------|----------|--------|
| Matrix PGD | X ∈ ℝ^(d×d) | None | O(d²) |
| Tucker RGD | G,U₁,U₂,U₃,U₄ | Tucker | O(4dr + r⁴) |
| **U-GD (this)** | **U ∈ ℝ^(d×r)** | **None** | **O(dr)** |

## Algorithm

```
Input: y, {A_i}, U0, T, μ
for t = 1 to T:
    X ← UU^T
    y_pred ← A(X⊗X)  // via trace(A_iXA_iX)
    residual ← y_pred - y
    grad_U ← 4 ∑_i residual_i · (A_iXA_iX)U
    U ← U - μ · grad_U
return U, X = UU^T
```

## Expected Performance

For d=20, r=1, m=500:
- Initial error: ~0.5-0.8
- Final error: <1e-6 (if converged)
- Time: <1 second for 100 iterations

## Advantages
1. **Simplest formulation**: Direct GD on U
2. **Low memory**: Only store U (d×r) + A_cells
3. **Automatic rank constraint**: X = UU^T is rank-r
4. **No manifold projection**: Standard gradient descent

## Limitations
1. **Fixed step size**: May need tuning for different problems
2. **No adaptive rank**: r must be specified
3. **Local minima**: May get stuck (like all non-convex methods)

## Notes
- Works best when r is correctly specified
- May need smaller step size for larger r
- Can be combined with initialization methods (power method, spectral, etc.)
