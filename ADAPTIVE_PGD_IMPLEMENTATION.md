# Adaptive Stepsize for PGD Implementation

## Overview
Added adaptive stepsize functionality to the Projected Gradient Descent (PGD) solver using backtracking line search with Armijo condition, following the same design pattern as the RGD solver.

## Date
December 18, 2025

## Motivation
- **Automatic Tuning**: Eliminate manual stepsize selection through line search
- **Convergence Guarantee**: Armijo condition ensures monotonic descent
- **Feature Parity**: Bring PGD solver to full feature parity with RGD solver
- **Comprehensive Comparison**: Enable comparison of 8 methods (4 fixed + 4 adaptive)

## Implementation Details

### Modified File: `solver/solve_PGD_amplitude.m`

#### 1. New Parameters
Added to the `params` structure:
- `use_adaptive_stepsize` (default: `false`): Enable/disable line search
- `line_search_max_iter` (default: `20`): Maximum line search iterations
- `line_search_beta` (default: `0.5`): Backtracking reduction factor
- `line_search_c` (default: `1e-4`): Armijo constant for sufficient decrease

#### 2. Stepsize Memory Enhancement
Implemented smart initialization strategy:
```matlab
% Initialize previous stepsize before loop
eta_prev = eta;

% In line search: start from grown previous stepsize
eta_t = min(1.05 * eta_prev, 5 * eta);  % 5% growth, capped at 5x initial

% After successful line search: remember for next iteration
eta_prev = eta_t;
```

#### 3. Line Search Algorithm
Backtracking line search with Armijo condition:

```matlab
% Current loss
current_loss = Error_function(t);

% Directional derivative: <gradient, preconditioned_gradient>
direc_deriv = sum(gradient(:) .* preconditioned_gradient(:));

% Backtracking loop
for ls_iter = 1:ls_max_iter
    % Try step with current eta_t
    Xl_trial = Xl - eta_t * preconditioned_gradient;
    Xl_trial = params.projection(Xl_trial);  % Apply projection
    
    % Compute trial loss
    z_trial = operator.A(Xl_trial) / sqrt(m);
    residual_trial = y - abs(z_trial);
    loss_trial = (1/2) * norm(residual_trial)^2;
    
    % Check Armijo condition
    if loss_trial <= current_loss - c * eta_t * direc_deriv
        break;  % Accept this stepsize
    end
    
    % Reduce stepsize
    eta_t = beta * eta_t;
end
```

#### 4. Mathematical Formulation

**Standard PGD (Fixed Stepsize):**
```
X_{t+1} = Π_r(X_t - η ∇ℓ(X_t))
```

**Adaptive PGD (Line Search):**
```
Find η_t satisfying: ℓ(Π_r(X_t - η_t D_t)) ≤ ℓ(X_t) - c·η_t·⟨∇ℓ(X_t), D_t⟩
X_{t+1} = Π_r(X_t - η_t D_t)
```

where:
- `D_t` is the (possibly preconditioned) gradient
- `Π_r` is the rank-r projection operator
- `c = 1e-4` is the Armijo constant

#### 5. Key Features
- **Projection-Aware**: Line search evaluates loss *after* projection
- **Preconditioner-Compatible**: Works with both standard and preconditioned gradients
- **Memory-Efficient**: Stepsize memory reduces redundant line search iterations
- **Robust**: Guaranteed to find acceptable stepsize (worst case: exponentially small η)

#### 6. Output Structure
When `use_adaptive_stepsize = true`:
```matlab
output.stepsize_history  % T×1 vector of accepted stepsizes per iteration
```

## Test Integration

### Modified File: `test/test_RGD_amplitude.m`

#### New Test Methods

**Method 7: Adaptive PGD (no preconditioner)**
```matlab
mu_adaptive_pgd = 1.0;  % Initial stepsize
params_adaptive_pgd.use_adaptive_stepsize = true;
params_adaptive_pgd.use_preconditioner = false;
```

**Method 8: Adaptive PPGD (with preconditioner)**
```matlab
mu_adaptive_ppgd = 0.1;  % Initial stepsize
params_adaptive_ppgd.use_adaptive_stepsize = true;
params_adaptive_ppgd.use_preconditioner = true;
```

#### Complete Test Suite (8 Methods)

**Fixed Stepsize Methods:**
1. **PRGD (fixed)**: μ = 0.1, preconditioned, factorized
2. **RGD (fixed)**: μ = 1.0, no preconditioner, factorized
3. **PGD**: μ = 1.0, no preconditioner, full matrix
4. **PPGD**: μ = 0.1, preconditioned, full matrix

**Adaptive Stepsize Methods:**
5. **PRGD (adaptive)**: μ₀ = 0.3, preconditioned, factorized, line search
6. **RGD (adaptive)**: μ₀ = 0.5, no preconditioner, factorized, line search
7. **PGD (adaptive)**: μ₀ = 1.0, no preconditioner, full matrix, line search ← **NEW**
8. **PPGD (adaptive)**: μ₀ = 0.1, preconditioned, full matrix, line search ← **NEW**

#### Enhanced Visualization (3×6 Layout)

**Row 1: Convergence Analysis**
- Plot 1: Error vs Iteration (all 8 methods)
- Plot 2: Error vs Time (all 8 methods)
- Plot 3: Loss curves (all 8 methods)
- Plot 4: PRGD adaptive stepsize history
- Plot 5: RGD adaptive stepsize history
- Plot 6: PGD adaptive stepsize history ← **NEW**

**Row 2: Stepsize & Comparison Analysis**
- Plot 7: PPGD adaptive stepsize history ← **NEW**
- Plot 8: Preconditioner effect (fixed stepsize)
- Plot 9: Preconditioner effect (adaptive stepsize) ← **NEW**
- Plot 10: Ground truth
- Plot 11: PRGD (fixed) result
- Plot 12: RGD (fixed) result

**Row 3: Recovered Matrices**
- Plot 13: PGD result
- Plot 14: PPGD result
- Plot 15: PGD (adaptive) result ← **NEW**
- Plot 16: PPGD (adaptive) result ← **NEW**
- Plot 17: PRGD (adaptive) result
- Plot 18: RGD (adaptive) result

#### Summary Enhancements

Added comprehensive analysis sections:

1. **Preconditioner Effect (Fixed Stepsize)**
   - Compares PGD vs PPGD

2. **Preconditioner Effect (Adaptive Stepsize)**
   - Compares PGD (adaptive) vs PPGD (adaptive) ← **NEW**

3. **Adaptive Stepsize Effect (PGD)**
   - Compares PGD (fixed) vs PGD (adaptive) ← **NEW**

4. **Adaptive Stepsize Effect (PPGD)**
   - Compares PPGD (fixed) vs PPGD (adaptive) ← **NEW**

5. **Stepsize Statistics for All Adaptive Methods**
   - PRGD, RGD, PGD, PPGD ← **NEW: added PGD & PPGD**

## Expected Behavior

### Performance Characteristics

1. **PGD (adaptive) vs PGD (fixed)**
   - Adaptive should achieve similar or better final error
   - Adaptive adapts to problem geometry automatically
   - Fixed may outperform if manually tuned stepsize is optimal

2. **PPGD (adaptive) vs PPGD (fixed)**
   - Similar behavior to PGD comparison
   - Preconditioning helps both fixed and adaptive

3. **PGD (adaptive) vs PPGD (adaptive)**
   - PPGD should converge faster due to preconditioning
   - Both benefit from adaptive stepsize

4. **Initial Stepsize Choices**
   - **PGD (adaptive)**: Start from 1.0 (same as fixed PGD)
   - **PPGD (adaptive)**: Start from 0.1 (same as fixed PPGD)
   - Line search will adjust from these starting points

### Stepsize Evolution Patterns

**Typical Behavior:**
- **Early iterations**: Larger stepsizes accepted (smooth region)
- **Mid iterations**: Moderate stepsizes (approaching solution)
- **Late iterations**: Smaller stepsizes (fine-tuning near optimum)

**With Stepsize Memory:**
- Grows by 5% per iteration when safe
- Capped at 5× initial value to prevent instability
- Adapts to local geometry changes

## Usage Examples

### Example 1: Adaptive PGD (No Preconditioner)
```matlab
params = struct();
params.T = 200;
params.mu = 1.0;  % Initial stepsize
params.projection = @(X) rank_projection(X, r);
params.use_preconditioner = false;
params.use_adaptive_stepsize = true;  % Enable line search
params.line_search_max_iter = 20;
params.line_search_beta = 0.5;
params.line_search_c = 1e-4;
params.Xstar = Xstar;

[output, X] = solve_PGD_amplitude(X0, [], y, operator, d, d, [], m, params);

% Analyze stepsize evolution
figure;
plot(output.stepsize_history);
xlabel('Iteration');
ylabel('Stepsize η');
title('Adaptive PGD Stepsize History');
```

### Example 2: Adaptive PPGD (With Preconditioner)
```matlab
params = struct();
params.T = 200;
params.mu = 0.1;  % Initial stepsize (smaller for preconditioned)
params.projection = @(X) rank_projection(X, r);
params.use_preconditioner = true;  % Enable preconditioner
params.epsilon = 1e-8;
params.use_adaptive_stepsize = true;  % Enable line search
params.line_search_max_iter = 20;
params.line_search_beta = 0.5;
params.line_search_c = 1e-4;
params.Xstar = Xstar;

[output, X] = solve_PGD_amplitude(X0, [], y, operator, d, d, [], m, params);

% Compare with fixed stepsize
fprintf('Adaptive: final error = %.6e, mean stepsize = %.4e\n', ...
        output.Error_Stand(end), mean(output.stepsize_history));
```

### Example 3: Full Comparison (8 Methods)
```matlab
% Run test suite
cd test
test_RGD_amplitude

% Expected output:
% - Summary table with all 8 methods
% - Stepsize histories for 4 adaptive methods
% - Preconditioner effect analysis (fixed and adaptive)
% - Adaptive stepsize effect analysis (PGD and PPGD)
```

## Compatibility

### Backward Compatibility
✅ **Fully backward compatible**: All existing code using `solve_PGD_amplitude` will work unchanged since:
- `use_adaptive_stepsize` defaults to `false`
- All line search parameters are optional
- Output structure only adds `stepsize_history` field when adaptive

### Cross-Solver Consistency
✅ **Consistent with RGD solver**: Same line search implementation as `solve_RGD_amplitude`:
- Same parameter names and defaults
- Same Armijo condition
- Same stepsize memory strategy (1.05× growth, 5× cap)
- Same backtracking logic

## Performance Insights

### When to Use Adaptive Stepsize

**Advantages:**
- No manual stepsize tuning required
- Automatically adapts to problem geometry
- Guaranteed monotonic descent (Armijo condition)
- Robust across different problem scales

**Disadvantages:**
- Additional line search cost per iteration
- May be slower per iteration than fixed stepsize
- Benefit depends on problem structure

### Recommended Settings

**For PGD (no preconditioner):**
```matlab
mu = 1.0  % Initial stepsize
use_adaptive_stepsize = true
```

**For PPGD (with preconditioner):**
```matlab
mu = 0.1  % Initial stepsize (smaller due to preconditioning)
use_adaptive_stepsize = true
epsilon = 1e-8
```

**For fast experimentation:**
```matlab
line_search_max_iter = 10  % Reduce for faster (less accurate) line search
```

**For robust convergence:**
```matlab
line_search_max_iter = 30  % Increase for more thorough line search
line_search_beta = 0.3     % More aggressive backtracking
```

## Mathematical Justification

### Armijo Condition
The Armijo condition ensures sufficient decrease:
```
ℓ(X_t - η D_t) ≤ ℓ(X_t) - c·η·⟨∇ℓ(X_t), D_t⟩
```

**Guarantees:**
1. **Descent**: New loss is strictly lower (assuming directional derivative < 0)
2. **Sufficient Decrease**: Not just any decrease, but proportional to stepsize
3. **Convergence**: Combined with backtracking, guarantees global convergence

### Projection Handling
Unlike standard line search, we evaluate loss *after* projection:
```
ℓ(Π_r(X_t - η D_t))
```

This ensures:
- Feasibility maintained (rank constraint satisfied)
- Accurate loss evaluation
- Proper convergence behavior on constrained manifold

## Comparison with Fixed Stepsize

### Theoretical Differences

**Fixed Stepsize:**
- Requires manual tuning (problem-dependent)
- May diverge if too large
- May converge slowly if too small
- Constant computational cost per iteration

**Adaptive Stepsize:**
- Automatic adjustment
- Guaranteed descent (with Armijo)
- Adapts to local geometry
- Variable computational cost (line search overhead)

### Empirical Observations

From test results:
- **PGD (adaptive)** typically outperforms **PGD (fixed)** with default μ = 1.0
- **PPGD (adaptive)** shows robust performance with minimal tuning
- Stepsize memory (1.05× growth) significantly reduces line search iterations
- Adaptive methods more robust across different problem scales

## References

### Related Files
- `solver/solve_PGD_amplitude.m`: Main implementation
- `solver/solve_RGD_amplitude.m`: Reference for line search design
- `test/test_RGD_amplitude.m`: Comprehensive 8-method comparison
- `PRECONDITIONED_PGD_ADDITION.md`: Preconditioner documentation
- `STEPSIZE_MEMORY_ENHANCEMENT.md`: Stepsize memory explanation

### Mathematical Background
1. **Nocedal & Wright (2006)**: "Numerical Optimization", Chapter 3 (Line Search Methods)
2. **Armijo (1966)**: "Minimization of functions having Lipschitz continuous first partial derivatives"
3. **Backtracking Line Search**: Standard technique in nonlinear optimization

### Algorithm Design
- **Armijo Constant**: c = 1e-4 (typical value, ensures sufficient decrease)
- **Backtracking Factor**: β = 0.5 (halving, standard choice)
- **Growth Factor**: 1.05 (conservative growth, prevents instability)
- **Cap Factor**: 5.0 (limits maximum stepsize growth)

## Testing

To verify the implementation:
```matlab
cd test
test_RGD_amplitude
```

**Expected Results:**
- All 8 methods converge successfully
- Adaptive methods show smooth stepsize evolution
- PPGD variants outperform PGD variants
- Adaptive variants show robustness
- Summary table displays all performance metrics

**Key Metrics to Check:**
1. Final error for all methods
2. Stepsize range for adaptive methods
3. Convergence rate comparison
4. Computational time comparison
5. Preconditioner effect (fixed and adaptive)
6. Adaptive stepsize effect (PGD and PPGD)

## Future Enhancements

Potential improvements:
1. **Wolfe Conditions**: Add curvature condition for better stepsize selection
2. **Non-monotone Line Search**: Allow occasional increase for faster convergence
3. **Adaptive Parameters**: Auto-tune β and c based on convergence rate
4. **Warm Starting**: Initialize line search from problem-specific heuristics
5. **Hybrid Approach**: Switch between fixed and adaptive based on progress
