# Adaptive Step Size Implementation in RGD Solver

## Overview

The `solve_RGD_amplitude.m` solver now includes an adaptive step size feature using the **Barzilai-Borwein (BB) method** with simple back**Changes to `solve_RGD_amplitude.m`

1. **Added parameters** (lines ~20-30):
   - `use_adaptive_stepsize`, `backtrack_beta`
   - `max_linesearch_iter`, `alpha_min`, `alpha_max`

2. **BB state variables** (before main loop):
   - `X_prev`: Previous iterate
   - `G_prev`: Previous gradient

3. **BB step size computation** (lines ~250-280):
   - Compute `s = X_t - X_{t-1}` and `y = G_t - G_{t-1}`
   - BB formula: `α = <s,s> / <s,y>`
   - Clip to `[alpha_min, alpha_max]`

4. **Simple backtracking** (lines ~280-310):
   - Reduce by factor `beta` until loss decreases
   - No complex Armijo conditions

5. **Output tracking** (line ~85):
   - Added `Step_sizes` array to output structa proven, simple, and effective scheme that automatically adjusts the step size based on local curvature information.

## Key Features

### 1. Barzilai-Borwein Step Size
The BB method estimates the optimal step size using information from the previous iteration:
- **Formula:** `α_t = <s, s> / <s, y>` where:
  - `s = X_t - X_{t-1}` (change in iterate)
  - `y = ∇f(X_t) - ∇f(X_{t-1})` (change in gradient)
- **Intuition:** Approximates inverse Hessian locally without computing second derivatives
- **Benefits:** 
  - Simple and efficient (no complex heuristics)
  - Adapts naturally to problem scale and conditioning
  - Proven convergence properties

### 2. Simple Backtracking
If the BB step doesn't reduce the loss:
- Reduce step size by factor `beta` (default: 0.5)
- Continue until loss decreases or reach minimum step size
- **No complex Armijo conditions** - just require `f(x_trial) < f(x_current)`

### 3. Automatic Step Size Bounds
- Parameter: `alpha_min` (default: 1e-10) - prevents step size from vanishing
- Parameter: `alpha_max` (default: 1.0) - prevents excessive steps
- BB estimate is clipped to range `[alpha_min, alpha_max]`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_adaptive_stepsize` | false | Enable adaptive step size (Barzilai-Borwein) |
| `backtrack_beta` | 0.5 | Step size reduction factor in backtracking |
| `max_linesearch_iter` | 20 | Maximum backtracking iterations |
| `alpha_min` | 1e-10 | Minimum acceptable step size |
| `alpha_max` | 1.0 | Maximum step size |

## Usage Example

```matlab
% Setup problem
d = 50;  % Matrix dimension
r = 5;   % Rank
m = 5*d*r;  % Number of measurements

% Generate ground truth and measurements
[A, y, Xstar] = generate_problem(d, r, m);

% Initialize
X0 = spectral_initialization(A, y, d, r);

% Configure solver with adaptive step size
params = struct();
params.maxiter = 1000;
params.tol = 1e-8;
params.verbose = 1;

% Enable adaptive step size (Barzilai-Borwein)
params.use_adaptive_stepsize = true;
params.backtrack_beta = 0.5;      % Backtracking reduction factor
params.max_linesearch_iter = 20;  % Max backtracking iterations
params.alpha_min = 1e-10;          % Minimum step size
params.alpha_max = 1.0;            % Maximum step size

% Enable preconditioning for better performance
params.use_precondition = true;

% Solve
[output, X_recovered] = solve_RGD_amplitude(X0, [], y, operator, d, d, [], m, params);

% Check step size statistics
fprintf('Average step size: %.6e\n', mean(output.Step_sizes));
fprintf('Min step size: %.6e\n', min(output.Step_sizes));
fprintf('Max step size: %.6e\n', max(output.Step_sizes));

% Plot step size evolution
figure;
plot(output.Step_sizes, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Step Size α_t');
title('Adaptive Step Size Evolution');
grid on;
```

## Comparison: Fixed vs Adaptive Step Size

### Fixed Step Size
- **Pros:** Simple, predictable behavior
- **Cons:** 
  - Too large → may not converge
  - Too small → slow convergence
  - Requires manual tuning for each problem

### Adaptive Step Size (Barzilai-Borwein)
- **Pros:**
  - **Simple:** Only 2 main parameters (alpha_min, alpha_max)
  - **Automatic:** Adapts to local curvature without tuning
  - **Efficient:** Minimal overhead (just one extra gradient/iterate storage)
  - **Robust:** Works well across different problem scales
  - **Proven:** Well-established method with convergence guarantees
- **Cons:**
  - Stores previous iterate/gradient (small memory overhead)
  - May require a few backtracks initially

## Test Results

Run `test_RGD_amplitude.m` to compare four methods:
1. **Method 1: RGD** - Basic Riemannian Gradient Descent
2. **Method 2: PRGD** - Preconditioned RGD with fixed step size
3. **Method 3: PGD** - Projected Gradient Descent
4. **Method 4: PRGD-Adaptive** - Preconditioned RGD with adaptive step size

The test provides:
- Error vs iteration plots
- Error vs computation time plots
- Loss function convergence
- Step size evolution (for adaptive method)
- Visual comparison of recovered matrices
- Detailed performance statistics

## Technical Details

### Step Size Update Logic (Barzilai-Borwein)

```matlab
% Barzilai-Borwein step size estimation
if t == 1
    % First iteration: use provided initial step size
    alpha_t = alpha;
else
    % BB step size: α_t = <s, s> / <s, y>
    % where s = X_t - X_{t-1}, y = ∇f(X_t) - ∇f(X_{t-1})
    s_norm_sq = ||X_t - X_{t-1}||²;
    sy = <X_t - X_{t-1}, ∇f(X_t) - ∇f(X_{t-1})>;
    
    if sy > ε * s_norm_sq  % Check denominator is positive
        alpha_bb = s_norm_sq / sy;
        alpha_t = clip(alpha_bb, alpha_min, alpha_max);
    else
        alpha_t = Step_sizes(t-1);  % Reuse previous
    end
end

% Simple backtracking if needed
for iter = 1:max_linesearch_iter
    X_trial = X_t - alpha_t * D_t;
    loss_trial = f(X_trial);
    
    if loss_trial < loss_current  % Simple decrease
        break;  % Accept step
    end
    
    alpha_t = backtrack_beta * alpha_t;
    if alpha_t < alpha_min
        alpha_t = alpha_min;
        break;
    end
end

% Store for next iteration
X_prev = X_t;
G_prev = ∇f(X_t);
```

### Why Barzilai-Borwein Works

The BB method approximates the inverse Hessian:
- **Intuition:** `α ≈ 1/λ` where `λ` is a local eigenvalue of Hessian
- **Curvature estimation:** `<s, y> ≈ <s, H*s>` captures local curvature
- **Automatic scaling:** Step size adapts to problem scale
- **No second derivatives:** Uses only gradient differences

### Preconditioner Integration

The BB method works naturally with preconditioners:
- BB estimate: `α_t = ||s||² / <s, y>` remains valid
- Preconditioned search direction: `D_t = M(G_t)` 
- Both the BB step and backtracking work without modification

## Troubleshooting

### Step size too small (slow progress)
- Check if `alpha_min` is too small → increase to 1e-8 or 1e-6
- Verify problem conditioning → may need better initialization
- Monitor `<s, y>` denominator - should be positive and reasonable

### Step size too large (oscillations)
- Reduce `alpha_max` to 0.1 or 0.01
- Check if BB denominator `<s, y>` is too small
- Increase `backtrack_beta` for more aggressive reduction (e.g., 0.3)

### Backtracking happens too often
- This is normal initially - BB adapts quickly
- If persistent: reduce initial `alpha` parameter
- Check if gradient computation is correct

## Implementation Notes

### Changes to `solve_RGD_amplitude.m`

1. **Added parameters** (lines ~20-30):
   - `use_adaptive_stepsize`, `armijo_c1`, `backtrack_beta`
   - `max_linesearch_iter`, `alpha_min`, `alpha_max`

2. **Step size initialization** (line ~140):
   - Use `alpha_max` for first iteration
   - Optimistic increase for subsequent iterations

3. **Armijo line search** (lines ~250-310):
   - Compute directional derivative
   - Backtracking loop with sufficient decrease check
   - Minimum step size acceptance
   - Store step size for next iteration's warm start

4. **Output tracking** (line ~85):
   - Added `Step_sizes` array to output struct

### Backward Compatibility

All changes are backward compatible:
- Default `use_adaptive_stepsize = false` → uses fixed step size
- Fixed step size mode unchanged
- Existing scripts continue to work without modification

## Future Improvements

Potential enhancements:
1. **Alternate BB formulas**: Try `α = <s,y> / <y,y>` (BB2 method)
2. **Nonmonotone line search**: Allow occasional objective increases for faster progress
3. **Adaptive bounds**: Automatically adjust `alpha_min` and `alpha_max` based on progress
4. **Gradient smoothing**: Average multiple gradients for more stable BB estimates

## References

- Barzilai & Borwein (1988), "Two-Point Step Size Gradient Methods", IMA J. Numer. Anal.
- Raydan (1997), "The Barzilai and Borwein Gradient Method for the Large Scale Unconstrained Minimization Problem", SIAM J. Optim.
- Nocedal & Wright, "Numerical Optimization", 2nd Ed., Chapter 3 (Line Search Methods)
- Dai & Fletcher (2005), "Projected Barzilai-Borwein methods for large-scale box-constrained quadratic programming"
