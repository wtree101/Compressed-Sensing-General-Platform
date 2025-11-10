# Revision Notes: Updated Input Format for solve_PGD_amplitude

## Date: November 10, 2025

## Summary
Updated the input format in test scripts and phase diagram to match the correct signature of `solve_PGD_amplitude`.

## Changes Made

### 1. File: `test/test_nuclear_init_plus_refine.m`

**Issue**: Incorrect function signature when calling `solve_PGD_amplitude`

**Old Format** (incorrect):
```matlab
[X_final, output] = solve_PGD_amplitude(operator, y, r, d, d, nonlinear_func, ...
    'T', refine_max_iter, ...
    'mu', mu, ...
    'Xstar', Xstar, ...
    'X0', X0, ...
    'verbose', verbose_refine);
```

**New Format** (correct):
```matlab
% Prepare parameters struct
refine_params = struct();
refine_params.T = refine_max_iter;
refine_params.mu = mu;
refine_params.Xstar = Xstar;
refine_params.projection = @(X) project_rank_r(X, r);  % Rank-r projection

% Call solver with correct signature: solve_PGD_amplitude(Xl, ~, y, operator, d1, d2, ~, m, params)
[output, X_final] = solve_PGD_amplitude(X0, [], y, operator, d, d, [], m, refine_params);
```

**Changes**:
- ✓ Fixed function signature to match: `solve_PGD_amplitude(Xl, ~, y, operator, d1, d2, ~, m, params)`
- ✓ All optional parameters now passed via `params` struct
- ✓ Added required `projection` function handle in params
- ✓ Corrected output order: `[output, Xl]` instead of `[Xl, output]`
- ✓ Added helper function `project_rank_r` at end of file

### 2. File: `matrix_recovery/Phasediagram_nuclear_init.m`

**Issue**: Missing required parameters for `solve_PGD_amplitude`

**Changes**:
```matlab
trial_params.mu = mu;  % Step size for solve_PGD_amplitude
trial_params.projection = @(X) project_rank_r(X, r);  % Rank-r projection
```

**Added**:
- ✓ `mu` parameter (step size)
- ✓ `projection` function handle (required by solver)
- ✓ Helper function `project_rank_r` at end of file

### 3. Helper Function Added to Both Files

```matlab
function X_proj = project_rank_r(X, r)
    % Project matrix X onto the set of symmetric rank-r matrices
    % Input:
    %   X - Input matrix (d x d)
    %   r - Target rank
    % Output:
    %   X_proj - Projected symmetric rank-r matrix
    
    % Symmetrize
    X_sym = (X + X') / 2;
    
    % SVD and truncate to rank r
    [U, S, ~] = svd(X_sym);
    U_r = U(:, 1:r);
    S_r = S(1:r, 1:r);
    
    % Reconstruct symmetric rank-r matrix
    X_proj = U_r * S_r * U_r';
end
```

## Reference: Correct Signature

From `solver/solve_PGD_amplitude.m`:
```matlab
function [output, Xl] = solve_PGD_amplitude(Xl, ~, y, operator, d1, d2, ~, m, params)
```

**Parameters**:
1. `Xl` - Initial matrix (d1 x d2)
2. `~` - Unused (for compatibility with other solvers like RGD)
3. `y` - Measurement vector (magnitudes)
4. `operator` - Struct with `A` (forward) and `A_star` (adjoint) operators
5. `d1, d2` - Matrix dimensions
6. `~` - Unused rank parameter (now in params)
7. `m` - Number of measurements
8. `params` - Parameter structure containing:
   - `T`: number of iterations
   - `mu` (or `eta`): step size
   - `projection`: projection function handle (REQUIRED)
   - `Xstar`: ground truth (optional, for error tracking)

**Returns**:
1. `output` - Struct with `Error_Stand` and `Error_function`
2. `Xl` - Final solution matrix

## Testing

Run these commands to test the updated files:

```matlab
% Test initialization + refinement pipeline
cd /Users/wutong/Documents/MATLAB/GeneralPlatform
test_nuclear_init_plus_refine

% Phase diagram (if needed, but this is expensive)
% Phasediagram_nuclear_init
```

## Notes

- The phase diagram script works through `onetrial_Mat.m`, which correctly passes all parameters to the solver
- The `projection` parameter is CRITICAL - the solver will error without it
- Both test script and phase diagram now use consistent parameter passing
