# Revision: Updated initialize_tensor_nuclear_norm Output Format

## Date: November 10, 2025

## Summary
Updated `initialize_tensor_nuclear_norm.m` to match the unified initialization interface used throughout the codebase, ensuring compatibility with `onetrial_Mat` and other framework components.

## Changes Made

### 1. Function Signature

**Old Signature**:
```matlab
function [X0,U0,history] = initialize_tensor_nuclear_norm(operator, y, d, varargin)
```

**New Signature** (unified interface):
```matlab
function [X0, U0, history] = initialize_tensor_nuclear_norm(y, operator, d1, d2, params)
```

**Key Changes**:
- ✓ Reordered parameters: `y` comes first, then `operator`
- ✓ Split dimension: now uses `d1, d2` (supports non-square matrices conceptually)
- ✓ Changed from name-value pairs (`varargin`) to params struct
- ✓ Matches signature of other initialization functions like `initialize_tensor_lift_tucker_spectral`

### 2. Output Variables

**Previously**:
- Only `X0` was properly set
- `U0` and `history` were not set

**Now**:
- `X0`: Initialized matrix (d×d symmetric)
- `U0`: Empty array `[]` (for compatibility)
- `history`: Struct with convergence information:
  - `.method`: 'tensor_nuclear_norm'
  - `.iterations`: actual iterations run
  - `.max_iter`: maximum iterations parameter
  - `.obj_values`: objective function history
  - `.constraint_violations`: constraint violation history
  - `.primal_residuals`: ADMM primal residual history
  - `.dual_residuals`: ADMM dual residual history
  - `.rank`: rank of recovered matrix
  - `.converged`: convergence flag

### 3. Parameter Handling

**Old Format** (name-value pairs):
```matlab
X0 = initialize_tensor_nuclear_norm(operator, y, 20, ...
    'rank', 2, 'max_iter', 10, 'lambda', 1.0, 'rho', 0.1);
```

**New Format** (params struct):
```matlab
params = struct();
params.rank = 2;          % or params.r = 2 (both supported)
params.max_iter = 10;
params.lambda = 1.0;
params.rho = 0.1;
params.verbose = 1;
params.normalize = true;

[X0, U0, history] = initialize_tensor_nuclear_norm(y, operator, 20, 20, params);
```

**Supported Parameters**:
- `r` or `rank`: Target rank for extraction (default: auto-estimated)
- `max_iter`: Number of iterations (default: 10)
- `lambda`: Regularization parameter (default: 1.0)
- `rho`: ADMM penalty parameter (default: 0.1)
- `verbose`: Verbosity level (default: 0)
- `normalize`: Normalize output (default: true)

### 4. Helper Function

Added `get_param` helper function:
```matlab
function value = get_param(params, field, default)
    % Get parameter from struct with default value
    if isfield(params, field)
        value = params.(field);
    else
        value = default;
    end
end
```

### 5. Enhanced Operator Handling

The function now automatically extracts `A_cells` from the operator if not present:
```matlab
if ~isfield(operator, 'A_cells')
    % Extract measurement matrices from operator.A
    % Create cell array A_cells{1}, ..., A_cells{m}
    % Symmetrize each matrix
end
```

## Updated Files

### 1. `Initialization_groundtruth/initialize_tensor_nuclear_norm.m`
- ✓ Changed function signature to unified interface
- ✓ Added proper `U0` and `history` outputs
- ✓ Changed from varargin to params struct
- ✓ Added automatic A_cells extraction
- ✓ Added get_param helper function
- ✓ Updated documentation

### 2. `test/test_nuclear_init_plus_refine.m`
- ✓ Updated function call to use new signature
- ✓ Changed from name-value pairs to params struct
- ✓ Now captures all three outputs: `[X0, ~, init_history]`

### 3. `matrix_recovery/Phasediagram_nuclear_init.m`
- ✓ Already compatible (passes function handle to onetrial_Mat)
- ✓ No changes needed - automatic compatibility through unified interface

## Compatibility

### With onetrial_Mat
The unified signature matches the expected interface:
```matlab
% In onetrial_Mat.m (line 75):
[Xl, Ul, ~] = params.init(y, operator, d1, d2, init_params);
```

Now `initialize_tensor_nuclear_norm` can be used seamlessly:
```matlab
params.init = @initialize_tensor_nuclear_norm;
[output, is_success] = onetrial_Mat(params);
```

### With Other Initialization Functions
The signature now matches:
- `initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params)`
- `initialize_tensor_lift(y, operator, d1, d2, params)`
- `initialize_tensor_lift_efficient(y, operator, d1, d2, params)`

## Testing

To test the updated function:

```matlab
% Test 1: Direct call with new signature
params = struct('rank', 2, 'max_iter', 10, 'verbose', 1);
[X0, U0, history] = initialize_tensor_nuclear_norm(y, operator, 20, 20, params);

% Test 2: Via test script
cd /Users/wutong/Documents/MATLAB/GeneralPlatform
test_nuclear_init_plus_refine

% Test 3: In phase diagram (expensive, optional)
% Phasediagram_nuclear_init
```

## Benefits

1. **Consistency**: Matches the unified initialization interface used throughout the codebase
2. **Compatibility**: Works seamlessly with `onetrial_Mat`, `multipletrial`, and phase diagram scripts
3. **Information**: Now returns detailed convergence history
4. **Flexibility**: Supports both `rank` and `r` parameter names
5. **Robustness**: Automatically extracts A_cells if not provided
6. **Maintainability**: Easier to extend and modify with params struct

## Migration Notes

If you have existing code calling the old signature:

**Old Code**:
```matlab
X0 = initialize_tensor_nuclear_norm(operator, y, 20, 'rank', 2, 'max_iter', 10);
```

**New Code**:
```matlab
params = struct('rank', 2, 'max_iter', 10);
[X0, U0, history] = initialize_tensor_nuclear_norm(y, operator, 20, 20, params);
```

## Implementation Details

The function internally:
1. Validates d1 == d2 (requires symmetric matrices)
2. Extracts parameters from params struct with defaults
3. Estimates rank if not provided
4. Extracts/creates A_cells if needed
5. Calls `solve_tensor_nuclear_norm` for limited iterations
6. Normalizes and symmetrizes output
7. Returns complete history struct

All ADMM convergence information is preserved in the history output for analysis.
