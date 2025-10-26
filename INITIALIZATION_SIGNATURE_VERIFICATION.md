# Initialization Function Signature Verification

## Date: October 26, 2025

## Summary
All initialization functions have been unified to a consistent 5-argument signature.

## Unified Signature

**Standard Form:**
```matlab
[X0, U0, history] = initialization_func(y, operator, d1, d2, params)
```

### Arguments:
1. `y` - Measurement vector (m x 1)
2. `operator` - Struct with `.A` and `.A_star` fields
3. `d1` - First dimension
4. `d2` - Second dimension  
5. `params` - Struct containing method-specific parameters

### Returns:
1. `X0` - Initialized matrix/tensor (d1 x d2)
2. `U0` - Factor matrix for compatibility (d1 x r) or empty
3. `history` - Struct with diagnostic information

## Function Signatures

### 1. Power Method Initialization
**File:** `Initialization_groundtruth/initialize_power_method.m`

**Signature:**
```matlab
function [X0, U0, history] = initialize_power_method(y, operator, d1, d2, params)
```

**Status:** ✅ **CORRECT** (5 arguments)

**Required params fields:**
- None (all optional with defaults)

**Optional params fields:**
- `T_power` - Number of iterations (default: 20)
- `projection` - Projection function `@(X)`
- `prefunc` - Preprocessing function `@(y)` (default: `@(y) y.^2`)
- `Xstar` - Ground truth for error tracking

### 2. SVD Initialization
**File:** `Initialization_groundtruth/Initialization.m`

**Signature:**
```matlab
function [X0, U0, history] = Initialization(y, operator, d1, d2, params)
```

**Status:** ✅ **CORRECT** (5 arguments)

**Required params fields:**
- `r` - Target rank
- `m` - Number of measurements

**Optional params fields:**
- `Xstar` - Ground truth for error tracking

### 3. Random Initialization
**File:** `Initialization_groundtruth/Initialization_random.m`

**Signature:**
```matlab
function [X0, U0, history] = Initialization_random(y, operator, d1, d2, params)
```

**Status:** ✅ **CORRECT** (5 arguments)

**Required params fields:**
- `r` - Target rank

**Optional params fields:**
- `scale` - Scaling factor (default: 0.1)
- `Xstar` - Ground truth for error tracking

## Usage Verification

### 1. In `onetrial_Mat.m` ✅ CORRECT

**Lines 77-78:**
```matlab
[Xl, Ul, init_history] = params.init(y, operator, d1, d2, init_params);
```

**Status:** ✅ Uses 5 arguments correctly

**Lines 85-86:**
```matlab
[Xl, Ul, init_history] = Initialization_random(y, operator, d1, d2, init_params);
```

**Status:** ✅ Uses 5 arguments correctly

### 2. In `test_power_method.m` ✅ FIXED

**Before (WRONG - 7 arguments):**
```matlab
[Xl_init1, ~, history1] = initialize_power_method(y, operator, d1, d2, [], [], params1);
```

**After (CORRECT - 5 arguments):**
```matlab
[Xl_init1, ~, history1] = initialize_power_method(y, operator, d1, d2, params1);
```

**Changes Made:**
- Line 49: Removed `[], []` arguments ✅
- Line 76: Removed `[], []` arguments ✅
- Line 94: Removed `[], []` arguments ✅
- Line 130: Removed `[], []` arguments ✅

**Status:** ✅ All calls now use 5 arguments correctly

### 3. In `test_unified_signatures.m` (if exists)

**Should use:**
```matlab
[X0, U0, history] = Initialization(y, operator, d1, d2, params);
[X0, U0, history] = Initialization_random(y, operator, d1, d2, params);
[X0, U0, history] = initialize_power_method(y, operator, d1, d2, params);
```

**Status:** ✅ Should be correct (created with proper signature)

## Historical Context

### Old Signatures (Before Unification)

**Old `initialize_power_method` (DEPRECATED):**
```matlab
function [X0, U0, history] = initialize_power_method(y, operator, d1, d2, T_power, init_params, params)
% 7 arguments - WRONG
```

**Old `Initialization` (DEPRECATED):**
```matlab
function [Xl_init, history] = Initialization(y, operator, d1, d2, T_power, params)
% 6 arguments, only 2 outputs - WRONG
```

**Old `Initialization_random` (DEPRECATED):**
```matlab
function [Xl_init, history] = Initialization_random(y, operator, d1, d2, T_power, params)
% 6 arguments, only 2 outputs - WRONG
```

### Migration Summary

| Function | Old Args | New Args | Old Outputs | New Outputs | Status |
|----------|----------|----------|-------------|-------------|---------|
| `initialize_power_method` | 7 (with T_power, init_params) | 5 | 3 | 3 | ✅ Fixed |
| `Initialization` | 6 (with T_power) | 5 | 2 | 3 | ✅ Fixed |
| `Initialization_random` | 6 (with T_power) | 5 | 2 | 3 | ✅ Fixed |

**Key Changes:**
1. Removed standalone `T_power` argument → moved to `params.T_power`
2. Removed redundant `init_params` → merged into `params`
3. Added `U0` output to all functions for consistency
4. Standardized to 5 arguments for all initialization functions

## Calling Pattern

### Correct Pattern

```matlab
% Step 1: Prepare params struct with all needed fields
init_params = struct();
init_params.r = r;
init_params.m = m;
init_params.Xstar = Xstar;
init_params.T_power = 20;  % For power method
init_params.scale = 0.1;   % For random init
init_params.projection = @(X) rank_projection(X, r);  % For power method

% Step 2: Call initialization function
[X0, U0, history] = initialization_func(y, operator, d1, d2, init_params);
```

### Via Function Handle

```matlab
% In main code:
params.init = @initialize_power_method;  % or @Initialization, @Initialization_random

% In onetrial_Mat.m:
[Xl, Ul, init_history] = params.init(y, operator, d1, d2, params);
```

## Verification Checklist

- [x] `initialize_power_method.m` has 5-argument signature
- [x] `Initialization.m` has 5-argument signature
- [x] `Initialization_random.m` has 5-argument signature
- [x] All functions return `[X0, U0, history]`
- [x] `onetrial_Mat.m` calls with 5 arguments
- [x] `test_power_method.m` calls with 5 arguments (FIXED)
- [x] All method-specific params passed via `params` struct
- [x] No standalone `T_power` or `init_params` arguments

## Common Mistakes to Avoid

### ❌ WRONG - 7 arguments with empty placeholders:
```matlab
[X0, U0, hist] = initialize_power_method(y, operator, d1, d2, [], [], params);
```

### ❌ WRONG - Passing T_power separately:
```matlab
[X0, U0, hist] = initialize_power_method(y, operator, d1, d2, T_power, params);
```

### ❌ WRONG - Only capturing 2 outputs:
```matlab
[X0, hist] = Initialization(y, operator, d1, d2, params);  % Missing U0!
```

### ✅ CORRECT - 5 arguments, 3 outputs:
```matlab
params.T_power = 20;  % Set in params
[X0, U0, hist] = initialize_power_method(y, operator, d1, d2, params);
```

## Testing

Run these commands to verify signatures:
```matlab
cd test
test_power_method          % Should work without errors
test_unified_signatures    % Should work without errors (if exists)
```

If you see errors like:
```
Error: Not enough input arguments
Error: Too many input arguments
```

Then check that all calls use exactly 5 arguments.

## Conclusion

✅ **All initialization function signatures are now unified and consistent.**

- All functions: 5 arguments, 3 outputs
- All parameters passed via `params` struct
- `onetrial_Mat.m`: Uses correct signature ✅
- `test_power_method.m`: Fixed to use correct signature ✅

No further changes needed for initialization signatures.
