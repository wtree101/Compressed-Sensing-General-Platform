# Unified Initialization Signature Guide

## Overview
All initialization functions now follow the same unified signature as `initialize_power_method`:

```matlab
[X0, U0, history] = initialization_func(y, operator, d1, d2, params)
```

## Function Signature

### Inputs
- `y` - Measurement vector (m x 1)
- `operator` - Struct with fields:
  - `.A`: Forward operator @(X) A*X(:)
  - `.A_star`: Adjoint operator @(y) reshape(A'*y, [d1,d2])
- `d1` - Matrix row dimension
- `d2` - Matrix column dimension  
- `params` - Struct with method-specific fields (see below)

### Outputs
- `X0` - Initialized matrix (d1 x d2)
- `U0` - Left factor for factorization compatibility (d1 x r)
- `history` - Struct with convergence/diagnostic information

## Available Initialization Methods

### 1. Spectral Initialization (SVD)
```matlab
[X0, U0, history] = Initialization(y, operator, d1, d2, params)
```

**Required params fields:**
- `r` - Target rank
- `m` - Number of measurements

**Optional params fields:**
- `Xstar` - Ground truth for error tracking

**Returns in history:**
- `method` - 'SVD'
- `U0` - Left singular vectors (d1 x r)
- `S0` - Singular values (r x r)
- `V0` - Right singular vectors (d2 x r)
- `singular_values` - All singular values
- `error` - Initial error (if Xstar provided)

### 2. Random Initialization
```matlab
[X0, U0, history] = Initialization_random(y, operator, d1, d2, params)
```

**Required params fields:**
- `r` - Target rank

**Optional params fields:**
- `scale` - Scaling factor (default: 0.1)
- `Xstar` - Ground truth for error tracking

**Returns in history:**
- `method` - 'Random'
- `U0` - Random factor (d1 x r)
- `scale` - Used scaling factor
- `rank` - Target rank
- `error` - Initial error (if Xstar provided)

### 3. Power Method Initialization
```matlab
[X0, U0, history] = initialize_power_method(y, operator, d1, d2, params)
```

**Required params fields:**
- None (all optional with defaults)

**Optional params fields:**
- `T_power` - Number of power iterations (default: 20)
- `projection` - Projection function handle @(X) proj(X)
- `prefunc` - Preprocessing function for measurements @(y) f(y) (default: @(y) y.^2)
- `Xstar` - Ground truth for error tracking

**Returns in history:**
- `errors` - Aligned error at each iteration (if Xstar provided)
- `norms` - Vector norms at each iteration
- `iterations` - Number of power iterations performed

## Usage Examples

### Example 1: Using SVD Initialization
```matlab
params = struct();
params.r = 5;
params.m = 1000;
params.Xstar = Xstar;  % Optional: for error tracking

[X0, U0, history] = Initialization(y, operator, d1, d2, params);
```

### Example 2: Using Random Initialization with Custom Scale
```matlab
params = struct();
params.r = 5;
params.scale = 0.05;  % Small initial scale
params.Xstar = Xstar;

[X0, U0, history] = Initialization_random(y, operator, d1, d2, params);
```

### Example 3: Using Power Method with Preprocessing
```matlab
% Define Z deflation
Z = ones([d1, d2]) / sqrt(d1 * d2) * 0.5;
AZ = operator.A(Z) / sqrt(m);

params = struct();
params.T_power = 50;  % More iterations
params.projection = @(X) rank_projection(X, 3);  % Rank-3 projection
params.prefunc = @(y) y.^2 - abs(AZ).^2;  % Deflate Z component
params.Xstar = Xstar;

[X0, U0, history] = initialize_power_method(y, operator, d1, d2, params);
```

### Example 4: Using in onetrial_Mat.m
```matlab
% In your main script, set up params
params.init = @Initialization;  % or @Initialization_random, @initialize_power_method
params.r = 5;
params.m = 1000;
params.Xstar = Xstar;
params.T_power = 30;  % For power method
params.scale = 0.1;   % For random init

% onetrial_Mat.m will automatically call with unified signature
[output, is_success] = onetrial_Mat(params);
```

### Example 5: Using set_init.m
```matlab
% Get initialization function handle
[init_name, init_handle] = set_init(init_flag);
% init_flag: 0=SVD, 1=Random, 2=Power Method

% Prepare params
params.r = 5;
params.m = 1000;
params.Xstar = Xstar;
params.T_power = 20;  % Only used by power method
params.scale = 0.1;   # Only used by random init

% Call initialization
[X0, U0, history] = init_handle(y, operator, d1, d2, params);
```

## Migration from Old Interface

### Old Interface (Before):
```matlab
% Different signatures for different methods
[X0, U0, S0, V0] = Initialization(y, A, d1, d2, r, m);
[X0, U0] = Initialization_random(y, A, d1, d2, r, m, scale);
[X0, U0] = initialize_power_method(y, operator, d1, d2, T_power, init_params);
```

### New Interface (After):
```matlab
% All methods use same signature
params = struct('r', r, 'm', m);
[X0, U0, history] = Initialization(y, operator, d1, d2, params);

params = struct('r', r, 'scale', scale);
[X0, U0, history] = Initialization_random(y, operator, d1, d2, params);

params = struct('T_power', T_power, 'Xstar', Xstar);
[X0, U0, history] = initialize_power_method(y, operator, d1, d2, params);
```

## Key Benefits

1. **Uniform Interface**: All initialization methods have the same call signature
2. **Flexible Parameters**: Use struct to pass any combination of parameters
3. **Rich History**: All methods return diagnostic information in `history`
4. **Easy Swapping**: Change initialization by swapping function handle only
5. **Error Tracking**: Consistent error tracking when `Xstar` is provided
6. **Future-Proof**: Easy to add new initialization methods

## Backward Compatibility Notes

⚠️ **Breaking Changes:**
- All initialization functions now require 5 arguments (y, operator, d1, d2, params)
- Old 6-7 argument signatures are no longer supported
- `operator` must be a struct with `.A` and `.A_star` fields
- All method-specific parameters (r, m, scale, T_power) are passed via `params` struct

## Testing

See these test files for examples:
- `test/test_power_method.m` - Comprehensive power method testing
- `test/check_error_computation.m` - Error tracking verification

Run tests:
```matlab
cd test
test_power_method
check_error_computation
```
