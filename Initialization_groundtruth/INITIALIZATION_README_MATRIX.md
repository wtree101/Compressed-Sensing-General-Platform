# Matrix Initialization Methods

This directory contains unified initialization methods for matrix recovery problems.

## Overview

All initialization functions now follow a consistent interface:

```matlab
[Xl_init, history] = initialization_function(y, operator, d1, d2, T_power, params)
```

### Unified Interface

**Inputs:**
- `y` - Vector of magnitude measurements
- `operator` - Struct with fields:
  - `A` - Forward operator (matrix to measurements)
  - `A_star` - Adjoint operator (measurements to matrix)
- `d1, d2` - Matrix dimensions (d1 × d2)
- `T_power` - Number of iterations (used by power method, ignored by others)
- `params` - Struct with method-specific parameters

**Outputs:**
- `Xl_init` - Initialized matrix (d1 × d2)
- `history` - Struct containing initialization info and convergence history

## Available Methods

### 1. **Power Method** (`initialize_power_method.m`)
Spectral initialization using iterative power method to find principal eigenvector.

**Usage:**
```matlab
params = struct();
params.Xstar = ground_truth;  % Optional
params.projection = @(X, p) project_low_rank(X, p);  % Optional
params.prefunc = @(y) y.^2;   % Optional preprocessing

[X0, hist] = initialize_power_method(y, operator, d1, d2, 50, params);
```

**Features:**
- Iterative refinement (T_power iterations)
- Optional projection at each iteration
- Tracks eigenvalues and convergence
- Best for: Problems requiring spectral initialization with refinement

**History fields:**
- `errors` - Error at each iteration (if Xstar provided)
- `norms` - Norm at each iteration
- `eigenvals` - Rayleigh quotient estimates

### 2. **SVD Method** (`Initialization.m`)
Single-step spectral initialization using truncated SVD.

**Usage:**
```matlab
params = struct();
params.r = 2;           % Target rank
params.m = 400;         % Number of measurements
params.Xstar = ground_truth;  % Optional

[X0, hist] = Initialization(y, operator, d1, d2, 0, params);
```

**Features:**
- Fast single-step initialization
- Exact rank-r truncation
- Returns SVD components
- Best for: Fast spectral initialization, well-conditioned problems

**History fields:**
- `U0, S0, V0` - SVD components
- `singular_values` - All singular values
- `error` - Initial error (if Xstar provided)

### 3. **Random Initialization** (`Initialization_random.m`)
Random low-rank matrix initialization.

**Usage:**
```matlab
params = struct();
params.r = 2;           % Target rank
params.scale = 0.1;     % Scaling factor (optional, default: 0.1)
params.Xstar = ground_truth;  % Optional

[X0, hist] = Initialization_random(y, operator, d1, d2, 0, params);
```

**Features:**
- Creates X0 = U*U' with random U
- Controlled scale
- Symmetric rank-r matrix
- Best for: Baseline comparison, testing convergence from scratch

**History fields:**
- `U0` - Random factor
- `scale` - Scale used
- `rank` - Target rank
- `error` - Initial error (if Xstar provided)

### 4. **Zero Initialization** (`Initialization_zero.m`)
Zero matrix initialization.

**Usage:**
```matlab
params = struct();
params.Xstar = ground_truth;  % Optional

[X0, hist] = Initialization_zero(y, operator, d1, d2, 0, params);
```

**Features:**
- X0 = zeros(d1, d2)
- Minimal overhead
- Best for: Baseline, testing algorithm convergence

**History fields:**
- `error` - Initial error (if Xstar provided)

## Unified Wrapper Function

### `initialize_matrix.m`
Single entry point for all initialization methods.

**Usage:**
```matlab
% Power method
[X0, hist] = initialize_matrix('power', y, operator, d1, d2, 50, params);

% SVD method
[X0, hist] = initialize_matrix('svd', y, operator, d1, d2, 0, params);

% Random initialization
[X0, hist] = initialize_matrix('random', y, operator, d1, d2, 0, params);

% Zero initialization
[X0, hist] = initialize_matrix('zero', y, operator, d1, d2, 0, params);
```

**Benefits:**
- Single function call for any method
- Case-insensitive method names
- Consistent error handling
- Automatic history augmentation

## Comparison Example

```matlab
% Setup
d1 = 20; d2 = 20; r = 2; m = 400;
Xstar = randn(d1, r) * randn(r, d2);
A = randn(m, d1*d2);
operator.A = @(X) A * X(:);
operator.A_star = @(y) reshape(A' * y, [d1, d2]);
y = operator.A(Xstar);

% Test all methods
params.r = r; params.m = m; params.Xstar = Xstar;

methods = {'power', 'svd', 'random', 'zero'};
for i = 1:length(methods)
    [X0, hist] = initialize_matrix(methods{i}, y, operator, d1, d2, 20, params);
    fprintf('%s: Initial error = %.4e\n', methods{i}, hist.error);
end
```

## Migration Guide

### Old Interface → New Interface

**Old `Initialization.m`:**
```matlab
[X0, U0, S0, V0] = Initialization(y, A, d1, d2, r, m)
```

**New Interface:**
```matlab
params.r = r; params.m = m;
[X0, hist] = Initialization(y, operator, d1, d2, 0, params);
U0 = hist.U0; S0 = hist.S0; V0 = hist.V0;
```

**Old `Initialization_random.m`:**
```matlab
[X0, U0] = Initialization_random(y, A, d1, d2, r, m, scale)
```

**New Interface:**
```matlab
params.r = r; params.scale = scale;
[X0, hist] = Initialization_random(y, operator, d1, d2, 0, params);
U0 = hist.U0;
```

## Notes

- All functions are backward compatible through history struct
- Operator struct must have fields `A` and `A_star`
- Ground truth `Xstar` is always optional
- T_power is only used by power method
- History struct may contain different fields depending on method

## See Also

- Vector initialization: `init_random_vector.m`, `init_zero_vector.m`
- Ground truth generation: `groundtruth.m`
