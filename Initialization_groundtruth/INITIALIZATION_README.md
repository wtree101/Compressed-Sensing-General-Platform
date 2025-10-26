# Matrix Initialization Functions - Unified Interface

This document describes the unified interface for matrix initialization functions used in phase retrieval and low-rank matrix recovery.

## Overview

All initialization functions now share a **consistent signature**:

```matlab
[X0, U0] = init_function(y, operator, d1, d2, r, m, scale)
```

## Common Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | vector (m×1) | Measurement vector |
| `operator` | struct | Operator with fields `.A` (forward) and `.A_star` (adjoint) |
| `d1` | integer | Matrix row dimension |
| `d2` | integer | Matrix column dimension |
| `r` | integer | Target rank for initialization |
| `m` | integer | Number of measurements |
| `scale` | scalar (optional) | Scaling factor (default varies by method) |

## Common Output Parameters

| Output | Type | Description |
|--------|------|-------------|
| `X0` | matrix (d1×d2) | Initialized matrix |
| `U0` | matrix (d1×r) | Factor matrix such that X0 ≈ U0*U0' |
| `history` | struct | Convergence history (non-empty only for power method) |

### History Structure (Power Method Only)

For `initialize_power_method`, the history struct contains:
- `.eigenvals` - Eigenvalue estimates at each iteration (T×1)
- `.norms` - Vector norms at each iteration (T×1)  
- `.iterations` - Number of power iterations performed

For other methods, `history` is an empty struct.

## Available Initialization Methods

### 1. Spectral Initialization (Truncated SVD)

**File**: `Initialization.m`

**Method**: Computes X0 = A'(y)/sqrt(m), then applies rank-r SVD truncation

**Usage**:
```matlab
[X0, U0, history] = Initialization(y, operator, d1, d2, r, m, scale);
% scale: default = 1.0
% history: empty struct
```

**Best for**: Standard phase retrieval, good signal-to-noise ratio

---

### 2. Random Initialization

**File**: `Initialization_random.m`

**Method**: Generates random U0 ∈ R^(d1×r), then X0 = U0*U0'

**Usage**:
```matlab
[X0, U0, history] = Initialization_random(y, operator, d1, d2, r, m, scale);
% scale: default = 0.1
% history: empty struct
```

**Best for**: Multiple random restarts, baseline comparison

---

### 3. Power Method Initialization

**File**: `initialize_power_method.m`

**Method**: Uses power iterations to find principal eigenvector of Y = sum(y_i^2 * a_i*a_i')

**Usage**:
```matlab
[X0, U0, history] = initialize_power_method(y, operator, d1, d2, r, m, scale);
% scale: default = 1.0
% Uses 50 power iterations (hardcoded)
% history: struct with .eigenvals, .norms, .iterations
```

**Best for**: Large-scale problems, better spectral initialization

**History tracking**: This method records convergence information:
- `history.eigenvals` - Rayleigh quotient at each iteration
- `history.norms` - Vector norm before normalization
- `history.iterations` - Total number of iterations (50)

---

### 4. Unified Interface

**File**: `init_matrix.m`

**Method**: Wrapper that calls appropriate initialization based on string argument

**Usage**:
```matlab
% Spectral initialization
[X0, U0] = init_matrix(y, operator, d1, d2, r, m, 'spectral');

% Random initialization
[X0, U0] = init_matrix(y, operator, d1, d2, r, m, 'random', 0.1);

% Power method
[X0, U0] = init_matrix(y, operator, d1, d2, r, m, 'power');

% Zero initialization
[X0, U0] = init_matrix(y, operator, d1, d2, r, m, 'zero');
```

**Supported method strings**:
- `'spectral'`, `'svd'`, `'truncated_svd'` → Spectral initialization
- `'random'`, `'rand'`, `'randn'` → Random initialization
- `'power'`, `'power_method'`, `'spectral_power'` → Power method
- `'zero'`, `'zeros'` → Zero initialization

---

## Operator Structure

All functions expect an `operator` struct with:

```matlab
operator.A = @(X) A * X(:);                    % Forward: matrix → measurements
operator.A_star = @(y) reshape(A' * y, [d1, d2]); % Adjoint: measurements → matrix
```

**Legacy support**: `Initialization.m` also supports passing the matrix `A` directly.

---

## Comparison of Methods

| Method | Pros | Cons | Typical Scale |
|--------|------|------|---------------|
| **Spectral** | Fast, theoretically grounded | Requires good measurements | 1.0 |
| **Random** | Simple, unbiased | May need multiple trials | 0.1 |
| **Power** | Better spectral estimate | More expensive (50 iters) | 1.0 |
| **Zero** | Baseline | No information from data | N/A |

---

## Usage Examples

### Example 1: Basic Usage
```matlab
% Setup problem
d1 = 20; d2 = 20; r = 2; m = 100;
A = randn(m, d1*d2);
y = abs(A * xstar(:));

% Create operator
operator.A = @(X) A * X(:);
operator.A_star = @(y) reshape(A' * y, [d1, d2]);

% Initialize with spectral method
[X0, U0] = Initialization(y, operator, d1, d2, r, m);
```

### Example 2: Using Unified Interface
```matlab
% Try different methods
methods = {'spectral', 'random', 'power'};
results = cell(length(methods), 1);

for i = 1:length(methods)
    [X0, U0, history] = init_matrix(y, operator, d1, d2, r, m, methods{i});
    results{i} = struct('X0', X0, 'U0', U0, 'method', methods{i}, 'history', history);
    
    % Only power method has non-empty history
    if strcmpi(methods{i}, 'power') && ~isempty(fieldnames(history))
        fprintf('%s: Final eigenval = %.4e\n', methods{i}, history.eigenvals(end));
    end
end
```

### Example 3: Using History from Power Method
```matlab
% Initialize with power method and analyze convergence
[X0, U0, history] = init_matrix(y, operator, d1, d2, r, m, 'power');

% Plot convergence
figure;
subplot(2,1,1);
plot(1:history.iterations, history.eigenvals, 'b-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Eigenvalue Estimate');
title('Power Method Convergence: Eigenvalue');
grid on;

subplot(2,1,2);
plot(1:history.iterations, history.norms, 'r-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Vector Norm');
title('Power Method Convergence: Norm');
grid on;
```

### Example 4: Using ~ to Ignore History
```matlab
% If you don't need history, use ~ to suppress it
[X0, U0, ~] = Initialization(y, operator, d1, d2, r, m);
[X0, U0, ~] = Initialization_random(y, operator, d1, d2, r, m, 0.1);

% For power method, you might still want history
[X0, U0, history] = initialize_power_method(y, operator, d1, d2, r, m);
```

### Example 5: In Phase Diagram Script
```matlab
% In Phasediagram_tensor.m or similar
trial_params.init_method = 'spectral';  % or 'random', 'power'
trial_params.init_scale = 1.0;

% In onetrial function
[X0, U0] = init_matrix(y, operator, d1, d2, r, m, ...
                       params.init_method, params.init_scale);
```

---

## Migration Guide

### Old Code:
```matlab
% Different signatures for each method
[X0, U0, S0, V0] = Initialization(y, A, d1, d2, r, m);
[X0, U0] = Initialization_random(y, A, d1, d2, r, m, scale);
[X0, history] = initialize_power_method(y, operator, d1, d2, T_power, params);
```

### New Code:
```matlab
% Unified signature for all methods
[X0, U0, history] = Initialization(y, operator, d1, d2, r, m, scale);
[X0, U0, history] = Initialization_random(y, operator, d1, d2, r, m, scale);
[X0, U0, history] = initialize_power_method(y, operator, d1, d2, r, m, scale);

% Or use unified interface
[X0, U0, history] = init_matrix(y, operator, d1, d2, r, m, 'spectral');

% Use ~ to ignore history if not needed
[X0, U0, ~] = init_matrix(y, operator, d1, d2, r, m, 'random');
```

---

## Future Enhancements

Potential additions:
- Weighted spectral initialization
- Orthogonal matching pursuit initialization
- Wirtinger flow initialization
- Amplitude-based initialization for specific problems

To add a new method:
1. Create function with signature: `[X0, U0] = new_init(y, operator, d1, d2, r, m, scale)`
2. Add case to `init_matrix.m` switch statement
3. Update this README

---

## Notes

- All methods return both `X0` (matrix) and `U0` (factor) for flexibility
- The `scale` parameter is optional and has method-dependent defaults
- Power method uses 50 iterations (can be made configurable)
- For symmetric problems, ensure `d1 == d2`
- The `r` parameter defines target rank for factorization
