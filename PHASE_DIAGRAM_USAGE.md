# Phase Diagram Script Usage Guide

## Overview

The `Phasediagram_tensor.m` script provides a unified framework for testing different initialization methods for low-rank matrix recovery via phase retrieval.

## Quick Start

1. Open `Phasediagram_tensor.m`
2. Choose your initialization method (lines 31-49)
3. Set `T_power` (number of initialization iterations)
4. Run the script

## Initialization Methods

### 1. Tucker Spectral (Default)
```matlab
init_method = @initialize_tensor_lift_tucker_spectral;
alg_name = 'MatsubGD_tensorSpectralinit';
T_power = 20;  % Number of RGD iterations on Tucker manifold
```

**Best for**: High-quality initialization with spectral method + Riemannian optimization

### 2. Tensor Nuclear Norm (TNN)
```matlab
init_method = @initialize_tensor_nuclear_norm;
alg_name = 'MatsubGD_TNNinit';
T_power = 10;  % Number of ADMM iterations
```

**Best for**: Convex relaxation approach, robust initialization
**Note**: Automatically falls back to random if solver encounters numerical issues

### 3. Power Method
```matlab
init_method = @initialize_power_method;
alg_name = 'MatsubGD_powerinit';
T_power = 20;  % Number of power iterations
```

**Best for**: Fast, simple initialization

### 4. Basic Tensor Lift
```matlab
init_method = @initialize_tensor_lift;
alg_name = 'MatsubGD_tensorinit';
T_power = 10;  % Number of iterations
```

**Best for**: Standard tensor lifting approach

## Key Parameters

### General Parameters
- `d1, d2`: Matrix dimensions (must be equal for symmetric matrices)
- `r_grid`: Range of ranks to test (e.g., `1:1:5`)
- `trial_num`: Number of trials per (rank, measurement) pair
- `problem_flag`: Problem type (2 = phase retrieval)

### Initialization Parameters
- `T_power`: **Number of initialization steps/iterations** (GENERAL PARAMETER)
  - For Tucker spectral: RGD iterations
  - For Tensor nuclear norm: ADMM iterations
  - For Power method: Power iterations
  
### Refinement Parameters
- `T`: Number of local refinement iterations (default: 200)
- `mu`: Step size for gradient descent (default: 0.1)

### Additional TNN Parameters
These are automatically set but can be customized:
- `lambda`: Regularization parameter (default: 1.0)
- `rho`: ADMM penalty parameter (default: 0.1)
- `normalize`: Normalize output (default: true)

## Workflow

```
┌─────────────────────────────────────┐
│  Generate Ground Truth (rank r)     │
│  X_true = UU^T (symmetric)          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Create Measurements                │
│  y_i = |⟨A_i, X⟩| / √m              │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Initialization (T_power steps)     │
│  • Tucker Spectral: RGD on manifold │
│  • TNN: ADMM minimization           │
│  • Power: Power iterations          │
│  → Produces X0                      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Local Refinement (T steps)         │
│  Gradient descent on rank-r manifold│
│  → Produces X_final                 │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Evaluate Success                   │
│  error = ||X_final - X_true|| / ||X||│
│  success if error < 1e-2            │
└─────────────────────────────────────┘
```

## Example Configurations

### Fast Testing (Tucker Spectral)
```matlab
init_method = @initialize_tensor_lift_tucker_spectral;
T_power = 10;   % Quick initialization
T = 100;        # Fewer refinement steps
trial_num = 1;  # Single trial
r_grid = 1:1:2; # Small rank range
```

### Production Run (TNN)
```matlab
init_method = @initialize_tensor_nuclear_norm;
T_power = 10;   # TNN iterations
T = 200;        # Full refinement
trial_num = 5;  # Multiple trials
r_grid = 1:1:5; # Full rank range
```

### Comparison Study
Run the script multiple times with different `init_method` settings:
1. Tucker Spectral (T_power=20)
2. TNN (T_power=10)
3. Power Method (T_power=20)

Compare results in saved `.mat` files.

## Output Files

Results are saved in:
```
data_f/err_data_d1_20_d2_20_rmax_5_kappa_2_prob_2_alg_<alg_name>/
  ├── mu_0.1000/
  │   ├── r_1.mat
  │   ├── r_2.mat
  │   └── ...
  └── ...
```

Each file contains:
- `results.success_count`: Successes per measurement count
- `results.avg_error`: Average reconstruction error
- `results.trial_errors`: Full error history

## Comparison: Phasediagram_nuclear_init.m vs Phasediagram_tensor.m

### Phasediagram_nuclear_init.m
- **Focused**: Only supports TNN initialization
- **Simplified**: Cleaner code for single initialization method
- **Parameters**: TNN-specific parameters at top

### Phasediagram_tensor.m
- **Flexible**: Supports all initialization methods
- **General**: Switch methods by uncommenting lines
- **Parameters**: T_power works for all methods

**Recommendation**: 
- Use `Phasediagram_tensor.m` for comparing multiple initialization methods
- Use `Phasediagram_nuclear_init.m` if you only need TNN initialization

## Troubleshooting

### TNN returns NaN/Inf
The initialization function automatically falls back to random initialization. Check:
- Measurement quality (are values reasonable?)
- Parameters (try smaller `rho`, e.g., 0.01)
- `T_power` (try fewer iterations, e.g., 5)

### Slow convergence
- Increase `T_power` for better initialization
- Decrease `mu` for more stable refinement
- Try different initialization method

### Memory issues
- Reduce `d1, d2` (matrix size)
- Reduce `r_max` (maximum rank)
- Use fewer measurement points in grid

## Notes

1. **T_power is the universal parameter**: All initialization methods respect this parameter
2. **Automatic parameter passing**: The script automatically passes all needed parameters to initialization functions
3. **Projection function**: Required by `solve_PGD_amplitude`, automatically included
4. **Measurement conversion**: TNN initialization automatically converts phase retrieval measurements to tensor format
