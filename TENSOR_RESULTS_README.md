# Tensor Phase Diagram Results Management

This document describes the modular system for saving and visualizing tensor phase diagram experiment results.

## Overview

The system separates concerns into:
1. **Experiment runner** - Generic function to run experiments with any solver
2. **Data saving utilities** - Handle writing results to disk
3. **Data loading utilities** - Handle reading and aggregating results
4. **Visualization scripts** - Create plots and figures

## File Structure

### Core Experiment Function

#### `run_rank_experiment()` (in `utilities/run_rank_experiment_generic.m`)
Generic function to run phase diagram experiments for a specific rank across multiple measurement counts.

**Key Feature**: Accepts any solver via function handle parameter!

**Usage:**
```matlab
run_rank_experiment(r, mu, m_all, d, T, trial_num, r_star, kappa, ...
                   init_method, save_dir, add_flag, verbose, use_parallel, onetrial_func)
```

**Parameters:**
- `r` - Rank value to test
- `mu` - Step size
- `m_all` - Array of measurement counts
- `d` - Dimension
- `T` - Iterations per trial
- `trial_num` - Trials per point
- `r_star` - Ground truth rank
- `kappa` - Condition number
- `init_method` - Initialization method
- `save_dir` - Output directory
- `add_flag` - 0: overwrite, 1: merge
- `verbose` - Verbosity level
- `use_parallel` - Use parallel processing
- `onetrial_func` - **Function handle** (e.g., `@onetrial_tensor`, `@onetrial_vec`)

**Examples:**
```matlab
% Tensor solver
run_rank_experiment(2, 0.01, [100 200 300], 20, 100, 5, 1, 2, ...
                   'zero', './results', 0, 0, false, @onetrial_tensor);

% Vector solver
run_rank_experiment(2, 0.01, [100 200 300], 20, 100, 5, 1, 2, ...
                   'zero', './results', 0, 0, false, @onetrial_vec);

% Custom solver
my_solver = @(params) my_custom_solver(params);
run_rank_experiment(2, 0.01, [100 200 300], 20, 100, 5, 1, 2, ...
                   'zero', './results', 0, 0, false, my_solver);
```

### Utilities (in `utilities/`)

#### `save_tensor_experiment_point.m`
Saves individual experiment point results with merge capability.

**Usage:**
```matlab
save_tensor_experiment_point(save_dir, r, m, mu, trial_num, ...
                             success_rate, output, T, init_method, add_flag)
```

**Features:**
- Saves to `r_<r>_m_<m>_t_<trial_num>.mat`
- When `add_flag=1`: merges with existing data using weighted averages
- When `add_flag=0`: overwrites existing data

**Point structure saved:**
```matlab
point.r           % Rank
point.m           % Number of measurements
point.mu          % Step size
point.trial_num   % Total trials
point.p           % Success probability
point.e           % Error history (T×1)
point.T           % Number of iterations
point.init_method % Initialization method
```

#### `load_tensor_experiment_results.m`
Loads and aggregates all results from a directory.

**Usage:**
```matlab
results = load_tensor_experiment_results(data_dir);
% OR with filtering:
results = load_tensor_experiment_results(data_dir, r_grid, m_grid);
```

**Returns:**
```matlab
results.r_values       % Unique rank values
results.m_values       % Unique measurement values
results.success_matrix % Success rate matrix (r × m)
results.error_matrix   % Final error matrix (r × m)
results.trial_count    % Trial count matrix (r × m)
results.points         % Cell array of all point structures
results.data_dir       % Source directory
```

### Visualization Scripts (in `draw/`)

#### `visualize_tensor_results.m`
Main visualization script for a single mu directory.

**Features:**
- Loads results automatically
- Creates success rate heatmap with annotations
- Creates log error heatmap
- Plots error evolution curves
- Saves figures to data directory

**Usage:**
```matlab
% Edit the data_dir path at top of file, then run:
cd draw
visualize_tensor_results
```

#### `batch_visualize_tensor.m`
Batch processing for multiple mu values.

**Features:**
- Automatically finds all `mu_*` subdirectories
- Creates individual visualizations for each mu
- Creates comparison plots across different mu values
- Shows success rate and error trends

**Usage:**
```matlab
% Edit the base_dir path at top of file, then run:
cd draw
batch_visualize_tensor
```

## Integration with Phasediagram_tensor.m

The main experiment script `Phasediagram_tensor.m` now uses `save_tensor_experiment_point()` to save results automatically after each experiment point.

**Key parameters:**
- `add_flag = 0`: Overwrite existing results
- `add_flag = 1`: Merge new trials with existing trials

## Example Workflow

### 1. Run Experiments
```matlab
cd matrix_recovery
Phasediagram_tensor  % Uses default parameters in script
```

This creates:
```
data_f/err_data_d1_20_d2_20_rmax_2_kappa_2_rstar_1_prob_2_alg_PGD/
├── config.mat
└── mu_0.0100/
    ├── r_1.mat                    % Summary file
    ├── r_2.mat                    % Summary file
    ├── r_1_m_100_t_5.mat         % Individual points
    ├── r_1_m_150_t_5.mat
    ├── r_2_m_100_t_5.mat
    └── ...
```

### 2. Visualize Results
```matlab
cd draw
visualize_tensor_results
```

### 3. Add More Trials (Optional)
```matlab
% In Phasediagram_tensor.m, set:
add_flag = 1;      % Enable merging
trial_num = 10;    % Add 10 more trials

% Run again - results will be merged with existing data
Phasediagram_tensor
```

### 4. Compare Multiple Step Sizes
```matlab
cd draw
batch_visualize_tensor
```

## Data Format Details

### Individual Point Files
Each `r_<r>_m_<m>_t_<trials>.mat` contains a `point` structure:
- Compatible with old format for `drawPR.m`
- Contains complete error history
- Can be incrementally updated with `add_flag=1`

### Summary Files
Each `r_<r>.mat` contains a `results` structure:
- Quick overview of all measurements for one rank
- Success counts and average errors
- Stored in cell array for error histories

## Advantages of This System

1. **Modularity**: Separate saving, loading, and visualization
2. **Reusability**: Utility functions can be used in other scripts
3. **Compatibility**: Works with existing drawing scripts
4. **Incremental updates**: Can add more trials without recomputing
5. **Clean code**: Main experiment script is more readable

## Customization

### To add new visualization types:
Create a new script in `draw/` that uses `load_tensor_experiment_results()`.

### To change saved data:
Modify `save_tensor_experiment_point()` to include additional fields.

### To batch process experiments:
Use the utility functions in your own scripts:
```matlab
% Your custom batch script
for mu = [0.001, 0.01, 0.1]
    % Set parameters
    % Run experiments
    save_tensor_experiment_point(...);
end

% Then visualize all at once
results = load_tensor_experiment_results(data_dir);
% Your custom visualization code
```
