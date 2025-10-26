# Algorithm Function Handle Integration

## Summary

Added `alg_func` parameter to the experiment framework, enabling easy switching between different optimization algorithms (PGD, GD, RGD, AP, etc.) without modifying the core experiment code.

## Changes Made

### 1. **Phasediagram_tensor.m**
- Added `alg_func = @solve_PGD` at the top configuration section
- Updated `run_rank_experiment` to accept `alg_func` parameter
- Passes `alg_func` through `trial_params` to multipletrial and onetrial functions

### 2. **utilities/run_rank_experiment_generic.m**
- Updated function signature to include `alg_func` parameter
- Added `trial_params.alg_func` in parameter setup
- Updated documentation with examples

### 3. **matrix_recovery/example_run_rank_experiment.m**
- Updated all examples to include `alg_func` parameter
- Added Example 6: Algorithm comparison workflow
- Shows how to compare PGD, GD, RGD, etc. side-by-side

## Usage

### Basic Usage
```matlab
% Configure at top of script
trial_func = @onetrial_tensor;
alg_func = @solve_PGD;  % or @solve_GD, @solve_RGD, @solve_AP, etc.

% Run experiments
run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
                   r_star, kappa, init_method, save_dir, ...
                   add_flag, verbose, use_parallel, ...
                   trial_func, alg_func);
```

### Algorithm Comparison
```matlab
% Compare different algorithms
algorithms = {@solve_PGD, @solve_GD, @solve_RGD};
alg_names = {'PGD', 'GD', 'RGD'};

for i = 1:length(algorithms)
    alg_dir = fullfile(base_dir, alg_names{i});
    mkdir(alg_dir);
    
    run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
                       r_star, kappa, init_method, alg_dir, ...
                       add_flag, verbose, use_parallel, ...
                       @onetrial_tensor, algorithms{i});
end

% Then visualize and compare results from different directories
```

### Combining Different Trial Functions and Algorithms
```matlab
% Tensor formulation with PGD
run_rank_experiment(..., @onetrial_tensor, @solve_PGD);

% Tensor formulation with GD
run_rank_experiment(..., @onetrial_tensor, @solve_GD);

% Vector formulation with PGD
run_rank_experiment(..., @onetrial_vec, @solve_PGD);

% Vector formulation with RGD
run_rank_experiment(..., @onetrial_vec, @solve_RGD);
```

## Parameter Flow

```
Phasediagram_tensor.m
├── trial_func = @onetrial_tensor
├── alg_func = @solve_PGD
└── run_rank_experiment(r, mu, ..., trial_func, alg_func)
    └── trial_params.onetrial = trial_func
    └── trial_params.alg_func = alg_func
        └── multipletrial(trial_params)
            └── onetrial(trial_params)  [uses params.onetrial and params.alg_func]
```

## Benefits

1. **Easy Algorithm Switching**: Change one line to test different algorithms
2. **Algorithm Comparison**: Run same experiments with different solvers
3. **Clean Separation**: Algorithm choice separated from experiment logic
4. **Flexibility**: Mix and match trial functions and algorithms
5. **Reproducibility**: Clear specification of which algorithm was used

## Example: Algorithm Comparison Study

```matlab
%% Setup
d = 20;
r = 2;
m_all = [100, 200, 300];
base_dir = './algorithm_comparison';

%% Test all algorithms
algorithms = {
    @solve_PGD,    'PGD';
    @solve_GD,     'GD';
    @solve_RGD,    'RGD';
    @solve_AP,     'AP'
};

for i = 1:size(algorithms, 1)
    alg_func = algorithms{i, 1};
    alg_name = algorithms{i, 2};
    save_dir = fullfile(base_dir, alg_name);
    
    fprintf('\n=== Testing %s ===\n', alg_name);
    run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
                       r_star, kappa, init_method, save_dir, ...
                       0, 0, false, @onetrial_tensor, alg_func);
end

%% Compare results
for i = 1:size(algorithms, 1)
    alg_name = algorithms{i, 2};
    results = load_tensor_experiment_results(fullfile(base_dir, alg_name));
    fprintf('%s: Success rate = %.1f%%\n', alg_name, ...
            mean(results.success_matrix(:))*100);
end
```

## Notes

- The `alg_func` must be compatible with the trial function
- Make sure onetrial functions extract and use `params.alg_func` correctly
- Different algorithms may require different parameters in `trial_params`
