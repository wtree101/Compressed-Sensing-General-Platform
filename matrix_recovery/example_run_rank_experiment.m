%%%%%%%%%% Example: Using run_rank_experiment with Different Solvers
% This script demonstrates how to use the generic run_rank_experiment function
% with different solver function handles

clear; clc;

%% Common Parameters
d = 20;              % Dimension
r_star = 1;          % Ground truth rank
kappa = 2;           % Condition number
T = 100;            % Iterations
trial_num = 5;       % Trials per point
mu = 0.01;          % Step size
verbose = 0;
add_flag = 0;
use_parallel = false;
init_method = 'zero';

% Measurement grid
m_all = [100, 200, 300];

% Rank to test
r = 2;

%% Example 1: Tensor PGD solver
fprintf('=== Example 1: Tensor PGD Solver ===\n');
save_dir_tensor = '../data_f/example_tensor_results';
if ~exist(save_dir_tensor, 'dir')
    mkdir(save_dir_tensor);
end

run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
                   r_star, kappa, init_method, save_dir_tensor, ...
                   add_flag, verbose, use_parallel, @onetrial_tensor, @solve_PGD);

%% Example 2: Vector PGD solver (if you have onetrial_vec)
fprintf('\n=== Example 2: Vector PGD Solver ===\n');
save_dir_vec = '../data_f/example_vector_results';
if ~exist(save_dir_vec, 'dir')
    mkdir(save_dir_vec);
end

% Uncomment if you have onetrial_vec function:
% run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
%                    r_star, kappa, init_method, save_dir_vec, ...
%                    add_flag, verbose, use_parallel, @onetrial_vec, @solve_PGD);

%% Example 3: Tensor with different algorithm (GD, RGD, AP, etc.)
fprintf('\n=== Example 3: Tensor with different algorithms ===\n');

% Using Gradient Descent
save_dir_gd = '../data_f/example_tensor_gd';
if ~exist(save_dir_gd, 'dir')
    mkdir(save_dir_gd);
end
% run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
%                    r_star, kappa, init_method, save_dir_gd, ...
%                    add_flag, verbose, use_parallel, @onetrial_tensor, @solve_GD);

% Using Alternating Projection
save_dir_ap = '../data_f/example_tensor_ap';
if ~exist(save_dir_ap, 'dir')
    mkdir(save_dir_ap);
end
% run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
%                    r_star, kappa, init_method, save_dir_ap, ...
%                    add_flag, verbose, use_parallel, @onetrial_tensor, @solve_AP);

%% Example 4: Batch processing multiple ranks
fprintf('\n=== Example 4: Batch Multiple Ranks ===\n');
save_dir_batch = '../data_f/example_batch_results';
if ~exist(save_dir_batch, 'dir')
    mkdir(save_dir_batch);
end

r_grid = [1, 2, 3];
for r = r_grid
    fprintf('\n--- Rank %d ---\n', r);
    run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
                       r_star, kappa, init_method, save_dir_batch, ...
                       add_flag, verbose, use_parallel, @onetrial_tensor, @solve_PGD);
end

%% Example 5: Comparing different step sizes
fprintf('\n=== Example 5: Multiple Step Sizes ===\n');
base_dir = '../data_f/example_mu_comparison';
if ~exist(base_dir, 'dir')
    mkdir(base_dir);
end

mu_list = [0.001, 0.01, 0.1];
for mu = mu_list
    mu_dir = fullfile(base_dir, sprintf('mu_%.4f', mu));
    if ~exist(mu_dir, 'dir')
        mkdir(mu_dir);
    end
    
    fprintf('\n--- Step size mu = %.4f ---\n', mu);
    run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
                       r_star, kappa, init_method, mu_dir, ...
                       add_flag, verbose, use_parallel, @onetrial_tensor, @solve_PGD);
end

%% Example 6: Comparing different algorithms for same problem
fprintf('\n=== Example 6: Algorithm Comparison ===\n');
base_dir_alg = '../data_f/example_algorithm_comparison';
if ~exist(base_dir_alg, 'dir')
    mkdir(base_dir_alg);
end

algorithms = {@solve_PGD, @solve_GD, @solve_RGD};
alg_names = {'PGD', 'GD', 'RGD'};

for alg_idx = 1:length(algorithms)
    alg_dir = fullfile(base_dir_alg, alg_names{alg_idx});
    if ~exist(alg_dir, 'dir')
        mkdir(alg_dir);
    end
    
    fprintf('\n--- Algorithm: %s ---\n', alg_names{alg_idx});
    % Uncomment to run:
    % run_rank_experiment(r, mu, m_all, d, T, trial_num, ...
    %                    r_star, kappa, init_method, alg_dir, ...
    %                    add_flag, verbose, use_parallel, ...
    %                    @onetrial_tensor, algorithms{alg_idx});
end

fprintf('\n=== All Examples Complete ===\n');

%% Visualize Results (Example)
fprintf('\nTo visualize results, use:\n');
fprintf('  results = load_tensor_experiment_results(''%s'');\n', save_dir_batch);
fprintf('  or run visualize_tensor_results.m with appropriate data_dir\n');
