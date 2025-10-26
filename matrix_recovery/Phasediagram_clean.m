%%%%%%%%%% Clean Phase Diagram Generation for Low-rank Matrix Recovery
% This is a refactored version of the original Phasediagram.m
% Functions have been modularized for better maintainability

clear; clc;

%% Experiment Configuration
% Matrix dimensions and problem setup
d1 = 60; d2 = 60;
kappa = 2;
r_star = 1;
r_max = 20;
r_grid = 1:1:20;

% Experiment parameters
trial_num = 10;
verbose = 0;
add_flag = 0;  % 0: overwrite existing data, 1: add to existing data
T = 1000;      % Number of iterations

% Problem and algorithm selection
problem_flag = 2; % 0=sensing, 1=phase retrieval, 2=symmetric Gaussian, 3=Richard's example, 4=PSD full rank
alg_flag = 3;     % 0=GD, 1=RGD, 2=SGD, 3=SubGD, 6=TensorPGD

% Grid generation parameters
scale_num = 3; % Number of scale levels for measurement grid

% Regularization parameters to test
lambda_list = [0];
% lambda_list = [1e-16,1e-15,1e-15,1e-13,1e-12,1e-11,1e-10,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2];

%% Setup Experiment
fprintf('=== Phase Diagram Experiment Setup ===\n');

% Configure experiment parameters and solvers
params = setup_experiment_params(d1, d2, kappa, trial_num, verbose, ...
                                problem_flag, alg_flag, r_star, T);

% Generate measurement grid and create data directory
grid_params = struct('d1', d1, 'd2', d2, 'r_max', r_max, 'kappa', kappa, ...
                     'r_star', r_star, 'problem_flag', problem_flag, ...
                     'alg_name', params.alg_name, 'scale_num', scale_num);
[m_all, data_dir] = setup_measurement_grid(grid_params);

%% Run Experiments
fprintf('\n=== Starting Experiments ===\n');

% Initialize parallel pool if needed
% parpool(16);

total_experiments = length(lambda_list) * length(r_grid);
experiment_count = 0;

for lambda_idx = 1:length(lambda_list)
    lambda = lambda_list(lambda_idx);
    
    % Create subdirectory for this lambda value
    lambda_dir = fullfile(data_dir, num2str(lambda));
    if ~exist(lambda_dir, 'dir')
        mkdir(lambda_dir);
    end
    
    fprintf('\n--- Lambda = %.2e (%d/%d) ---\n', lambda, lambda_idx, length(lambda_list));
    
    for r = r_grid
        experiment_count = experiment_count + 1;
        
        fprintf('\nExperiment %d/%d: r=%d, lambda=%.2e\n', ...
                experiment_count, total_experiments, r, lambda);
        
        % Run experiments for this rank and lambda
        run_rank_experiment(r, lambda, m_all, params, lambda_dir, add_flag);
    end
end

%% Cleanup
fprintf('\n=== Experiment Complete ===\n');
fprintf('Results saved in: %s\n', data_dir);

% Clean up parallel pool if needed
% delete(gcp('nocreate'));

fprintf('Phase diagram generation completed successfully!\n');
