%%%%%%%%%% Phase Diagram for Tensor-Lifted Matrix Recovery
% This script generates phase diagrams for low-rank symmetric matrix recovery
% using various initialization methods followed by local refinement.
%
% Available Initialization Methods:
%   1. Tucker Spectral:        @initialize_tensor_lift_tucker_spectral (default)
%   2. Tensor Nuclear Norm:    @initialize_tensor_nuclear_norm
%   3. Power Method:           @initialize_power_method
%   4. Basic Tensor Lift:      @initialize_tensor_lift
%   5. Random:                 @Initialization_random
%
% Key Parameters:
%   - T_power: Number of initialization steps/iterations (general parameter)
%              Works for all initialization methods
%   - T: Number of local refinement iterations
%   - mu: Step size for gradient descent
%
% Workflow:
%   1. Initialize with T_power iterations (specified initialization method)
%   2. Refine with T iterations (gradient descent on matrix manifold)
%
% The script uses a modular design that accepts different initialization
% function handles. Simply uncomment your preferred initialization method
% in the configuration section below.

clear; clc;

%% Experiment Configuration
fprintf('=== Low rank phase retrieval; Support Tensor-Lifted method; Phase Diagram Setup ===\n');

% trial_func = @onetrial_MatTensor;
% alg_func = @solve_PGD;
% alg_name = 'TensorPGD';
% init_method = [];
% nonlinear_func = [];

%% Select Trial and Algorithm Functions
trial_func = @onetrial_Mat;
alg_func = @solve_PGD_amplitude;
nonlinear_func = @(y) abs(y);  % Phase retrieval model (amplitude measurements)
pre_func = [];  % Preprocessing function (optional, e.g., @set_zero_outside_range_tensor)

%% Select Initialization Method
% Available initialization methods:
% 1. Power method:               @initialize_power_method
% 2. Tensor lift (basic):        @initialize_tensor_lift
% 3. Tucker spectral:            @initialize_tensor_lift_tucker_spectral
% 4. Tensor nuclear norm v1:     @initialize_tensor_nuclear_norm
% 5. Tensor nuclear norm v2:     @initialize_tensor_nuclear_norm_v2 (NEW - separate penalties)
% 6. Random:                     @Initialization_random

% Choose one initialization method:
% init_method = @initialize_tensor_lift_tucker_spectral;
% alg_name = 'MatsubGD_tensorSpectralinit';
% T_power = 20;  % Number of initialization steps/iterations

% Tensor Nuclear Norm v2 initialization (with separate penalty parameters)
init_method = @initialize_tensor_nuclear_norm_v2;
alg_name = 'MatsubGD_TNNv2init';
T_power = 100;  % Number of TNN initialization iterations

% Alternative: Tensor Nuclear Norm v1 initialization
% init_method = @initialize_tensor_nuclear_norm;
% alg_name = 'MatsubGD_TNNinit';
% T_power = 100;  % Number of TNN initialization iterations

% Alternative: Power method initialization
% init_method = @initialize_power_method;
% alg_name = 'MatsubGD_powerinit';
% T_power = 20;  % Number of power method iterations

% Matrix dimensions and problem setup
d1 = 20;             % Matrix row dimension
d2 = d1;             % Matrix column dimension (d1 x d2)
kappa = 2;           % Condition number
         % Target rank for ground truth
r_max = 5;          % Maximum rank to test
r_grid = 1:1:5;     % Rank values to test

% Experiment parameters
trial_num = 3;       % Number of trials per (r, m) pair
verbose = 0;         % 0: minimal output, 1: detailed output
add_flag = 0;        % 0: overwrite existing data, 1: add to existing data
T = 200;             % Number of local refinement iterations
problem_flag = 2;    % 2: phase retrieval, 0: sensing
use_parallel = false; % true: use parpool/parfor, false: sequential computation

%% Initialization-Specific Parameters
% T_power: General parameter for number of initialization steps/iterations
%   - For Tucker spectral: number of RGD iterations on Tucker manifold
%   - For Tensor nuclear norm v1/v2: number of ADMM iterations
%   - For Power method: number of power iterations
%
% Note: T_power is set above with the initialization method selection

% V2-specific penalty parameters (for initialize_tensor_nuclear_norm_v2)
rho_k = 0.1;         % Penalty parameter for unfolding constraints W_k = T_(k)
rho_m = 1.0;         % Penalty parameter for measurement constraints
lambda_vec = [1, 1, 1, 1];  % Weight vector for mode nuclear norms

% Grid generation parameters
scale_num = 4;       % Number of scale levels for measurement grid

% Step size parameters to test
mu_list = [0.1];    % Step sizes for tensor PGD
% mu_list = [0.1, 0.05, 0.01, 0.005, 0.001];

fprintf('Configuration:\n');
fprintf('  Matrix size: %dx%d\n', d1, d2);
fprintf('  Rank grid: [%d, %d] with %d values\n', min(r_grid), max(r_grid), length(r_grid));
fprintf('  Trials per point: %d\n', trial_num);
fprintf('  Initialization iterations (T_power): %d\n', T_power);
fprintf('  Local refinement iterations (T): %d\n', T);
fprintf('  Step sizes (mu): %d values\n', length(mu_list));
fprintf('  Initialization method: %s\n', func2str(init_method));
fprintf('  Algorithm name: %s\n', alg_name);
if strcmp(func2str(init_method), 'initialize_tensor_nuclear_norm_v2')
    fprintf('  V2 Parameters: rho_k=%.2e, rho_m=%.2e, lambda=[%.2f,%.2f,%.2f,%.2f]\n', ...
            rho_k, rho_m, lambda_vec(1), lambda_vec(2), lambda_vec(3), lambda_vec(4));
end

%% Generate Measurement Grid and Setup Directory
fprintf('\n=== Setting up Measurement Grid ===\n');
grid_params = struct('d1', d1, 'd2', d2, 'r_max', r_max, 'kappa', kappa, ...
                     'problem_flag', problem_flag, ...
                     'alg_name', alg_name, 'scale_num', scale_num);
[m_all, data_dir] = setup_measurement_grid(grid_params);
fprintf('Measurement grid: %d values from m=%d to m=%d\n', ...
        length(m_all), min(m_all), max(m_all));
fprintf('Total experiments: %d rank values × %d mu values × %d m values = %d points\n', ...
        length(r_grid), length(mu_list), length(m_all), ...
        length(r_grid) * length(mu_list) * length(m_all));


%% Run Experiments
fprintf('\n=== Starting Tensor Phase Diagram Experiments ===\n');
experiment_start_time = tic;  % Start timing the entire experiment

% Initialize parallel pool if requested
if use_parallel
    pool = gcp('nocreate'); % Check if pool already exists
    if isempty(pool)
        parpool(5); % Create pool with 5 workers
        fprintf('Parallel pool initialized with 5 workers.\n');
    else
        fprintf('Using existing parallel pool with %d workers.\n', pool.NumWorkers);
    end
else
    fprintf('Running in sequential mode (parallel processing disabled).\n');
end

total_experiments = length(mu_list) * length(r_grid);
experiment_count = 0;

for mu_idx = 1:length(mu_list)
    mu = mu_list(mu_idx);
    mu_start_time = tic;  % Start timing for this mu value
    
    % Create subdirectory for this step size
    mu_dir = fullfile(data_dir, sprintf('mu_%.4f', mu));
    if ~exist(mu_dir, 'dir')
        mkdir(mu_dir);
    end
    
    fprintf('\n--- Step size mu = %.4f (%d/%d) ---\n', mu, mu_idx, length(mu_list));
    
    for r = r_grid
        experiment_count = experiment_count + 1;
        rank_start_time = tic;  % Start timing for this rank
        
        fprintf('\nExperiment %d/%d: r=%d, mu=%.4f\n', ...
                experiment_count, total_experiments, r, mu);
        
        %% Run experiments for this rank across all measurement counts
        % File to store aggregated results for this rank
        result_file = fullfile(mu_dir, sprintf('r_%d.mat', r));
        
        % Check if results already exist
        if add_flag && exist(result_file, 'file')
            load(result_file, 'results');
            fprintf('  Loaded existing results from %s\n', result_file);
        else
            results = struct();
            results.r = r;
            results.mu = mu;
            results.m_values = m_all;
            results.success_count = zeros(size(m_all));
            results.avg_error = zeros(size(m_all));
            results.std_error = zeros(size(m_all));
            results.avg_time = zeros(size(m_all));
            results.trial_errors = cell(size(m_all));
        end
        
        % Generate ground truth once for all trials at this rank
        if d1 ~= d2
            warning('d1 != d2: generating non-square ground truth matrix');
        end
        U_true = randn(d1, r);
        %Xstar = U_true * U_true';
        Xstar = abs(U_true) * abs(U_true)';  % Symmetric rank-r_star matrix
        Xstar = Xstar / norm(Xstar, 'fro');
        
        % Loop over measurement counts
        for m_idx = 1:length(m_all)
            m = m_all(m_idx);
            point_start_time = tic;  % Start timing for this (r, m) point
            
            fprintf('  m=%d (%d/%d): ', m, m_idx, length(m_all));
            
            % Skip if already computed and add_flag is on
            if add_flag && results.success_count(m_idx) >= trial_num
                fprintf('Already computed. Skipping.\n');
                continue;
            end
            
            % Setup parameters for multiple trials
            trial_params = struct();
            trial_params.d1 = d1;
            trial_params.d2 = d2;
            trial_params.m = m;
            trial_params.r = r;
            trial_params.kappa = kappa;
            trial_params.T = T;  % Local refinement iterations
            trial_params.T_power = T_power;  % Initialization iterations (general parameter)
            trial_params.mu = mu;  % Step size for local refinement
            trial_params.Xstar = Xstar;
            trial_params.verbose = verbose;
            trial_params.init = init_method;  % Initialization function handle
            trial_params.trial_num = trial_num;
            trial_params.use_parallel = use_parallel;
            trial_params.onetrial = trial_func;
            trial_params.alg_func = alg_func;
            trial_params.nonlinear_func = nonlinear_func;
            trial_params.pre_func = pre_func;
            trial_params.projection = @(X) project_rank_r(X, r);  % Rank-r projection
            
            % Additional parameters for specific initialization methods
            % These are passed to the initialization function via params struct
            trial_params.max_iter = T_power;  % For tensor nuclear norm: ADMM iterations
            trial_params.normalize = true;    % Normalize initialization output
            
            % V2-specific parameters (for initialize_tensor_nuclear_norm_v2)
            trial_params.rho_k = rho_k;       % Penalty for unfolding constraints
            trial_params.rho_m = rho_m;       % Penalty for measurement constraints
            trial_params.lambda = lambda_vec; % Mode nuclear norm weights
            
            % V1-specific parameters (for initialize_tensor_nuclear_norm)
            trial_params.rho = 0.1;           % For TNN v1: single ADMM penalty
            % Run multiple trials using existing multipletrial function
            [output, success_rate] = multipletrial(trial_params);
            
            % Store results from output
            point_elapsed_time = toc(point_start_time);  % Compute elapsed time
            results.success_count(m_idx) = round(success_rate * trial_num);
            results.avg_error(m_idx) = output(end);  % Final average error
            results.std_error(m_idx) = 0;  % Not computed by multipletrial
            results.avg_time(m_idx) = point_elapsed_time;   % Store actual time
            results.trial_errors{m_idx} = output;  % Store average error history
            
            fprintf('Success: %d/%d (%.1f%%), Final Error: %.4e, Time: %.2f s\n', ...
                    results.success_count(m_idx), trial_num, success_rate*100, ...
                    results.avg_error(m_idx), point_elapsed_time);
            
            % Save individual point data using helper function
            save_experiment_point(mu_dir, r, m, mu, trial_num, ...
                                 success_rate, output, T, init_method, add_flag);
            
            % Save intermediate results (summary format)
            save(result_file, 'results');
        end
        
        rank_elapsed_time = toc(rank_start_time);  % Compute elapsed time for this rank
        fprintf('  Results saved to: %s\n', result_file);
        fprintf('  Rank r=%d completed in %.2f seconds (%.2f minutes)\n', r, rank_elapsed_time, rank_elapsed_time/60);
    end
    
    mu_elapsed_time = toc(mu_start_time);  % Compute elapsed time for this mu value
    fprintf('\n--- mu=%.4f experiments completed in %.2f seconds (%.2f minutes) ---\n', ...
            mu, mu_elapsed_time, mu_elapsed_time/60);
end

%% Cleanup
total_elapsed_time = toc(experiment_start_time);  % Compute total elapsed time
total_points_computed = total_experiments * length(m_all);
avg_time_per_point = total_elapsed_time / total_points_computed;

fprintf('\n=== Experiment Complete ===\n');
fprintf('Results saved in: %s\n', data_dir);
fprintf('\n--- Timing Summary ---\n');
fprintf('Total experiment time: %.2f seconds (%.2f minutes, %.2f hours)\n', ...
        total_elapsed_time, total_elapsed_time/60, total_elapsed_time/3600);
fprintf('Total points computed: %d (r_grid: %d, m_grid: %d)\n', ...
        total_points_computed, length(r_grid), length(m_all));
fprintf('Average time per (r,m) point: %.2f seconds\n', avg_time_per_point);

% Clean up parallel pool if it was created
if use_parallel
    delete(gcp('nocreate'));
    fprintf('Parallel pool closed.\n');
end

fprintf('Tensor phase diagram generation completed successfully!\n');


