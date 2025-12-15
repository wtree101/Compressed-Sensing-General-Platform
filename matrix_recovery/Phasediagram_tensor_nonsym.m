%%%%%%%%%% Phase Diagram for Tensor-Lifted Matrix Recovery (Non-Symmetric)
% This script generates phase diagrams for low-rank matrix recovery
% supporting BOTH symmetric and non-symmetric matrices
% using fourth-order tensor formulation: X = USV^T, T = X ⊗ X
% Linear model: y_i = ⟨A_i ⊗ A_i, T⟩
%
% For square matrices (d1 = d2): can use symmetric (X = USU^T) or non-symmetric
% For non-square matrices (d1 ≠ d2): automatically uses non-symmetric (X = USV^T)
%
% The script uses a modular design with run_rank_experiment() function
% that accepts different solver function handles (e.g., @onetrial_tensor)

clear; clc;

%% Experiment Configuration
fprintf('=== Low-Rank Matrix Recovery (Symmetric & Non-Symmetric): Phase Diagram Setup ===\n');

% trial_func = @onetrial_MatTensor;
% alg_func = @solve_PGD;
% alg_name = 'TensorPGD';
% init_method = [];
% nonlinear_func = [];

% Algorithm and trial configuration
trial_func = @onetrial_Mat;
alg_func = @solve_PGD_amplitude;
alg_name = 'PPM';  % TNN = Tensor Nuclear Norm
% tensorSpectralinit
% MatsubGD_TNNinit
%PPM

% Nonlinear function for measurements
nonlinear_func = @(y) abs(y);  % Phase retrieval model (set to [] or @(y) abs(y) for amplitude)

% Preprocessing function
pre_func = [];  % Optional: @(y) set_zero_outside_range_tensor(y)

% Initialization method
init_method = @initialize_tensor_lift_tucker_spectral;  % Options: @initialize_tensor_lift, @initialize_tensor_lift_tucker_spectral, @initialize_power_method, @Initialization, @Initialization_random
T_power = 0;  % Number of power iterations (if using power method initialization)


%% Matrix dimensions and problem setup
d1 = 10;             % Matrix row dimension
d2 = 20;             % Matrix column dimension (d1 x d2)
kappa = 2;           % Condition number
         % Target rank for ground truth
r_max = 3;          % Maximum rank to test
r_grid = 1:1:r_max;     % Rank values to test

% Experiment parameters
trial_num = 5;      % Number of trials per (r, m) pair
verbose = 0;         % 0: minimal output, 1: detailed output
add_flag = 0;        % 0: overwrite existing data, 1: add to existing data
T = 200;             % Number of iterations per trial
problem_flag = 2;
use_parallel = false; % true: use parpool/parfor, false: sequential computation

% Grid generation parameters
scale_num = 3;       % Number of scale levels for measurement grid

% Step size parameters to test
mu_list = [0.1];    % Step sizes for tensor PGD
% mu_list = [0.1, 0.05, 0.01, 0.005, 0.001];

fprintf('Configuration:\n');
if d1 == d2
    fprintf('  Matrix size: %dx%d (SQUARE - can be symmetric or non-symmetric)\n', d1, d2);
else
    fprintf('  Matrix size: %dx%d (NON-SQUARE - must be non-symmetric)\n', d1, d2);
end
fprintf('  Rank grid: [%d, %d] with %d values\n', min(r_grid), max(r_grid), length(r_grid));
fprintf('  Condition number: %.2f\n', kappa);
fprintf('  Trials per point: %d\n', trial_num);
fprintf('  Iterations per trial: %d\n', T);
fprintf('  Step sizes (mu): %d values\n', length(mu_list));
% fprintf('  Initialization: %s\n', func2str(init_method));

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
        % Use groundtruth function: supports both symmetric and non-symmetric
        if d1 == d2
            % Square matrix: can be symmetric or non-symmetric
            symflag = 1;  % Set to 1 for X = USU^T, 0 for X = USV^T
            fprintf('  Generating ground truth: %dx%d, rank=%d, symmetric=%d\n', ...
                    d1, d2, r, symflag);
        else
            % Non-square matrix: must be non-symmetric
            symflag = 0;  % Non-symmetric: X = USV^T
            fprintf('  Generating ground truth: %dx%d, rank=%d, non-symmetric (required)\n', ...
                    d1, d2, r);
        end
        
        % Call groundtruth function
        Xstar = groundtruth(d1, d2, r, kappa, symflag);
        
        fprintf('  Ground truth norm: %.6f, actual rank: %d\n', ...
                norm(Xstar, 'fro'), rank(Xstar, 1e-10));
        
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
            trial_params.T = T;
            trial_params.T_power = T_power;
            % trial_params.mu = mu;
            trial_params.Xstar = Xstar;
            trial_params.verbose = verbose;
            trial_params.init = init_method;  % Direct function handle
            trial_params.trial_num = trial_num;
            trial_params.use_parallel = use_parallel;
            trial_params.onetrial = trial_func;
            trial_params.alg_func = alg_func;
            trial_params.nonlinear_func = nonlinear_func;
            trial_params.pre_func = pre_func;
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
