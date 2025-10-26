function run_rank_experiment_generic(r, mu, m_all, d1, d2, T, trial_num, ...
                            r_star, kappa, init_method, save_dir, add_flag, verbose, use_parallel, onetrial_func, alg_func)
    % RUN_RANK_EXPERIMENT Run phase diagram experiments for a specific rank
    %
    % This is a generic function for running phase diagram experiments across
    % different measurement counts for a fixed rank value. It can be used with
    % different solvers by passing appropriate function handles.
    %
    % Inputs:
    %   r              - Rank value to test
    %   mu             - Step size for gradient descent
    %   m_all          - Array of measurement counts to test
    %   d1             - Matrix row dimension
    %   d2             - Matrix column dimension
    %   T              - Number of iterations per trial
    %   trial_num      - Number of trials per (r, m) point
    %   r_star         - Ground truth rank
    %   kappa          - Condition number
    %   init_method    - Initialization method ('zero', 'random', 'power', etc.)
    %   save_dir       - Directory to save results
    %   add_flag       - 0: overwrite existing data, 1: merge with existing
    %   verbose        - Verbosity level (0: minimal, 1: detailed)
    %   use_parallel   - Whether to use parallel processing (true/false)
    %   onetrial_func  - Function handle for single trial experiment
    %                    Examples: @onetrial_tensor, @onetrial_vec, @onetrial_GD
    %   alg_func       - Function handle for algorithm solver
    %                    Examples: @solve_PGD, @solve_GD, @solve_RGD, @solve_AP
    %
    % Outputs:
    %   Saves results to two types of files:
    %   1. r_<r>.mat - Summary file with aggregated results
    %   2. r_<r>_m_<m>_t_<trials>.mat - Individual point files
    %
    % Example usage:
    %   run_rank_experiment_generic(2, 0.01, [100 200 300], 20, 20, 100, 10, ...
    %                       1, 2, 'zero', './results', 0, 0, false, ...
    %                       @onetrial_tensor, @solve_PGD)
    
    % File to store aggregated results
    result_file = fullfile(save_dir, sprintf('r_%d.mat', r));
    
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
    
    % Generate ground truth once for all trials
    % For symmetric matrices, d1 should equal d2
    if d1 ~= d2
        warning('d1 != d2: generating non-square ground truth matrix');
    end
    U_true = randn(d1, r_star);
    Xstar = U_true * U_true';  % Symmetric rank-r_star matrix
    Xstar = Xstar / norm(Xstar, 'fro');
    
    % Loop over measurement counts
    for m_idx = 1:length(m_all)
        m = m_all(m_idx);
        
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
        trial_params.mu = mu;
        trial_params.Xstar = Xstar;
        trial_params.verbose = verbose;
        trial_params.init_method = init_method;
        trial_params.trial_num = trial_num;
        trial_params.use_parallel = use_parallel;  % Pass parallel flag
        trial_params.onetrial = onetrial_func;     % Use passed function handle
        trial_params.alg_func = alg_func;          % Pass algorithm function handle
        
        % Run multiple trials using existing multipletrial function
        % multipletrial returns [output, success_rate]
        % where output is the averaged error history across trials
        [output, success_rate] = multipletrial(trial_params);
        
        % Store results from output
        results.success_count(m_idx) = round(success_rate * trial_num);
        results.avg_error(m_idx) = output(end);  % Final average error
        results.std_error(m_idx) = 0;  % Not computed by multipletrial
        results.avg_time(m_idx) = 0;   % Not computed by multipletrial
        results.trial_errors{m_idx} = output;  % Store average error history
        
        fprintf('Success: %d/%d (%.1f%%), Final Error: %.4e\n', ...
                results.success_count(m_idx), trial_num, success_rate*100, ...
                results.avg_error(m_idx));
        
        % Save individual point data using helper function
        save_experiment_point(save_dir, r, m, mu, trial_num, ...
                             success_rate, output, T, init_method, add_flag);
        
        % Save intermediate results (summary format)
        save(result_file, 'results');
    end
    
    fprintf('  Results saved to: %s\n', result_file);
end
