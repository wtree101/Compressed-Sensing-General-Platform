function run_rank_experiment(r, lambda, m_all, params, save_dir, add_flag)
    % Run experiments for a single rank value across all measurements
    % Inputs:
    %   r - Current rank value
    %   lambda - Regularization parameter
    %   m_all - Array of measurement numbers
    %   params - Experiment parameters
    %   save_dir - Directory to save results
    %   add_flag - Whether to add to existing data (1) or overwrite (0)
    
    points_num = length(m_all);
    trial_num = params.trial_num;
    
    % Display progress
    fprintf('\nRunning experiments for rank r=%d, lambda=%.2e\n', r, lambda);
    fprintf('Parameters: d1=%d, d2=%d, trials=%d, T=%d, points=%d\n', ...
        params.d1, params.d2, trial_num, params.T, points_num);
    
    tic;
    
    % Update params for this rank and lambda
    params.r = r;
    params.lambda = lambda;
    
    % Preallocate results
    points_r = cell(points_num, 1);
    
    % Run experiments in parallel
    parfor i = 1:points_num
        m = m_all(i);
        [err_list, p_list, err_list_f] = multipletrial(m, r, params.kappa, lambda, params);
        
        points = struct();
        points.r = r;
        points.m = m;
        points.e = err_list;
        points.p = p_list;
        points.e_f = err_list_f;
        points.trial_num = trial_num;
        
        points_r{i} = points;
    end
    
    elapsed_time = toc;
    fprintf('Computation completed in %.2f seconds\n', elapsed_time);
    
    % Save results
    fprintf('Saving results...\n');
    save_experiment_results(points_r, save_dir, add_flag);
    fprintf('Results saved to: %s\n', save_dir);
end
