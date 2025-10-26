function params = setup_experiment_params(d1, d2, kappa, trial_num, verbose, problem_flag, alg_flag, r_star, T)
    % Setup experiment parameters and configure solvers, initializers, and nonlinear functions
    % Inputs:
    %   d1, d2 - Matrix dimensions
    %   kappa - Condition number
    %   trial_num - Number of trials
    %   verbose - Verbosity flag
    %   problem_flag - Problem type (0=sensing, 1=phase retrieval, etc.)
    %   alg_flag - Algorithm type (0=GD, 1=RGD, 2=SGD, 3=SubGD)
    %   r_star - True rank
    %   T - Number of iterations
    % Output:
    %   params - Configured parameter structure
    
    % Basic parameters
    params.d1 = d1;
    params.d2 = d2;
    params.kappa = kappa;
    params.trial_num = trial_num;
    params.verbose = verbose;
    params.problem_flag = problem_flag;
    params.T = T;
    params.init_scale = 1e-10;
    
    % Set algorithm name and solver
    [alg_name, alg_handle] = set_solver(alg_flag);
    params.alg = alg_handle;
    params.alg_name = alg_name;
    
    % Set initialization method
    params.init_flag = 1; % Use random initialization
    [init_name, init_handle] = set_init(params.init_flag);
    params.init = init_handle;
    params.init_name = init_name;
    
    % Set nonlinear function based on problem type
    if problem_flag == 1
        nonlinear_flag = 1; % Absolute value for phase retrieval
    else
        nonlinear_flag = 0; % Identity for other problems
    end
    params.nonlinear_flag = nonlinear_flag;
    [nonlinear_name, nonlinear_handle] = set_nonlinear(nonlinear_flag);
    params.nonlinear_func = nonlinear_handle;
    params.nonlinear_name = nonlinear_name;
    
    % Set ground truth and special parameters
    if problem_flag == 3
        % Richard's example
        [A, Xstar] = prob3(d1, 100, r_star, r_star);
        Xstar = Xstar / norm(Xstar, 'fro'); % Normalize Xstar
        params.A = A;
        params.mu = 20; % Special mu for Richard's example
    else
        Xstar = groundtruth(d1, d2, r_star, kappa, 1);
    end
    params.Xstar = Xstar;
    
    % Display configuration
    fprintf('Experiment Configuration:\n');
    fprintf('  Problem: %s, Algorithm: %s, Init: %s, Nonlinear: %s\n', ...
        get_problem_name(problem_flag), alg_name, init_name, nonlinear_name);
    fprintf('  Dimensions: %dx%d, Rank: %d, Kappa: %.1f, Iterations: %d\n', ...
        d1, d2, r_star, kappa, T);
end

function problem_name = get_problem_name(problem_flag)
    switch problem_flag
        case 0
            problem_name = 'Sensing';
        case 1
            problem_name = 'Phase Retrieval';
        case 2
            problem_name = 'Symmetric Gaussian';
        case 3
            problem_name = 'Richard''s Example';
        case 4
            problem_name = 'PSD Full Rank';
        otherwise
            problem_name = 'Unknown';
    end
end
