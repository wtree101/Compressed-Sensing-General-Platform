%% Minimal Test Script
clear; clc;

fprintf('Starting minimal test...\n');

try
    % Test 1: Check if basic functions are available
    fprintf('Test 1: Checking function availability...\n');
    if exist('setup_experiment_params', 'file')
        fprintf('  ✓ setup_experiment_params found\n');
    else
        error('setup_experiment_params not found');
    end
    
    if exist('onetrial', 'file')
        fprintf('  ✓ onetrial found\n');
    else
        error('onetrial not found');
    end
    
    % Test 2: Setup basic parameters
    fprintf('Test 2: Setting up parameters...\n');
    m = 10; r = 2; kappa = 0.5;
    d1 = 10; d2 = 10; trial_num = 1; verbose = false;
    problem_flag = 0; alg_flag = 0; r_star = r; T = 10;
    lambda = 0.01;
    
    % Setup params structure
    params = setup_experiment_params(d1, d2, kappa, trial_num, verbose, ...
                                    problem_flag, alg_flag, r_star, T);
    params.lambda = lambda;
    
    fprintf('  ✓ Parameters set up successfully\n');
    fprintf('  ✓ params.alg = %s\n', params.alg);
    fprintf('  ✓ params.init = %s\n', params.init);
    fprintf('  ✓ params.lambda = %g\n', params.lambda);
    
    % Test 3: Run onetrial
    fprintf('Test 3: Running onetrial...\n');
    [Error_Stand, Error_function] = onetrial(m, r, kappa, params);
    
    fprintf('  ✓ onetrial completed successfully\n');
    fprintf('  ✓ Error_Stand = %g\n', Error_Stand);
    fprintf('  ✓ Error_function = %g\n', Error_function);
    
    fprintf('\n=== ALL TESTS PASSED ===\n');
    
catch ME
    fprintf('\n=== TEST FAILED ===\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  File: %s, Function: %s, Line: %d\n', ...
                ME.stack(i).file, ME.stack(i).name, ME.stack(i).line);
    end
end
