%%%%%%%%%% Project Diagnostic Test
% Check if all required functions and dependencies are available

clear; clc;

fprintf('=== Project Diagnostic Test ===\n\n');

%% Test 1: Check if all utility functions exist
fprintf('1. Checking utility functions...\n');

required_functions = {
    'setup_experiment_params',
    'set_solver', 
    'set_init',
    'set_nonlinear',
    'generate_A',
    'groundtruth',
    'solve_GD',
    'solve_RGD', 
    'solve_SGD',
    'solve_SubGD'
};

missing_functions = {};
for i = 1:length(required_functions)
    func_name = required_functions{i};
    if exist(func_name, 'file') == 2
        fprintf('  ✓ %s found\n', func_name);
    else
        fprintf('  ✗ %s MISSING\n', func_name);
        missing_functions{end+1} = func_name;
    end
end

if isempty(missing_functions)
    fprintf('  All utility functions found!\n');
else
    fprintf('  WARNING: %d functions missing\n', length(missing_functions));
end

%% Test 2: Simple parameter setup test
fprintf('\n2. Testing parameter setup...\n');
try
    params = setup_experiment_params(10, 10, 2, 1, 0, 0, 1, 1, 10);
    fprintf('  ✓ Parameter setup successful\n');
    fprintf('  ✓ Algorithm: %s\n', params.alg_name);
    fprintf('  ✓ Init method: %s\n', params.init_name);
    fprintf('  ✓ Nonlinear: %s\n', params.nonlinear_name);
catch ME
    fprintf('  ✗ Parameter setup failed: %s\n', ME.message);
end

%% Test 3: Test solver functions directly
fprintf('\n3. Testing solver functions...\n');

% Create minimal test data
d1 = 5; d2 = 5; r = 2; m = 10; T = 5;
Xl = randn(d1, d2); 
Ul = randn(d1, r);
y = randn(m, 1);
A = randn(m, d1*d2);
Xstar = randn(d1, d2);

test_params = struct();
test_params.T = T;
test_params.mu = 0.1;
test_params.lambda = 0;
test_params.d1 = d1;
test_params.d2 = d2;

solvers = {'solve_GD', 'solve_RGD', 'solve_SGD', 'solve_SubGD'};
for i = 1:length(solvers)
    solver_name = solvers{i};
    try
        if exist(solver_name, 'file') == 2
            solver_func = str2func(solver_name);
            [~, ~] = solver_func(Xl, Ul, y, A, Xstar, d1, d2, test_params);
            fprintf('  ✓ %s works\n', solver_name);
        else
            fprintf('  ✗ %s not found\n', solver_name);
        end
    catch ME
        fprintf('  ✗ %s failed: %s\n', solver_name, ME.message);
    end
end

%% Test 4: Test onetrial function
fprintf('\n4. Testing onetrial function...\n');
try
    % Basic parameters for onetrial test
    m = 20; r = 2; kappa = 2; lambda = 0;
    
    % Setup minimal params
    params = struct();
    params.d1 = 10;
    params.d2 = 10;
    params.T = 5;
    params.verbose = 0;
    params.problem_flag = 0;
    params.init_flag = 1;
    params.init_scale = 1e-3;
    params.Xstar = randn(10, 10);
    
    % Add required function handles
    [~, alg_handle] = set_solver(1); % RGD
    params.alg = alg_handle;
    
    [~, init_handle] = set_init(1); % Random init
    params.init = init_handle;
    
    [~, nonlinear_handle] = set_nonlinear(0); % Identity
    params.nonlinear_func = nonlinear_handle;
    
    [Error_Stand, Error_function] = onetrial(m, r, kappa, lambda, params);
    fprintf('  ✓ onetrial works (final error: %.2e)\n', Error_Stand(end));
    
catch ME
    fprintf('  ✗ onetrial failed: %s\n', ME.message);
    if contains(ME.message, 'Undefined')
        fprintf('    This suggests a missing function dependency\n');
    end
end

%% Summary
fprintf('\n=== Diagnostic Summary ===\n');
if isempty(missing_functions)
    fprintf('✓ All core functions available\n');
    fprintf('✓ Ready to run full tests\n');
    fprintf('\nTry running: simple_test\n');
else
    fprintf('✗ Missing functions need to be addressed:\n');
    for i = 1:length(missing_functions)
        fprintf('  - %s\n', missing_functions{i});
    end
end
