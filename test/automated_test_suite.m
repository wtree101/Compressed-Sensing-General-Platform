% Automated test suite that can be run with matlab -batch
% Usage: matlab -batch "automated_test_suite"

fprintf('=== Automated MATLAB Test Suite ===\n');

% Add paths
try
    addpath(pwd);
    addpath('utilities');
    addpath('solver');
    addpath('Initialization_groundtruth');
    fprintf('✓ Paths configured\n');
catch ME
    fprintf('✗ Path setup failed: %s\n', ME.message);
    exit(1);
end

% Test counter
tests_passed = 0;
total_tests = 2;

%% Test 1: Diagnostic Test
fprintf('\n1. Running diagnostic test...\n');
try
    diagnostic_test;
    fprintf('✓ Diagnostic test passed\n');
    tests_passed = tests_passed + 1;
catch ME
    fprintf('✗ Diagnostic test failed: %s\n', ME.message);
end

%% Test 2: Simple Algorithm Test
fprintf('\n2. Running simple algorithm test...\n');
try
    % Suppress plots for automated testing
    set(0, 'DefaultFigureVisible', 'off');
    simple_test;
    set(0, 'DefaultFigureVisible', 'on');
    fprintf('✓ Simple test passed\n');
    tests_passed = tests_passed + 1;
catch ME
    fprintf('✗ Simple test failed: %s\n', ME.message);
end

%% Summary
fprintf('\n=== Test Results Summary ===\n');
fprintf('Tests passed: %d/%d\n', tests_passed, total_tests);

if tests_passed == total_tests
    fprintf('🎉 All tests passed successfully!\n');
    fprintf('✓ Project is ready for use\n');
    exit(0);
else
    fprintf('❌ Some tests failed\n');
    fprintf('Check error messages above for debugging\n');
    exit(1);
end
