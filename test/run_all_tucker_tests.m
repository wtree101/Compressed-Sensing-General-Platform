%% Run All Tucker Tensor Tests
% This script runs all Tucker tensor tests in sequence and generates a summary report
%
% Tests included:
%   1. test_core_tensor_structure.m       - Mathematical proposition verification
%   2. test_tucker_spectral_symmetry.m    - Symmetric case + extraction methods
%   3. test_tucker_nonsymmetric.m         - Non-symmetric cases
%
% Date: 2025-12-23

clear; clc;

fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║         Tucker Tensor Test Suite - Full Run             ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');

% Initialize test results
test_results = struct();
test_names = {'Core Structure', 'Symmetric Extraction', 'Non-Symmetric'};
test_files = {'test_core_tensor_structure', 'test_tucker_spectral_symmetry', 'test_tucker_nonsymmetric'};
num_tests = length(test_files);

test_passed = false(num_tests, 1);
test_times = zeros(num_tests, 1);
test_errors = cell(num_tests, 1);

fprintf('Running %d test suites...\n\n', num_tests);
fprintf('═══════════════════════════════════════════════════════════\n\n');

%% Test 1: Core Tensor Structure
fprintf('[Test 1/%d] %s\n', num_tests, test_names{1});
fprintf('File: %s.m\n', test_files{1});
fprintf('─────────────────────────────────────────────────────────\n');

test_start = tic;
try
    run(test_files{1});
    test_passed(1) = true;
    test_errors{1} = '';
    fprintf('\n✓ Test 1 COMPLETED\n');
catch ME
    test_passed(1) = false;
    test_errors{1} = ME.message;
    fprintf('\n✗ Test 1 FAILED: %s\n', ME.message);
end
test_times(1) = toc(test_start);
fprintf('Time: %.2f seconds\n', test_times(1));
fprintf('\n═══════════════════════════════════════════════════════════\n\n');

%% Test 2: Symmetric Extraction Methods
fprintf('[Test 2/%d] %s\n', num_tests, test_names{2});
fprintf('File: %s.m\n', test_files{2});
fprintf('─────────────────────────────────────────────────────────\n');

test_start = tic;
try
    run(test_files{2});
    test_passed(2) = true;
    test_errors{2} = '';
    fprintf('\n✓ Test 2 COMPLETED\n');
catch ME
    test_passed(2) = false;
    test_errors{2} = ME.message;
    fprintf('\n✗ Test 2 FAILED: %s\n', ME.message);
end
test_times(2) = toc(test_start);
fprintf('Time: %.2f seconds\n', test_times(2));
fprintf('\n═══════════════════════════════════════════════════════════\n\n');

%% Test 3: Non-Symmetric Case
fprintf('[Test 3/%d] %s\n', num_tests, test_names{3});
fprintf('File: %s.m\n', test_files{3});
fprintf('─────────────────────────────────────────────────────────\n');

test_start = tic;
try
    run(test_files{3});
    test_passed(3) = true;
    test_errors{3} = '';
    fprintf('\n✓ Test 3 COMPLETED\n');
catch ME
    test_passed(3) = false;
    test_errors{3} = ME.message;
    fprintf('\n✗ Test 3 FAILED: %s\n', ME.message);
end
test_times(3) = toc(test_start);
fprintf('Time: %.2f seconds\n', test_times(3));
fprintf('\n═══════════════════════════════════════════════════════════\n\n');

%% Summary Report
total_time = sum(test_times);
num_passed = sum(test_passed);
num_failed = num_tests - num_passed;

fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║                    TEST SUMMARY REPORT                    ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');

fprintf('Test Results:\n');
fprintf('─────────────────────────────────────────────────────────\n');
for i = 1:num_tests
    status_str = test_passed(i) ? '✓ PASS' : '✗ FAIL';
    fprintf('  [%d] %-25s %s (%.2fs)\n', i, test_names{i}, status_str, test_times(i));
    if ~test_passed(i) && ~isempty(test_errors{i})
        fprintf('      Error: %s\n', test_errors{i});
    end
end
fprintf('─────────────────────────────────────────────────────────\n\n');

fprintf('Statistics:\n');
fprintf('  Total tests:    %d\n', num_tests);
fprintf('  Passed:         %d (%.1f%%)\n', num_passed, num_passed/num_tests*100);
fprintf('  Failed:         %d (%.1f%%)\n', num_failed, num_failed/num_tests*100);
fprintf('  Total time:     %.2f seconds\n', total_time);
fprintf('  Average time:   %.2f seconds per test\n', total_time/num_tests);
fprintf('\n');

% Overall verdict
if num_passed == num_tests
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('  ✓✓✓ ALL TESTS PASSED ✓✓✓\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
else
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('  ✗✗✗ SOME TESTS FAILED (%d/%d) ✗✗✗\n', num_failed, num_tests);
    fprintf('═══════════════════════════════════════════════════════════\n');
end

fprintf('\nTest suite completed at: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('\n');

%% Save results to file
results_file = 'tucker_test_results.mat';
save(results_file, 'test_passed', 'test_times', 'test_errors', 'test_names');
fprintf('Results saved to: %s\n\n', results_file);

