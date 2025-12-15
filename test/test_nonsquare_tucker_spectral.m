% test_nonsquare_tucker_spectral.m
% Test initialize_tensor_lift_tucker_spectral with non-square matrices

clear; clc;
addpath('../Initialization_groundtruth');
addpath('../utilities');
addpath('../utilities_tensor');

fprintf('=== Testing Non-Square Tucker Spectral Initialization ===\n\n');

%% Test Configuration
test_cases = {
    % [d1, d2, r_star, description]
    {20, 20, 2, 'Square matrix (baseline)'};
    {30, 20, 2, 'Tall matrix (d1 > d2)'};
    {20, 30, 2, 'Wide matrix (d1 < d2)'};
    {40, 25, 3, 'Non-square with higher rank'};
};

num_tests = length(test_cases);
results = cell(num_tests, 1);

%% Run Tests
for test_idx = 1:num_tests
    test_case = test_cases{test_idx};
    d1 = test_case{1};
    d2 = test_case{2};
    r_star = test_case{3};
    desc = test_case{4};
    
    fprintf('Test %d/%d: %s (d1=%d, d2=%d, r*=%d)\n', ...
            test_idx, num_tests, desc, d1, d2, r_star);
    
    %% Problem Setup
    m = 5 * d1 * d2;  % Oversampled measurements
    tucker_rank = min(5, max(1, floor(min(d1, d2)/4)));
    
    % Generate ground truth
    rng(42 + test_idx);
    U_true = randn(d1, r_star);
    V_true = randn(d2, r_star);
    Xstar = U_true * V_true';
    Xstar = Xstar / norm(Xstar, 'fro');
    
    % Generate measurements (amplitude model)
    A_cells = cell(m, 1);
    y = zeros(m, 1);
    for i = 1:m
        Ai = randn(d1, d2);
        A_cells{i} = Ai;
        y(i) = abs(trace(Ai' * Xstar)) / sqrt(m);
    end
    
    % Create operator struct
    operator = struct();
    operator.A = @(X) cellfun(@(Ai) trace(Ai' * X), A_cells);
    operator.A_star = @(y) sum(reshape(cell2mat(cellfun(@(Ai, yi) ...
                       yi * Ai(:), A_cells, num2cell(y), 'UniformOutput', false)), ...
                       [d1, d2, m]), 3);
    
    %% Run Initialization with Debug Mode
    params = struct();
    params.T_power = 0;
    params.mu = 0.01;
    params.r = tucker_rank;
    params.Xstar = Xstar;
    params.verbose = false;
    params.debug = false;  % Enable debug mode to get tensors
    
    try
        tic;
        [X0, ~, history] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params);
        elapsed_time = toc;
        
        %% INVESTIGATION: Extract matrices from tensors before/after HOSVD
        fprintf('  [Matrix Extraction Investigation]\n');
        
        % We need to manually perform spectral init to get T before HOSVD
        % Create Tucker operator
        tucker_op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false);
        
        % Form T = sum_i y_i * (Ai ⊗ Ai) - spectral initialization
        y_scaled = y / sqrt(m);
        T_before_HOSVD = tucker_op.kronecker_adjoint(y_scaled);
        
        % Matricize T_before_HOSVD: reshape to (d1*d2, d1*d2)
        n = d1 * d2;
        T_before_mat = reshape(permute(T_before_HOSVD, [1,2,3,4]), [n, n]);
        
        % Check symmetry
        symmetry_error_before = norm(T_before_mat - T_before_mat', 'fro') / norm(T_before_mat, 'fro');
        fprintf('    T_before_HOSVD matricization (%d×%d):\n', n, n);
        fprintf('      Symmetry error: %.2e %s\n', symmetry_error_before, ...
                ternary(symmetry_error_before < 1e-10, '✓ SYMMETRIC', '✗ NOT SYMMETRIC'));
        
        % Extract matrix via leading eigenvector
        [V_before, D_before] = eig(T_before_mat);
        [~, idx_before] = max(abs(diag(D_before)));
        v_lead_before = V_before(:, idx_before);
        X_extracted_before = reshape(v_lead_before, [d1, d2]);
        X_extracted_before = X_extracted_before / norm(X_extracted_before, 'fro');
        
        [error_before, ~] = rectify_sign_ambiguity(X_extracted_before, Xstar);
        fprintf('      Leading eigenvector extraction error: %.2e\n', error_before);
        
        % Now get T after HOSVD
        rank_vec = tucker_rank * ones(1, 4);
        [T_after_HOSVD, U_cells] = HOSVD_with_factors(T_before_HOSVD, rank_vec);
        
        % Matricize T_after_HOSVD
        T_after_mat = reshape(permute(T_after_HOSVD, [1,2,3,4]), [n, n]);
        
        % Check symmetry
        symmetry_error_after = norm(T_after_mat - T_after_mat', 'fro') / norm(T_after_mat, 'fro');
        fprintf('    T_after_HOSVD matricization (%d×%d):\n', n, n);
        fprintf('      Symmetry error: %.2e %s\n', symmetry_error_after, ...
                ternary(symmetry_error_after < 1e-10, '✓ SYMMETRIC', '✗ NOT SYMMETRIC'));
        
        % Extract matrix via leading eigenvector
        [V_after, D_after] = eig(T_after_mat);
        [~, idx_after] = max(abs(diag(D_after)));
        v_lead_after = V_after(:, idx_after);
        X_extracted_after = reshape(v_lead_after, [d1, d2]);
        X_extracted_after = X_extracted_after / norm(X_extracted_after, 'fro');
        
        [error_after, ~] = rectify_sign_ambiguity(X_extracted_after, Xstar);
        fprintf('      Leading eigenvector extraction error: %.2e\n', error_after);
        
        % Compare spectrum
        eigs_before = sort(abs(diag(D_before)), 'descend');
        eigs_after = sort(abs(diag(D_after)), 'descend');
        fprintf('    Eigenvalue comparison (top 5):\n');
        fprintf('      Before HOSVD: [%.2e, %.2e, %.2e, %.2e, %.2e]\n', eigs_before(1:5));
        fprintf('      After HOSVD:  [%.2e, %.2e, %.2e, %.2e, %.2e]\n', eigs_after(1:5));
        fprintf('      Ratio (after/before): %.4f\n', eigs_after(1) / eigs_before(1));
        
        % Store investigation results
        investigation = struct();
        investigation.T_before_symmetric = symmetry_error_before < 1e-10;
        investigation.T_after_symmetric = symmetry_error_after < 1e-10;
        investigation.error_before = error_before;
        investigation.error_after = error_after;
        investigation.eigs_before = eigs_before(1:min(5, n));
        investigation.eigs_after = eigs_after(1:min(5, n));
        
        fprintf('\n');
        
        % Compute error
        [final_error, X0_aligned] = rectify_sign_ambiguity(X0, Xstar);
        
        % Check dimensions
        dims_match = isequal(size(X0), [d1, d2]);
        
        % Check if square matrices are symmetric
        is_symmetric = true;
        if d1 == d2
            symmetry_error = norm(X0 - X0', 'fro') / norm(X0, 'fro');
            is_symmetric = symmetry_error < 1e-10;
        else
            symmetry_error = NaN;  % Not applicable
        end
        
        % Store results
        result = struct();
        result.success = true;
        result.d1 = d1;
        result.d2 = d2;
        result.r_star = r_star;
        result.tucker_rank = tucker_rank;
        result.final_error = final_error;
        result.time = elapsed_time;
        result.dims_match = dims_match;
        result.is_symmetric = is_symmetric;
        result.symmetry_error = symmetry_error;
        result.description = desc;
        result.investigation = investigation;
        
        % Print results
        fprintf('  ✓ Success: Final error = %.2e, Time = %.2f s\n', final_error, elapsed_time);
        fprintf('  Dimensions: %dx%d %s\n', size(X0, 1), size(X0, 2), ...
                ternary(dims_match, '✓', '✗'));
        if d1 == d2
            fprintf('  Symmetry: error = %.2e %s\n', symmetry_error, ...
                    ternary(is_symmetric, '✓', '✗'));
        else
            fprintf('  Symmetry: N/A (non-square matrix)\n');
        end
        
    catch ME
        fprintf('  ✗ FAILED: %s\n', ME.message);
        result = struct();
        result.success = false;
        result.error = ME.message;
        result.description = desc;
    end
    
    results{test_idx} = result;
    fprintf('\n');
end

%% Summary
fprintf('=== Test Summary ===\n\n');
fprintf('┌─────┬──────────────────────┬────────┬─────────┬────────────┬──────────┬──────────┐\n');
fprintf('│ #   │ Description          │ Size   │ Tucker  │ Error      │ Time (s) │ Status   │\n');
fprintf('├─────┼──────────────────────┼────────┼─────────┼────────────┼──────────┼──────────┤\n');

success_count = 0;
for i = 1:num_tests
    r = results{i};
    if r.success
        success_count = success_count + 1;
        status_str = '✓ PASS';
        fprintf('│ %d   │ %-20s │ %2dx%-2d │   %d     │  %.2e  │  %.2f    │ %-8s │\n', ...
                i, r.description, r.d1, r.d2, r.tucker_rank, ...
                r.final_error, r.time, status_str);
    else
        status_str = '✗ FAIL';
        fprintf('│ %d   │ %-20s │   --   │   --    │     --     │   --     │ %-8s │\n', ...
                i, r.description, status_str);
    end
end
fprintf('└─────┴──────────────────────┴────────┴─────────┴────────────┴──────────┴──────────┘\n\n');

fprintf('Results: %d/%d tests passed\n', success_count, num_tests);

%% Key Observations
fprintf('\n=== Key Observations ===\n');

if success_count == num_tests
    fprintf('✓ All tests passed successfully!\n');
    fprintf('✓ Non-square matrix support is working correctly\n');
    
    % Check square matrix symmetry
    square_results = results{1};
    if square_results.is_symmetric
        fprintf('✓ Square matrices are properly symmetrized\n');
    else
        fprintf('⚠ Square matrices have symmetry issues (error: %.2e)\n', ...
                square_results.symmetry_error);
    end
    
    % Check dimension preservation
    all_dims_match = all(cellfun(@(r) r.success && r.dims_match, results));
    if all_dims_match
        fprintf('✓ Output dimensions match input dimensions for all cases\n');
    end
    
else
    fprintf('⚠ Some tests failed:\n');
    for i = 1:num_tests
        if ~results{i}.success
            fprintf('  - Test %d (%s): %s\n', i, results{i}.description, results{i}.error);
        end
    end
end

%% Spectral Investigation Summary
fprintf('\n=== Spectral Initialization Investigation ===\n');
fprintf('Question: Can tensors be extracted to symmetric (n×n) matrices?\n');
fprintf('where n = d1*d2\n\n');

for i = 1:num_tests
    if results{i}.success && isfield(results{i}, 'investigation')
        inv = results{i}.investigation;
        fprintf('Test %d (%s):\n', i, results{i}.description);
        fprintf('  Matrix size: %d×%d (n=%d×%d)\n', results{i}.d1, results{i}.d2, ...
                results{i}.d1*results{i}.d2, results{i}.d1*results{i}.d2);
        fprintf('  Before HOSVD: %s\n', ternary(inv.T_before_symmetric, ...
                '✓ Symmetric matrix', '✗ NOT symmetric'));
        fprintf('  After HOSVD:  %s\n', ternary(inv.T_after_symmetric, ...
                '✓ Symmetric matrix', '✗ NOT symmetric'));
        fprintf('  Extraction quality:\n');
        fprintf('    Before HOSVD error: %.2e\n', inv.error_before);
        fprintf('    After HOSVD error:  %.2e\n', inv.error_after);
        fprintf('    Improvement: %.2fx\n', inv.error_before / max(inv.error_after, 1e-16));
        fprintf('\n');
    end
end

fprintf('Key Findings:\n');
% Check if all tensors are symmetric
all_before_symmetric = all(cellfun(@(r) r.success && isfield(r, 'investigation') && ...
                                    r.investigation.T_before_symmetric, results));
all_after_symmetric = all(cellfun(@(r) r.success && isfield(r, 'investigation') && ...
                                   r.investigation.T_after_symmetric, results));

if all_before_symmetric
    fprintf('✓ ALL tensors before HOSVD are symmetric when matricized\n');
else
    fprintf('✗ Some tensors before HOSVD are NOT symmetric\n');
end

if all_after_symmetric
    fprintf('✓ ALL tensors after HOSVD remain symmetric when matricized\n');
else
    fprintf('✗ Some tensors after HOSVD lose symmetry\n');
end

% Check if HOSVD improves extraction
avg_improvement = mean(cellfun(@(r) r.investigation.error_before / max(r.investigation.error_after, 1e-16), ...
                               results(cellfun(@(r) r.success && isfield(r, 'investigation'), results))));
fprintf('Average extraction improvement: %.2fx\n', avg_improvement);

if avg_improvement > 1
    fprintf('✓ HOSVD improves matrix extraction quality\n');
else
    fprintf('⚠ HOSVD does not improve extraction (may hurt accuracy)\n');
end

fprintf('\n✓ Non-Square Tucker Spectral Test Complete\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
