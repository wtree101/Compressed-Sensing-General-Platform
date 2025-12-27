%% Test: Core Projection Benefit Analysis (Simplified)
% Test whether projecting the core tensor to diagonal structure improves performance
% Uses solve_RGD_tucker_kronecker with use_core_projection flag

clear; clc;
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  Core Projection Benefit Test (Simplified)                ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

%% Test Configuration  
test_cases = {
    % Test 1: Standard case
    struct('name', 'Standard r=3 (with abs)', ...
           'd1', 20, 'd2', 30, 'r', 3, 'm', 800, ...
           'use_abs', true, 'mu', 0.1, 'T', 200, 'rng_seed', 40, ...
           'test_type', 'rgd_tucker');
    
    % % Test 2: Smaller m
    % struct('name', 'Small m r=3 (with abs)', ...
    %        'd1', 20, 'd2', 30, 'r', 3, 'm', 1000, ...
    %        'use_abs', true, 'mu', 0.1, 'T', 200, 'rng_seed', 40, ...
    %        'test_type', 'rgd_tucker');
    % 
    % % Test 3: Without abs
    % struct('name', 'Standard r=3 (NO abs)', ...
    %        'd1', 20, 'd2', 30, 'r', 4, 'm', 1200, ...
    %        'use_abs', false, 'mu', 0.1, 'T', 200, 'rng_seed', 40, ...
    %        'test_type', 'rgd_tucker');
    % 
    % Test 4: Init + RGD_amplitude (NO Tucker RGD)
    struct('name', 'Init + RGD_amplitude (r=3)', ...
           'd1', 20, 'd2', 30, 'r', 3, 'm', 1500, ...
           'use_abs', true, 'mu', 0.1, 'T', 200, 'rng_seed', 40, ...
           'test_type', 'famplitude');
    
    % % Test 5: Rank 1
    % struct('name', 'Rank r=1 (with abs)', ...
    %        'd1', 20, 'd2', 30, 'r', 1, 'm', 800, ...
    %        'use_abs', true, 'mu', 0.1, 'T', 200, 'rng_seed', 45, ...
    %        'test_type', 'rgd_tucker');
};

num_tests = length(test_cases);
results = cell(num_tests, 1);

%% Run All Test Cases
for test_idx = 1:num_tests
    tc = test_cases{test_idx};
    fprintf('\n');
    fprintf('════════════════════════════════════════════════════════════\n');
    fprintf('Test %d/%d: %s\n', test_idx, num_tests, tc.name);
    fprintf('  Config: d1=%d, d2=%d, r=%d, m=%d, use_abs=%s\n', ...
            tc.d1, tc.d2, tc.r, tc.m, mat2str(tc.use_abs));
    fprintf('════════════════════════════════════════════════════════════\n');
    
    % Run test
    result = run_single_test(tc);
    results{test_idx} = result;
    
    % Display summary
    fprintf('\n--- Results Summary ---\n');
    
    % Check if this is an amplitude test
    if isfield(result, 'test_type') && strcmp(result.test_type, 'amplitude')
        fprintf('Init + RGD_amplitude (T=%d iter):\n', tc.T);
        fprintf('  Final Error: %.6e, Final Loss: %.6e\n', ...
                result.no_proj.error_final, result.no_proj.loss_final);
        fprintf('  (No core projection tested - amplitude solver only)\n');
    else
        fprintf('WITHOUT Projection:\n');
        fprintf('  Final Error: %.6e, Final Loss: %.6e\n', ...
                result.no_proj.error_final, result.no_proj.loss_final);
        fprintf('WITH Projection:\n');
        fprintf('  Final Error: %.6e, Final Loss: %.6e\n', ...
                result.with_proj.error_final, result.with_proj.loss_final);
        fprintf('Improvement:\n');
        if result.with_proj.error_final < result.no_proj.error_final
            improvement = (result.no_proj.error_final - result.with_proj.error_final) / result.no_proj.error_final * 100;
            fprintf('  ✓ Error reduced by %.2f%%\n', improvement);
        else
            degradation = (result.with_proj.error_final - result.no_proj.error_final) / result.no_proj.error_final * 100;
            fprintf('  ✗ Error increased by %.2f%%\n', degradation);
        end
    end
end

%% Generate Comprehensive Comparison Report
fprintf('\n\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  Comprehensive Comparison Report                          ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% Create comparison table
fprintf('%-30s | %-15s | %-15s | %-15s\n', 'Test Case', 'No Proj Error', 'With Proj Error', 'Improvement');
fprintf('%-30s-|%-15s-|%-15s-|%-15s\n', repmat('-', 1, 30), repmat('-', 1, 15), repmat('-', 1, 15), repmat('-', 1, 15));

for test_idx = 1:num_tests
    tc = test_cases{test_idx};
    result = results{test_idx};
    
    error_no = result.no_proj.error_final;
    error_with = result.with_proj.error_final;
    
    % Check if this is an amplitude test
    if isfield(result, 'test_type') && strcmp(result.test_type, 'amplitude')
        improve_str = 'N/A (Ampl.)';
    elseif isnan(error_with)
        improve_str = 'N/A';
    elseif error_with < error_no
        improvement = (error_no - error_with) / error_no * 100;
        improve_str = sprintf('✓ %.1f%%', improvement);
    else
        degradation = (error_with - error_no) / error_no * 100;
        improve_str = sprintf('✗ +%.1f%%', degradation);
    end
    
    if isnan(error_with)
        fprintf('%-30s | %.6e    | %15s | %s\n', ...
                tc.name, error_no, 'N/A', improve_str);
    else
        fprintf('%-30s | %.6e    | %.6e    | %s\n', ...
                tc.name, error_no, error_with, improve_str);
    end
end

fprintf('\n');

%% Generate Visualization
fprintf('--- Generating Comparison Plots ---\n');
generate_comparison_plots(test_cases, results);
fprintf('✓ Plots generated\n\n');

%% Recommendations
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  Recommendations                                          ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% Count improvements
num_improved = 0;
num_degraded = 0;
avg_improvement = 0;
num_tucker_tests = 0;  % Count only Tucker RGD tests (exclude amplitude)

for test_idx = 1:num_tests
    result = results{test_idx};
    
    % Skip amplitude tests in improvement statistics
    if isfield(result, 'test_type') && strcmp(result.test_type, 'amplitude')
        continue;
    end
    
    num_tucker_tests = num_tucker_tests + 1;
    error_no = result.no_proj.error_final;
    error_with = result.with_proj.error_final;
    
    if ~isnan(error_with) && error_with < error_no
        num_improved = num_improved + 1;
        improvement = (error_no - error_with) / error_no * 100;
        avg_improvement = avg_improvement + improvement;
    elseif ~isnan(error_with)
        num_degraded = num_degraded + 1;
    end
end

if num_improved > 0
    avg_improvement = avg_improvement / num_improved;
end

fprintf('Summary:\n');
fprintf('  - Tucker RGD tests: %d (excluding amplitude tests)\n', num_tucker_tests);
fprintf('  - Core projection improved %d/%d test cases\n', num_improved, num_tucker_tests);
fprintf('  - Core projection degraded %d/%d test cases\n', num_degraded, num_tucker_tests);
if num_improved > 0
    fprintf('  - Average improvement when beneficial: %.1f%%\n', avg_improvement);
end

fprintf('\n');

if num_tucker_tests == 0
    fprintf('→ No Tucker RGD tests to evaluate core projection benefit.\n');
elseif num_improved >= num_tucker_tests * 0.7
    fprintf('✓✓✓ RECOMMENDATION: ENABLE core projection (use_core_projection=true)\n');
    fprintf('    Benefits observed in %.0f%% of test cases\n', num_improved/num_tucker_tests*100);
elseif num_improved > num_degraded
    fprintf('→ RECOMMENDATION: Consider enabling for specific scenarios\n');
    fprintf('    Particularly beneficial when m is small or without abs()\n');
else
    fprintf('✗ RECOMMENDATION: Keep core projection DISABLED (default)\n');
    fprintf('    No significant benefit or potential degradation\n');
end

fprintf('\n✓ Test Complete\n');

%% ========================================================================
%% Helper Functions
%% ========================================================================

function result = run_single_test(tc)
    % Run test with and without core projection using solver
    
    % Setup problem
    rng(tc.rng_seed);
    
    % Ground truth
    U1_true = randn(tc.d1, tc.r);
    U2_true = randn(tc.d2, tc.r);
    if tc.use_abs
        Xstar = abs(U1_true) * abs(U2_true)';
    else
        Xstar = U1_true * U2_true';
    end
    Xstar = Xstar / norm(Xstar, 'fro');
    
    % Measurement operator
    n = tc.d1 * tc.d2;
    A = randn(tc.m, n);
    operator = struct();
    operator.A = @(X) A * X(:);
    operator.A_star = @(y) reshape(A' * y, [tc.d1, tc.d2]);
    
    % Generate measurements
    y = abs(operator.A(Xstar)) / sqrt(tc.m);
    
    % Check test type (default to 'rgd_tucker' if not specified)
    if ~isfield(tc, 'test_type')
        tc.test_type = 'rgd_tucker';
    end
    
    if strcmp(tc.test_type, 'amplitude')
        % Test type: Initialization + 0 RGD + T iterations of solve_RGD_amplitude
        % This tests initialization quality without Tucker RGD refinement
        result = run_amplitude_test(tc, Xstar, operator, y, A);
    else
        % Default test type: Tucker RGD with/without core projection
        result = run_tucker_rgd_test(tc, Xstar, operator, y);
    end
end

function result = run_tucker_rgd_test(tc, Xstar, operator, y)
    % Run Tucker RGD test with and without core projection
    
    % Setup Tucker operator
    n = tc.d1 * tc.d2;
    A_matrix = zeros(tc.m, n);
    for j = 1:n
        e_j = zeros(n, 1);
        e_j(j) = 1;
        E_j = reshape(e_j, [tc.d1, tc.d2]);
        A_matrix(:, j) = operator.A(E_j);
    end
    
    A_cells = cell(tc.m, 1);
    for i = 1:tc.m
        Ai = reshape(A_matrix(i, :), [tc.d1, tc.d2]);
        A_cells{i} = Ai;
    end
    
    tucker_op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false);
    tucker_op.A_mat = A_matrix';
    
    % Spectral initialization
    dims = [tc.d1, tc.d2, tc.d1, tc.d2];
    T_tucker_init = TuckerTensor(dims, tc.r, 'symmetric', false, 'init_method', 'zeros');
    
    spectral_operator = struct();
    spectral_operator.A_cells = A_cells;
    y_spectral = y.^2 * sqrt(tc.m);
    
    [U_cell_init, G_init] = T_tucker_init.initialize_spectral(spectral_operator, y_spectral, tc.m);
    
    for k = 1:4
        T_tucker_init.U{k} = U_cell_init{k};
    end
    T_tucker_init.G = G_init;
    
    % Test WITHOUT projection
    fprintf('  Testing WITHOUT core projection...\n');
    T_tucker_no = T_tucker_init.copy();
    params_no = struct('T', tc.T, 'mu', tc.mu, 'Xstar', Xstar, 'verbose', false, 'use_core_projection', false);
    [output_no, ~] = solve_RGD_tucker_kronecker(T_tucker_no, [], y_spectral, tucker_op, [], [], [], tc.m, params_no);
    
    result_no = struct();
    result_no.errors = output_no.Error_Stand;
    result_no.losses = output_no.Error_function;
    result_no.error_final = output_no.Error_Stand(end);
    result_no.loss_final = output_no.Error_function(end);
    
    % Test WITH projection
    fprintf('  Testing WITH core projection...\n');
    T_tucker_with = T_tucker_init.copy();
    params_with = struct('T', tc.T, 'mu', tc.mu, 'Xstar', Xstar, 'verbose', false, 'use_core_projection', true);
    [output_with, ~] = solve_RGD_tucker_kronecker(T_tucker_with, [], y_spectral, tucker_op, [], [], [], tc.m, params_with);
    
    result_with = struct();
    result_with.errors = output_with.Error_Stand;
    result_with.losses = output_with.Error_function;
    result_with.error_final = output_with.Error_Stand(end);
    result_with.loss_final = output_with.Error_function(end);
    
    % Package results
    result = struct();
    result.no_proj = result_no;
    result.with_proj = result_with;
    result.test_config = tc;
end

function result = run_amplitude_test(tc, Xstar, operator, y, A)
    % Run amplitude test: Initialization + 0 Tucker RGD + T iterations of solve_RGD_amplitude
    % This provides a baseline comparison for initialization quality
    
    fprintf('  Testing Initialization + RGD_amplitude (NO Tucker RGD, NO core projection effect)...\n');
    
    % Perform spectral initialization using initialize_tensor_lift_tucker_spectral
    % with T=0 (no Tucker RGD iterations)
    init_params = struct();
    init_params.r = tc.r;
    init_params.T = 0;  % NO Tucker RGD iterations
    init_params.mu = tc.mu;
    init_params.verbose = false;
    
    % Call initialization function (similar to Phasediagram_tensor_nonsym.m)
    [X_init, ~, ~] = initialize_tensor_lift_tucker_spectral(y, operator, tc.d1, tc.d2, init_params);
    
    % Now run solve_RGD_amplitude for T iterations
    amplitude_params = struct();
    amplitude_params.T = tc.T;
    amplitude_params.mu = tc.mu;
    amplitude_params.r = tc.r;
    amplitude_params.Xstar = Xstar;
    amplitude_params.verbose = false;
    amplitude_params.use_preconditioner = true;  % Use preconditioned RGD
    
    fprintf('  Running solve_RGD_amplitude for %d iterations...\n', tc.T);
    [output_amplitude, ~] = solve_RGD_amplitude(X_init, [], y, operator, tc.d1, tc.d2, tc.r, tc.m, amplitude_params);
    
    % Package results (use same structure as Tucker RGD for compatibility)
    % For amplitude test, "no_proj" represents the result (no projection is used)
    result_amplitude = struct();
    result_amplitude.errors = output_amplitude.Error_Stand;
    result_amplitude.losses = output_amplitude.Error_function;
    result_amplitude.error_final = output_amplitude.Error_Stand(end);
    result_amplitude.loss_final = output_amplitude.Error_function(end);
    
    % For amplitude test, "with_proj" is not applicable, so copy the same result
    % or set to NaN to indicate not tested
    result_no_test = struct();
    result_no_test.errors = nan(size(result_amplitude.errors));
    result_no_test.losses = nan(size(result_amplitude.losses));
    result_no_test.error_final = NaN;
    result_no_test.loss_final = NaN;
    
    result = struct();
    result.no_proj = result_amplitude;  % Amplitude solver result
    result.with_proj = result_no_test;  % Not tested for amplitude
    result.test_config = tc;
    result.test_type = 'amplitude';  % Mark as amplitude test
end

function generate_comparison_plots(test_cases, results)
    num_tests = length(test_cases);
    
    % Create figure with subplots
    figure('Position', [100, 100, 1800, 1000]);
    
    % Error convergence plots
    for test_idx = 1:min(num_tests, 6)  % Show up to 6 tests
        subplot(2, 3, test_idx);
        
        result = results{test_idx};
        tc = test_cases{test_idx};
        
        % Check if this is an amplitude test
        if isfield(result, 'test_type') && strcmp(result.test_type, 'amplitude')
            % Only plot the amplitude result
            semilogy(result.no_proj.errors, 'b-', 'LineWidth', 2, 'DisplayName', 'Init + RGD\_amplitude');
            
            xlabel('Iteration');
            ylabel('Relative Error');
            title(sprintf('%s\n(m=%d, r=%d, abs=%s)', tc.name, tc.m, tc.r, mat2str(tc.use_abs)));
            legend('Location', 'best');
            grid on;
        else
            % Plot Tucker RGD with/without projection
            semilogy(result.no_proj.errors, 'b-', 'LineWidth', 2, 'DisplayName', 'No Projection');
            hold on;
            
            if ~isnan(result.with_proj.error_final)
                semilogy(result.with_proj.errors, 'r--', 'LineWidth', 2, 'DisplayName', 'With Projection');
            end
            
            xlabel('Iteration');
            ylabel('Relative Error');
            title(sprintf('%s\n(m=%d, r=%d, abs=%s)', tc.name, tc.m, tc.r, mat2str(tc.use_abs)));
            legend('Location', 'best');
            grid on;
        end
    end
    
    sgtitle('Error Convergence Comparison: Core Projection Impact');
    
    % Summary bar chart
    figure('Position', [100, 100, 1400, 600]);
    
    subplot(1, 2, 1);
    final_errors_no = zeros(num_tests, 1);
    final_errors_with = zeros(num_tests, 1);
    test_labels = cell(num_tests, 1);
    
    for i = 1:num_tests
        final_errors_no(i) = results{i}.no_proj.error_final;
        final_errors_with(i) = results{i}.with_proj.error_final;
        
        % Mark amplitude tests in labels
        if isfield(results{i}, 'test_type') && strcmp(results{i}.test_type, 'amplitude')
            test_labels{i} = [test_cases{i}.name, ' (Ampl.)'];
        else
            test_labels{i} = test_cases{i}.name;
        end
    end
    
    x = 1:num_tests;
    bar_data = [final_errors_no, final_errors_with];
    bar_data(isnan(bar_data)) = 0;  % Set NaN to 0 for plotting
    bar(x, bar_data);
    set(gca, 'XTickLabel', test_labels);
    xtickangle(45);
    ylabel('Final Error');
    title('Final Error Comparison');
    legend('No Projection / Amplitude', 'With Projection', 'Location', 'best');
    grid on;
    set(gca, 'YScale', 'log');
    
    subplot(1, 2, 2);
    improvements = (final_errors_no - final_errors_with) ./ final_errors_no * 100;
    
    % Separate Tucker RGD tests and amplitude tests
    is_amplitude = false(num_tests, 1);
    for i = 1:num_tests
        if isfield(results{i}, 'test_type') && strcmp(results{i}.test_type, 'amplitude')
            is_amplitude(i) = true;
            improvements(i) = NaN;  % Don't show improvement for amplitude
        end
    end
    
    bar(x, improvements);
    hold on;
    yline(0, 'k--', 'LineWidth', 1.5);
    set(gca, 'XTickLabel', test_labels);
    xtickangle(45);
    ylabel('Improvement (%)');
    title('Error Improvement with Core Projection');
    grid on;
    
    % Add text annotation for amplitude tests
    for i = 1:num_tests
        if is_amplitude(i)
            text(i, 0, 'N/A', 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end
