%%%%%%%%%% Test Symmetric Fourth-Order Tensor Phase Retrieval
% Test PGD-Tensor solver: X = UU^T viewed as fourth-order tensor T = X ⊗ X
% Linear model: y_i = ⟨A_i ⊗ A_i, T⟩

clear; clc; close all;

fprintf('=== Fourth-Order Symmetric Tensor Phase Retrieval Test ===\n');
fprintf('Problem: X = UU^T, Tensor T = X ⊗ X\n\n');

%% Problem Setup
d = 20;                 % Matrix size (d x d, symmetric)
n = d * d;              % Total dimension
m = 200;                % Number of measurements
rank_true = 1;          % True rank
T = 100;                % Iterations

fprintf('Configuration:\n');
fprintf('  Matrix size: %dx%d (symmetric)\n', d, d);
fprintf('  Measurements: m = %d\n', m);
fprintf('  True rank: r = %d\n', rank_true);
fprintf('  Tensor dimension: %d^4 = %d\n', d, d^4);
fprintf('  Minimum measurements required: ~%d (r*(2d-r))\n', rank_true*(2*d-rank_true));
fprintf('  Iterations: %d\n\n', T);

%% Generate Symmetric Low-Rank Ground Truth: X = UU^T
fprintf('--- Generating Ground Truth ---\n');
U_true = randn(d, rank_true);
Xstar = U_true * U_true';  % Symmetric rank-r matrix
Xstar = Xstar / norm(Xstar, 'fro');  % Normalize

% Fourth-order tensor: T = X ⊗ X
% T_{ijkl} = X_{ij} * X_{kl}
tensor_Xstar = reshape(reshape(Xstar, n, 1) * reshape(Xstar, 1, n), [d, d, d, d]);

fprintf('Ground truth properties:\n');
fprintf('  Matrix X rank: %d (should be %d)\n', rank(Xstar), rank_true);
fprintf('  Matrix X norm: %.6f\n', norm(Xstar, 'fro'));
fprintf('  Symmetry error: %.2e\n', norm(Xstar - Xstar', 'fro'));

eigenvals_true = sort(eig(Xstar), 'descend');
fprintf('  Top %d eigenvalues: [', min(5, rank_true));
fprintf('%.4f ', eigenvals_true(1:min(5, rank_true)));
fprintf(']\n\n');

%% Generate Tensor Measurements (Linear Model)
fprintf('--- Generating Tensor Measurements ---\n');
fprintf('Creating measurement operators A_i ⊗ A_i...\n');

% Store measurement matrix A: size m x d^4
% Each row corresponds to A_i ⊗ A_i flattened
A = zeros(m, n*n);
for i = 1:m
    Ai = randn(d, d);
    Ai = (Ai + Ai')/2; % Symmetric measurement matrices
    % Fourth-order tensor A_i ⊗ A_i
    AiAi = reshape(Ai, n, 1) * reshape(Ai, 1, n); % d^2 x d^2
    A(i, :) = AiAi(:)'; % Flatten and store
end

% Define operators for fourth-order tensor
operator = struct();
operator.A = @(T) tensor_forward(T, A, d);
operator.A_star = @(z) tensor_adjoint(z, A, d);

% Generate measurements (linear model for tensor)
y = operator.A(tensor_Xstar) / sqrt(m);

fprintf('Measurement statistics:\n');
fprintf('  Range: [%.3f, %.3f]\n', min(y), max(y));
fprintf('  Mean: %.3f, Std: %.3f\n', mean(y), std(y));
fprintf('  Measurement matrix size: %d x %d\n\n', size(A));



%% Test Different Initializations
init_methods = {'zero'};
num_methods = length(init_methods);
results = cell(num_methods, 1);

for init_idx = 1:num_methods
    init_method = init_methods{init_idx};
    fprintf('\n========== Testing %s initialization ==========\n', upper(init_method));
    
   
   
    %% Setup PGD-Tensor Parameters
    % still use PGD, though for tensor
    params = struct();
    params.T = T;
    params.Xstar = tensor_Xstar;
    params.mu = 0.01;  % Step size for tensor formulation
    params.r = rank_true;
    params.verbose = 1;
    params.projection = @tensor_projection_rank_r;

    
     %% Initialize
    switch init_method
        case 'zero'
            fprintf('--- Zero Initialization ---\n');
            Xl_tensor_init = zeros(d, d, d, d); % Zero tensor
        case 'random'
            fprintf('--- Random Initialization ---\n');
            % Initialize tensor directly in 4D
            Xl_tensor_init = randn(d, d, d, d) * 0.01;
            % Apply tensor projection to ensure it has rank-r structure
            Xl_tensor_init = tensor_projection_rank_r(Xl_tensor_init, params);
    end
    
    % Extract matrix X from tensor (assuming T = X ⊗ X structure)
    % For analysis, we can approximate X from the leading mode
    params.Xl = Xl_tensor_init; % Pass tensor initialization
    
    init_error = norm(Xl_tensor_init(:) - tensor_Xstar(:), 'fro') / norm(tensor_Xstar(:), 'fro');
    fprintf('Initialization properties:\n');
    fprintf('  Tensor T: Dimension = %dx%dx%dx%d, Rel. Error = %.6e\n', d, d, d, d, init_error);
    
    %% Run PGD-Tensor Algorithm
    fprintf('--- Running PGD-Tensor ---\n');
    tic;
    [solver_output, Xl_tensor_final] = solve_PGD(Xl_tensor_init, [], y, operator, d, [], [], m, params);
    Error_Stand = solver_output.Error_Stand;
    Error_function = solver_output.Error_function;
    time_elapsed = toc;
    
    % Extract final matrix from tensor for analysis
    Xl_final = extract_matrix_from_tensor(Xl_tensor_final, params);
    
    %% Analyze Results
    fprintf('\n--- Results for %s initialization ---\n', init_method);
    [final_error, Xl_aligned] = rectify_sign_ambiguity(Xl_final, Xstar);
    
    fprintf('Recovery quality:\n');
    fprintf('  Final relative error: %.6e\n', final_error);
    fprintf('  Final loss: %.6e\n', Error_function(end));
    fprintf('  Computation time: %.3f seconds\n', time_elapsed);
    
    fprintf('\nRecovered matrix properties:\n');
    recovered_rank = rank(Xl_final, 1e-6);
    fprintf('  Rank: %d (true: %d, constraint: %d)\n', recovered_rank, rank_true, rank_true);
    fprintf('  Symmetry error: %.2e\n', norm(Xl_final - Xl_final', 'fro'));
    fprintf('  Frobenius norm: %.6f (true: %.6f)\n', norm(Xl_final, 'fro'), norm(Xstar, 'fro'));
    
    % Recovery status
    if final_error < 1e-2 && recovered_rank <= rank_true + 1
        fprintf('  Status: RECOVERED ✓\n');
        status = 'Success';
    else
        fprintf('  Status: Partial recovery\n');
        status = 'Partial';
    end
    
    %% Store Results
    results{init_idx} = struct();
    results{init_idx}.method = init_method;
    results{init_idx}.Error_Stand = Error_Stand;
    results{init_idx}.Error_function = Error_function;
    results{init_idx}.Xl_tensor_final = Xl_tensor_final;
    results{init_idx}.Xl_final = Xl_final;
    results{init_idx}.Xl_aligned = Xl_aligned;
    results{init_idx}.final_error = final_error;
    results{init_idx}.time = time_elapsed;
    results{init_idx}.recovered_rank = recovered_rank;
    results{init_idx}.status = status;
end

%% Comparison Summary
fprintf('\n========== Initialization Comparison ==========\n');
fprintf('Method      Final Error   Recovered Rank   Time (s)   Status\n');
fprintf('-------------------------------------------------------------\n');
for init_idx = 1:num_methods
    res = results{init_idx};
    fprintf('%-10s  %.4e     %-14d  %.3f      %s\n', ...
        res.method, res.final_error, res.recovered_rank, res.time, res.status);
end

% Find best result
[~, best_idx] = min(cellfun(@(x) x.final_error, results));
best_result = results{best_idx};
fprintf('\nBest initialization: %s\n', best_result.method);

%% Detailed Analysis for Best Result
fprintf('\n========== Detailed Analysis (Best: %s) ==========\n', best_result.method);

% Eigenvalue comparison
eigenvals_recovered = sort(eig(best_result.Xl_final), 'descend');

fprintf('\nEigenvalue comparison:\n');
fprintf('  True:      [');
fprintf('%.4f ', eigenvals_true(1:min(5, d)));
fprintf(']\n');
fprintf('  Recovered: [');
fprintf('%.4f ', eigenvals_recovered(1:min(5, d)));
fprintf(']\n');



%% Visualization
figure('Position', [100, 100, 1600, 1000]);

% Plot 1: Convergence Comparison
subplot(2, 4, 1);
hold on;
colors = {'b-', 'r-'};
for init_idx = 1:num_methods
    res = results{init_idx};
    semilogy(1:T, res.Error_Stand, colors{init_idx}, 'LineWidth', 2, 'DisplayName', res.method);
end
xlabel('Iteration');
ylabel('Relative Error');
title('Convergence Comparison');
legend('show');
grid on;

% Plot 2: Loss Comparison
subplot(2, 4, 2);
hold on;
for init_idx = 1:num_methods
    res = results{init_idx};
    semilogy(1:T, res.Error_function, colors{init_idx}, 'LineWidth', 2, 'DisplayName', res.method);
end
xlabel('Iteration');
ylabel('Loss');
title('Loss: (1/2m)||y - A(X⊗X)||^2');
legend('show');
grid on;

% Plot 3: Eigenvalue Spectrum
subplot(2, 4, 3);
stem(1:min(10, d), eigenvals_true(1:min(10, d)), 'b', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
stem(1:min(10, d), eigenvals_recovered(1:min(10, d)), 'r--', 'LineWidth', 2, 'DisplayName', 'Recovered');
xlabel('Index');
ylabel('Eigenvalue');
title('Eigenvalue Spectrum (Best Result)');
legend('Location', 'best');
grid on;

% Plot 4: Final Error Comparison
subplot(2, 4, 4);
final_errors = cellfun(@(x) x.final_error, results);
bar(final_errors);
set(gca, 'XTickLabel', init_methods);
ylabel('Final Relative Error');
title('Final Error by Initialization');
set(gca, 'YScale', 'log');
grid on;

% Plot 5: Ground Truth
subplot(2, 4, 5);
imagesc(Xstar);
colorbar;
title('Ground Truth X* (Symmetric)');
axis equal tight;

% Plot 6: Best Recovered Matrix
subplot(2, 4, 6);
imagesc(best_result.Xl_aligned);
colorbar;
title(sprintf('Best Recovered X (%s)\nError: %.2e', best_result.method, best_result.final_error));
axis equal tight;

% Plot 7: Recovery Error (Best)
subplot(2, 4, 7);
imagesc(best_result.Xl_aligned - Xstar);
colorbar;
title('Recovery Error: X - X*');
axis equal tight;

% Plot 8: Comparison of Both Results
subplot(2, 4, 8);
error_diff = abs(results{1}.Xl_aligned - results{2}.Xl_aligned);
imagesc(error_diff);
colorbar;
title(sprintf('Difference: |X_{%s} - X_{%s}|', init_methods{1}, init_methods{2}));
axis equal tight;

%% Save Results
fprintf('\n--- Saving Results ---\n');
save('symmetric_tensor_phase_retrieval_results.mat', 'results', 'Xstar', ...
     'eigenvals_true', 'eigenvals_recovered', 'params', ...
    'rank_true', 'd', 'm', 'T', 'tensor_Xstar');
fprintf('Results saved to: symmetric_tensor_phase_retrieval_results.mat\n');

fprintf('\n========== Test Complete ==========\n');

%% Helper Functions
function y = tensor_forward(T, A, d)
    % Forward operator: y = A(T) for 4D tensor T
    n = d * d;
    T_vec = reshape(T, [n * n, 1]);
    y = A * T_vec;
end

function T = tensor_adjoint(z, A, d)
    % Adjoint operator: T = A^*(z) returns 4D tensor
    T_vec = A' * z;
    T = reshape(T_vec, [d, d, d, d]);
end

