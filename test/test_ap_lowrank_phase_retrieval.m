%%%%%%%%%% Test Script for AP Low-Rank Phase Retrieval
% Test AP algorithm for real low-rank matrix phase retrieval problem

clear; clc; close all;

fprintf('=== AP Low-Rank Phase Retrieval Test ===\n');

%% Problem Setup for Low-Rank Phase Retrieval
% Matrix parameters
d1 = 30; d2 = 30;       % Matrix dimensions
n = d1 * d2;            % Total matrix dimension
m = 400;                % Number of measurements (should be > rank * (d1+d2-rank))
rank_true = 1;          % True rank of the matrix

% Algorithm parameters
T = 500;                % Number of iterations
verbose = 1;            % Show progress

fprintf('Problem Configuration:\n');
fprintf('  Matrix size: %dx%d (n=%d)\n', d1, d2, n);
fprintf('  Measurements: %d\n', m);
fprintf('  True rank: %d\n', rank_true);
fprintf('  Iterations: %d\n', T);

%% Generate Low-Rank Ground Truth Matrix
% Create low-rank matrix as product of two random matrices
U_true = randn(d1, rank_true);
V_true = randn(d2, rank_true);
Xstar = U_true * V_true';
Xstar = abs(Xstar);
Xstar = Xstar / norm(Xstar, 'fro'); % Normalize

fprintf('Ground truth rank: %d (should be %d)\n', rank(Xstar), rank_true);
fprintf('Ground truth Frobenius norm: %.6f\n', norm(Xstar, 'fro'));

%% Generate Random Gaussian Measurements
% Create random real measurement matrix
A = randn(m, n) ;

% Define operators
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(z) reshape(A' * z, [d1, d2]);

% Generate measurements (magnitude-only measurements for phase retrieval)
z_true = operator.A(Xstar)/sqrt(m);
y = abs(z_true); % Phase retrieval: only magnitude measurements available

fprintf('Measurement range: [%.3f, %.3f]\n', min(y), max(y));

%% Test Different Rank Constraints
test_ranks = [rank_true, rank_true*2,rank_true*10 ]; % Test different rank constraints
num_ranks = length(test_ranks);

results = cell(num_ranks, 1);

for rank_idx = 1:num_ranks
    test_rank = test_ranks(rank_idx);
    fprintf('\n--- Testing rank constraint r=%d ---\n', test_rank);
    
   
    
    %% Setup Init and AP Parameters
    params = struct();
    params.T = T;
    params.Xstar = Xstar;
    params.nonlinear_func = @(z) abs(z); % Phase retrieval: observe magnitudes only
    params.r = test_rank; % Rank constraint
    
    % Low-rank projection function
    params.projection = @(X, p) project_low_rank(X, p.r);
    
    % PGD parameters for subproblem
    params.mu = 0.01;
    params.lambda = 0;

     %% Initialize with Power Method
    T_power = 10; % Number of power iterations
    [Xl_init, ~] = initialize_power_method(y, operator, d1, d2, T_power, params);
    
    % Scale the initialization
    % Xl_init = Xl_init * norm(y, 'fro') / norm(operator.A(Xl_init), 'fro');
    
    %% Run AP Algorithm
    tic;
    [solver_output, Xl_final] = solve_AP(Xl_init, [], y, operator, d1, d2, [], [], params);
    Error_Stand = solver_output.Error_Stand;
    Error_function = solver_output.Error_function;
    elapsed_time = toc;
    
    %% Store Results
    results{rank_idx} = struct();
    results{rank_idx}.test_rank = test_rank;
    results{rank_idx}.Error_Stand = Error_Stand;
    results{rank_idx}.Error_function = Error_function;
    results{rank_idx}.Xl_final = Xl_final;
    results{rank_idx}.final_error = Error_Stand(end);
    results{rank_idx}.time = elapsed_time;
    
    % Check recovery quality
    recovered_rank = rank(Xl_final, 1e-6);
    nuclear_norm_error = abs(norm(Xl_final, 'fro') - norm(Xstar, 'fro')) / norm(Xstar, 'fro');
    
    fprintf('  Final relative error: %.6f\n', Error_Stand(end));
    fprintf('  Recovered rank: %d (true: %d, constraint: %d)\n', recovered_rank, rank_true, test_rank);
    fprintf('  Nuclear norm error: %.6f\n', nuclear_norm_error);
    fprintf('  Computation time: %.3f seconds\n', elapsed_time);
    
    % Store additional metrics
    results{rank_idx}.recovered_rank = recovered_rank;
    results{rank_idx}.nuclear_norm_error = nuclear_norm_error;
    
    % Check convergence
    if Error_Stand(end) < 1e-2 && recovered_rank <= test_rank
        fprintf('  Status: RECOVERED ✓\n');
        results{rank_idx}.status = 'Success';
    else
        fprintf('  Status: Not recovered\n');
        results{rank_idx}.status = 'Failed';
    end
end

%% Display Comparison
fprintf('\n=== Rank Constraint Comparison ===\n');
fprintf('Rank Constraint   Final Error   Recovered Rank   Nuclear Error   Time (s)   Status\n');
fprintf('--------------------------------------------------------------------------\n');

for rank_idx = 1:num_ranks
    res = results{rank_idx};
    fprintf('r=%-13d  %.4e     %-14d  %.4e       %.3f      %s\n', ...
            res.test_rank, res.final_error, res.recovered_rank, ...
            res.nuclear_norm_error, res.time, res.status);
end

%% Detailed Analysis for Best Result
[~, best_idx] = min(cellfun(@(x) x.final_error, results));
best_result = results{best_idx};

fprintf('\n=== Best Result Analysis (r=%d constraint) ===\n', best_result.test_rank);

% Compare true and recovered matrices
Xl_best = best_result.Xl_final;
fprintf('Matrix comparison:\n');
fprintf('  True matrix norm: %.6f\n', norm(Xstar, 'fro'));
fprintf('  Recovered matrix norm: %.6f\n', norm(Xl_best, 'fro'));
fprintf('  Relative error: %.6f\n', norm(Xl_best - Xstar, 'fro') / norm(Xstar, 'fro'));
fprintf('  True rank: %d, Recovered rank: %d\n', rank(Xstar), rank(Xl_best, 1e-6));

% Singular values comparison
[~, S_true, ~] = svd(Xstar);
[~, S_recovered, ~] = svd(Xl_best);
fprintf('  True singular values: [');
fprintf('%.3f ', diag(S_true(1:min(5, end), 1:min(5, end))));
fprintf(']\n');
fprintf('  Recovered singular values: [');
fprintf('%.3f ', diag(S_recovered(1:min(5, end), 1:min(5, end))));
fprintf(']\n');

%% Plot Results
figure('Position', [100, 100, 1500, 1200]);

% Plot 1: Convergence curves
subplot(3, 3, 1);
hold on;
colors = ['b', 'r', 'g', 'm', 'c'];
for rank_idx = 1:num_ranks
    res = results{rank_idx};
    semilogy(1:T, res.Error_Stand, colors(rank_idx), 'LineWidth', 2, ...
             'DisplayName', sprintf('r=%d', res.test_rank));
end
xlabel('Iteration');
ylabel('Relative Error');
title('Convergence Comparison');
legend('show');
grid on;

% Plot 2: Function error
subplot(3, 3, 2);
hold on;
for rank_idx = 1:num_ranks
    res = results{rank_idx};
    semilogy(1:T, res.Error_function, colors(rank_idx), 'LineWidth', 2, ...
             'DisplayName', sprintf('r=%d', res.test_rank));
end
xlabel('Iteration');
ylabel('Function Error');
title('Function Error');
legend('show');
grid on;

% Plot 3: Final error vs rank constraint
subplot(3, 3, 3);
final_errors = cellfun(@(x) x.final_error, results);
bar(final_errors);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('r=%d', x), test_ranks, 'UniformOutput', false));
ylabel('Final Relative Error');
title('Final Error vs Rank');
set(gca, 'YScale', 'log');
grid on;

% Plot 4: True matrix
subplot(3, 3, 4);
imagesc(Xstar);
colorbar;
title('True Low-Rank Matrix');
axis equal tight;

% Plot 5: Best recovered matrix
subplot(3, 3, 5);
imagesc(Xl_best);
colorbar;
title(['Recovered Matrix (r=', num2str(best_result.test_rank), ')']);
axis equal tight;

% Plot 6: Difference
subplot(3, 3, 6);
imagesc(Xl_best - Xstar);
colorbar;
title('Recovery Error');
axis equal tight;



% Plot 9: Recovered rank
subplot(3, 3, 9);
recovered_ranks = cellfun(@(x) x.recovered_rank, results);
bar(recovered_ranks);
hold on;
plot(1:num_ranks, rank_true * ones(num_ranks, 1), 'r--', 'LineWidth', 2, 'DisplayName', 'True Rank');
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('r=%d', x), test_ranks, 'UniformOutput', false));
ylabel('Recovered Rank');
title('Recovered Rank vs Constraint');
legend('show');
grid on;

%% Save Results
save('ap_lowrank_phase_retrieval_test_results.mat', 'results', 'Xstar', 'y', 'params', ...
     'rank_true', 'd1', 'd2', 'm', 'T', 'test_ranks');

fprintf('\n=== Test Complete ===\n');
fprintf('Results saved to: ap_lowrank_phase_retrieval_test_results.mat\n');

%% Helper Function: Low-Rank Projection
function X_proj = project_low_rank(X, r)
    % Project onto the set of rank-r matrices using SVD
    % Keep only the top r singular values
    
    if r <= 0 || r >= min(size(X))
        % No projection needed
        X_proj = X;
        return;
    end
    
    % SVD-based projection
    [U, S, V] = svd(X);
    
    % Keep only top r singular values
    S_proj = S;
    S_proj(r+1:end, r+1:end) = 0;
    
    % Reconstruct low-rank matrix
    X_proj = U * S_proj * V';
end
