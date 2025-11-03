% test_tensor_U_GD_overparametrization.m
% Test gradient descent on U with overparameterization
% Try to recover rank r_star ground truth using rank r > r_star

clear; clc;
addpath('../solver');
addpath('../utilities');

fprintf('=== Testing Tensor U-GD Overparameterization ===\n\n');

%% Problem Setup
d = 40;              % Matrix dimension
r_star = 1;          % True rank
r_list = [1, 2, 3, 5, 7, 10,20,30,40];  % Test ranks (overparameterization when r > r_star)
m = 600;             % Number of measurements
T = 200;             % Iterations                                                                                              
mu = 0.01;           % Step size

rng(42);

% Ground truth (rank r_star = 1)
U_true = randn(d, r_star);
Xstar = U_true * U_true';
Xstar = Xstar / norm(Xstar, 'fro');

fprintf('Setup:\n');
fprintf('  Matrix dimension: d=%d\n', d);
fprintf('  True rank: r_star=%d\n', r_star);
fprintf('  Test ranks: [%s]\n', num2str(r_list));
fprintf('  Measurements: m=%d\n', m);
fprintf('  Iterations: T=%d\n\n', T);

%% Generate Measurements
fprintf('Generating measurements from rank-%d ground truth...\n', r_star);
A_cells = cell(m, 1);
y = zeros(m, 1);

for i = 1:m
    Ai = randn(d, d);
    Ai = (Ai + Ai') / 2;  % Symmetrize
    A_cells{i} = Ai;
    
    % Compute yᵢ = ⟨Aᵢ⊗Aᵢ, Xstar⊗Xstar⟩ = trace(AᵢXstarAᵢXstar)
    temp = Ai * Xstar;
    y(i) = trace(temp * temp) / sqrt(m);
end

fprintf('Measurements generated.\n');
fprintf('Measurement range: [%.2e, %.2e]\n\n', min(y), max(y));

%% Run experiments for different ranks
results = struct();
results.r_list = r_list;
results.final_errors = zeros(size(r_list));
results.final_losses = zeros(size(r_list));
results.histories = cell(size(r_list));
results.recovered_ranks = zeros(size(r_list));

fprintf('=== Running Experiments ===\n\n');

for idx = 1:length(r_list)
    r = r_list(idx);
    
    fprintf('Experiment %d/%d: r=%d %s\n', idx, length(r_list), r, ...
            ternary(r == r_star, '(exact)', sprintf('(over by %d)', r - r_star)));
    
    %% Initialize
    U0 = randn(d, r) * 0.05;
    X0 = U0 * U0';
    X0 = (X0 + X0') / 2;
    [init_error, ~] = rectify_sign_ambiguity(X0, Xstar);
    fprintf('  Initial error: %.6e\n', init_error);
    
    %% Run Solver
    params = struct();
    params.T = T;
    params.mu = mu;
    params.Xstar = Xstar;
    params.verbose = 0;
    
    tic;
    [U_final, X_final, history] = solve_tensor_U_GD(y, A_cells, U0, params);
    time_elapsed = toc;
    
    %% Analyze results
    [final_error, ~] = rectify_sign_ambiguity(X_final, Xstar);
    
    % Compute effective rank of recovered matrix
    [~, S, ~] = svd(X_final);
    singular_vals = diag(S);
    % Rank = number of singular values > threshold
    threshold = 1e-6 * singular_vals(1);
    recovered_rank = sum(singular_vals > threshold);
    
    % Store results
    results.final_errors(idx) = final_error;
    results.final_losses(idx) = history.loss(end);
    results.histories{idx} = history;
    results.recovered_ranks(idx) = recovered_rank;
    
    fprintf('  Final error: %.6e\n', final_error);
    fprintf('  Final loss: %.6e\n', history.loss(end));
    fprintf('  Recovered rank: %d (threshold=%.1e)\n', recovered_rank, threshold);
    fprintf('  Time: %.2f seconds\n', time_elapsed);
    fprintf('  Status: %s\n\n', ternary(final_error < 1e-4, '✓ Success', '✗ Failed'));
end

%% Visualization
figure('Position', [100, 100, 1400, 900]);

% Plot 1: Error vs Rank
subplot(2, 3, 1);
semilogy(r_list, results.final_errors, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot([r_star, r_star], [min(results.final_errors), max(results.final_errors)], ...
     'r--', 'LineWidth', 2);
xlabel('Rank r');
ylabel('Final Error');
title('Final Error vs Rank');
legend('Error', sprintf('True rank r*=%d', r_star), 'Location', 'best');
grid on;

% Plot 2: Loss vs Rank
subplot(2, 3, 2);
semilogy(r_list, results.final_losses, 'rs-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot([r_star, r_star], [min(results.final_losses), max(results.final_losses)], ...
     'r--', 'LineWidth', 2);
xlabel('Rank r');
ylabel('Final Loss');
title('Final Loss vs Rank');
legend('Loss', sprintf('True rank r*=%d', r_star), 'Location', 'best');
grid on;

% Plot 3: Recovered Rank vs Used Rank
subplot(2, 3, 3);
plot(r_list, results.recovered_ranks, 'go-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(r_list, r_list, 'k--', 'LineWidth', 1);
plot([r_star, r_star], [0, max(r_list)], 'r--', 'LineWidth', 2);
xlabel('Used Rank r');
ylabel('Recovered Rank (SVD)');
title('Effective Rank Recovery');
legend('Recovered', 'Used', sprintf('True r*=%d', r_star), 'Location', 'best');
grid on;
axis tight;

% Plot 4-6: Error Convergence for Selected Ranks
selected_idx = [1, min(3, length(r_list)), length(r_list)];
for plot_idx = 1:3
    subplot(2, 3, 3 + plot_idx);
    idx = selected_idx(plot_idx);
    r = r_list(idx);
    hist = results.histories{idx};
    
    semilogy(1:T, hist.errors, 'b-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Error');
    title(sprintf('r=%d %s', r, ternary(r == r_star, '(exact)', sprintf('(over +%d)', r - r_star))));
    grid on;
end

sgtitle(sprintf('Overparameterization Analysis: d=%d, r*=%d, m=%d', d, r_star, m));

%% Analysis Table
fprintf('=== Results Summary ===\n\n');
fprintf('┌──────┬──────────┬──────────┬──────────┬────────┐\n');
fprintf('│ Rank │  Final   │  Final   │ Recovered│ Status │\n');
fprintf('│  r   │  Error   │   Loss   │   Rank   │        │\n');
fprintf('├──────┼──────────┼──────────┼──────────┼────────┤\n');
for idx = 1:length(r_list)
    r = r_list(idx);
    status = ternary(results.final_errors(idx) < 1e-4, ' ✓ ', ' ✗ ');
    marker = ternary(r == r_star, '*', ' ');
    fprintf('│ %2d%s  │ %.2e │ %.2e │    %2d    │  %s   │\n', ...
            r, marker, results.final_errors(idx), results.final_losses(idx), ...
            results.recovered_ranks(idx), status);
end
fprintf('└──────┴──────────┴──────────┴──────────┴────────┘\n');
fprintf('* = true rank\n\n');

%% Key Observations
fprintf('=== Key Observations ===\n');

% Find best performing rank
[best_error, best_idx] = min(results.final_errors);
best_r = r_list(best_idx);

fprintf('1. Best performing rank: r=%d (error=%.2e)\n', best_r, best_error);

% Check if overparameterization helps or hurts
if best_r == r_star
    fprintf('2. Exact rank (r=r*) is optimal\n');
elseif best_r > r_star
    fprintf('2. Overparameterization helps: r=%d > r*=%d\n', best_r, r_star);
else
    fprintf('2. Underparameterization: r=%d < r*=%d (should not happen)\n', best_r, r_star);
end

% Check rank recovery
perfect_rank_recovery = all(results.recovered_ranks == r_star);
if perfect_rank_recovery
    fprintf('3. All methods recover correct rank r*=%d\n', r_star);
else
    fprintf('3. Rank recovery varies: [%s]\n', num2str(results.recovered_ranks));
end

% Check overparameterization degradation
if length(r_list) > 1
    error_increase = results.final_errors(end) / results.final_errors(1);
    if error_increase > 2
        fprintf('4. ⚠ Warning: High overparameterization (r=%d) degrades performance (%.1fx worse)\n', ...
                r_list(end), error_increase);
    elseif error_increase > 1.1
        fprintf('4. Mild degradation with high overparameterization (%.1fx worse)\n', error_increase);
    else
        fprintf('4. ✓ Robust to overparameterization (error ratio: %.2f)\n', error_increase);
    end
end

%% Convergence Speed Analysis
fprintf('\n=== Convergence Speed ===\n');
convergence_threshold = 1e-3;
for idx = 1:length(r_list)
    r = r_list(idx);
    hist = results.histories{idx};
    
    % Find iteration where error < threshold
    converged_iter = find(hist.errors < convergence_threshold, 1);
    if isempty(converged_iter)
        fprintf('r=%2d: Did not converge to %.0e\n', r, convergence_threshold);
    else
        fprintf('r=%2d: Converged to %.0e at iteration %d\n', r, convergence_threshold, converged_iter);
    end
end

fprintf('\n✓ Overparameterization Test Complete\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
