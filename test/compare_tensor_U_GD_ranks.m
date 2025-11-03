% compare_tensor_U_GD_ranks.m
% Compare performance across different ranks for tensor U-GD
% Focus on overparameterization: recovering r_star=1 with various r

clear; clc;
addpath('../solver');
addpath('../utilities');

fprintf('=== Tensor U-GD: Rank Comparison Study ===\n\n');

%% Configuration
configs = {
    % [d, r_star, m, description]
    {20, 1, 400, 'Small: d=20, r*=1, m=400'};
    {20, 1, 800, 'Small: d=20, r*=1, m=800'};
    {40, 1, 800, 'Medium: d=40, r*=1, m=800'};
    {40, 1, 1600, 'Medium: d=40, r*=1, m=1600'};
};

config_idx = 3;  % Choose which configuration to test
config = configs{config_idx};

d = config{1};
r_star = config{2};
m = config{3};
desc = config{4};

T = 150;             % Iterations
mu = 0.01;           % Step size
num_trials = 3;      % Number of random trials per rank

% Rank grid: test from r_star to overparameterized
r_list = [1, 2, 3, 4, 5, 7, 10];
r_list = r_list(r_list <= min(d/2, 10));  % Cap at d/2 for stability

fprintf('Configuration: %s\n', desc);
fprintf('Rank grid: [%s]\n', num2str(r_list));
fprintf('Trials per rank: %d\n', num2str(num_trials));
fprintf('Iterations: %d, Step size: %.4f\n\n', T, mu);

%% Storage
results = struct();
results.config = desc;
results.r_list = r_list;
results.errors_mean = zeros(size(r_list));
results.errors_std = zeros(size(r_list));
results.losses_mean = zeros(size(r_list));
results.ranks_mean = zeros(size(r_list));
results.all_trials = cell(size(r_list));

%% Run Experiments
fprintf('=== Running Experiments ===\n\n');

for r_idx = 1:length(r_list)
    r = r_list(r_idx);
    
    fprintf('Rank r=%d (%d/%d)\n', r, r_idx, length(r_list));
    
    trial_errors = zeros(num_trials, 1);
    trial_losses = zeros(num_trials, 1);
    trial_ranks = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        %% Generate problem
        rng(100 + r_idx * 10 + trial);  % Different seed per trial
        
        % Ground truth
        U_true = randn(d, r_star);
        Xstar = U_true * U_true';
        Xstar = Xstar / norm(Xstar, 'fro');
        
        % Measurements
        A_cells = cell(m, 1);
        y = zeros(m, 1);
        for i = 1:m
            Ai = randn(d, d);
            Ai = (Ai + Ai') / 2;
            A_cells{i} = Ai;
            temp = Ai * Xstar;
            y(i) = trace(temp * temp) / sqrt(m);
        end
        
        % Initialize
        U0 = randn(d, r) * 0.05;
        
        % Solve
        params = struct('T', T, 'mu', mu, 'Xstar', Xstar, 'verbose', 0);
        [U_final, X_final, history] = solve_tensor_U_GD(y, A_cells, U0, params);
        
        % Record results
        [final_error, ~] = rectify_sign_ambiguity(X_final, Xstar);
        trial_errors(trial) = final_error;
        trial_losses(trial) = history.loss(end);
        
        % Effective rank
        [~, S, ~] = svd(X_final);
        sv = diag(S);
        recovered_rank = sum(sv > 1e-6 * sv(1));
        trial_ranks(trial) = recovered_rank;
    end
    
    % Aggregate statistics
    results.errors_mean(r_idx) = mean(trial_errors);
    results.errors_std(r_idx) = std(trial_errors);
    results.losses_mean(r_idx) = mean(trial_losses);
    results.ranks_mean(r_idx) = mean(trial_ranks);
    results.all_trials{r_idx} = struct('errors', trial_errors, 'losses', trial_losses, 'ranks', trial_ranks);
    
    fprintf('  Mean error: %.2e ± %.2e\n', results.errors_mean(r_idx), results.errors_std(r_idx));
    fprintf('  Mean loss: %.2e\n', results.losses_mean(r_idx));
    fprintf('  Mean rank: %.1f\n', results.ranks_mean(r_idx));
    
    % Check convergence
    success_rate = sum(trial_errors < 1e-3) / num_trials;
    fprintf('  Success rate: %.0f%% (error < 1e-3)\n', success_rate * 100);
    fprintf('\n');
end

%% Visualization
figure('Position', [100, 100, 1400, 800]);

% Plot 1: Mean Error vs Rank (with error bars)
subplot(2, 3, 1);
errorbar(r_list, results.errors_mean, results.errors_std, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot([r_star, r_star], ylim, 'r--', 'LineWidth', 2);
set(gca, 'YScale', 'log');
xlabel('Rank r');
ylabel('Final Error (mean ± std)');
title('Error vs Rank');
legend('Error', sprintf('True r*=%d', r_star), 'Location', 'best');
grid on;

% Plot 2: Loss vs Rank
subplot(2, 3, 2);
semilogy(r_list, results.losses_mean, 'rs-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot([r_star, r_star], ylim, 'r--', 'LineWidth', 2);
xlabel('Rank r');
ylabel('Final Loss (mean)');
title('Loss vs Rank');
legend('Loss', sprintf('True r*=%d', r_star), 'Location', 'best');
grid on;

% Plot 3: Recovered Rank
subplot(2, 3, 3);
plot(r_list, results.ranks_mean, 'go-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(r_list, r_list, 'k--', 'LineWidth', 1);
plot([r_star, r_star], [0, max(r_list)], 'r--', 'LineWidth', 2);
xlabel('Used Rank r');
ylabel('Recovered Rank (mean)');
title('Effective Rank Recovery');
legend('Recovered', 'Used', sprintf('True r*=%d', r_star), 'Location', 'best');
grid on;

% Plot 4: Box plot of errors
subplot(2, 3, 4);
all_errors = [];
all_groups = [];
for r_idx = 1:length(r_list)
    all_errors = [all_errors; results.all_trials{r_idx}.errors];
    all_groups = [all_groups; r_list(r_idx) * ones(num_trials, 1)];
end
boxplot(log10(all_errors), all_groups);
xlabel('Rank r');
ylabel('log10(Error)');
title('Error Distribution (Box Plot)');
grid on;

% Plot 5: Success rate
subplot(2, 3, 5);
success_rates = zeros(size(r_list));
for r_idx = 1:length(r_list)
    success_rates(r_idx) = sum(results.all_trials{r_idx}.errors < 1e-3) / num_trials;
end
bar(r_list, success_rates * 100);
hold on;
plot([r_star, r_star], [0, 100], 'r--', 'LineWidth', 2);
xlabel('Rank r');
ylabel('Success Rate (%)');
title('Success Rate (error < 1e-3)');
ylim([0, 105]);
grid on;

% Plot 6: Error degradation from optimal
subplot(2, 3, 6);
[min_error, opt_idx] = min(results.errors_mean);
degradation = results.errors_mean / min_error;
semilogy(r_list, degradation, 'mo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot([r_star, r_star], ylim, 'r--', 'LineWidth', 2);
plot(xlim, [1, 1], 'k--', 'LineWidth', 1);
xlabel('Rank r');
ylabel('Error Ratio (vs optimal)');
title(sprintf('Degradation (optimal: r=%d)', r_list(opt_idx)));
legend('Ratio', sprintf('True r*=%d', r_star), 'Optimal', 'Location', 'best');
grid on;

sgtitle(sprintf('Rank Comparison: %s, %d trials/rank', desc, num_trials));

%% Summary Table
fprintf('=== Summary Table ===\n\n');
fprintf('Configuration: %s\n', desc);
fprintf('True rank: r*=%d\n\n', r_star);

fprintf('┌──────┬────────────────┬────────────────┬──────────┬─────────┐\n');
fprintf('│ Rank │  Mean Error    │  Std Error     │ Mean Rank│ Success │\n');
fprintf('├──────┼────────────────┼────────────────┼──────────┼─────────┤\n');
for r_idx = 1:length(r_list)
    r = r_list(r_idx);
    marker = ternary(r == r_star, '*', ' ');
    sr = sum(results.all_trials{r_idx}.errors < 1e-3) / num_trials * 100;
    fprintf('│ %2d%s  │  %.4e     │  %.4e     │   %.1f    │  %3.0f%%  │\n', ...
            r, marker, results.errors_mean(r_idx), results.errors_std(r_idx), ...
            results.ranks_mean(r_idx), sr);
end
fprintf('└──────┴────────────────┴────────────────┴──────────┴─────────┘\n');
fprintf('* = true rank\n\n');

%% Key Findings
fprintf('=== Key Findings ===\n');

[best_error, best_idx] = min(results.errors_mean);
best_r = r_list(best_idx);

fprintf('1. Best rank: r=%d (mean error: %.2e)\n', best_r, best_error);

if best_r == r_star
    fprintf('   → Exact parameterization is optimal\n');
elseif best_r > r_star
    fprintf('   → Overparameterization helps (r=%d > r*=%d)\n', best_r, r_star);
end

% Robustness to overparameterization
max_degradation = max(results.errors_mean / best_error);
fprintf('2. Max degradation: %.2fx (at r=%d)\n', max_degradation, r_list(end));

if max_degradation < 2
    fprintf('   → ✓ Robust to overparameterization\n');
elseif max_degradation < 5
    fprintf('   → Moderate sensitivity to overparameterization\n');
else
    fprintf('   → ⚠ High sensitivity to overparameterization\n');
end

% Rank recovery
perfect_recovery = all(abs(results.ranks_mean - r_star) < 0.5);
if perfect_recovery
    fprintf('3. ✓ Perfect rank recovery across all r\n');
else
    fprintf('3. Rank recovery varies with used rank r\n');
end

fprintf('\n✓ Rank Comparison Complete\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
