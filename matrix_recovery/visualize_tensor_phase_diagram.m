%%%%%%%%%% Visualize Tensor Phase Diagram Results
% This script loads and visualizes the results from Phasediagram_tensor.m

clear; clc; close all;

%% Load Configuration and Results
fprintf('=== Loading Tensor Phase Diagram Results ===\n');

% Specify the data directory (modify this to your actual directory)
% Example: data_dir = 'DATA/tensor_phase_diagram/d40_r1_T500_20251025_123456';
data_dir = input('Enter data directory path: ', 's');

if ~exist(data_dir, 'dir')
    error('Directory does not exist: %s', data_dir);
end

% Load configuration
config_file = fullfile(data_dir, 'config.mat');
if ~exist(config_file, 'file')
    error('Configuration file not found: %s', config_file);
end

load(config_file, 'config');
fprintf('Configuration loaded:\n');
fprintf('  d = %d\n', config.d);
fprintf('  r_star = %d\n', config.r_star);
fprintf('  Rank grid: [%d, %d]\n', min(config.r_grid), max(config.r_grid));
fprintf('  Trials per point: %d\n', config.trial_num);

%% Load Results for All Step Sizes
mu_dirs = dir(fullfile(data_dir, 'mu_*'));
num_mu = length(mu_dirs);

if num_mu == 0
    error('No results found in directory: %s', data_dir);
end

fprintf('\nFound %d step size directories\n', num_mu);

all_results = cell(num_mu, 1);
mu_values = zeros(num_mu, 1);

for mu_idx = 1:num_mu
    mu_dir = fullfile(data_dir, mu_dirs(mu_idx).name);
    
    % Extract mu value from directory name
    mu_str = strrep(mu_dirs(mu_idx).name, 'mu_', '');
    mu_values(mu_idx) = str2double(mu_str);
    
    fprintf('Loading results for mu = %.4f...\n', mu_values(mu_idx));
    
    % Load all rank results
    results_mu = cell(length(config.r_grid), 1);
    for r_idx = 1:length(config.r_grid)
        r = config.r_grid(r_idx);
        result_file = fullfile(mu_dir, sprintf('r_%d.mat', r));
        
        if exist(result_file, 'file')
            load(result_file, 'results');
            results_mu{r_idx} = results;
        else
            warning('Result file not found: %s', result_file);
        end
    end
    
    all_results{mu_idx} = results_mu;
end

%% Generate Phase Diagrams
fprintf('\n=== Generating Phase Diagrams ===\n');

for mu_idx = 1:num_mu
    mu = mu_values(mu_idx);
    results_mu = all_results{mu_idx};
    
    % Extract data for phase diagram
    r_vals = config.r_grid;
    num_r = length(r_vals);
    
    % Get measurement values from first non-empty result
    m_vals = [];
    for r_idx = 1:num_r
        if ~isempty(results_mu{r_idx})
            m_vals = results_mu{r_idx}.m_values;
            break;
        end
    end
    
    if isempty(m_vals)
        warning('No valid results for mu = %.4f', mu);
        continue;
    end
    
    num_m = length(m_vals);
    
    % Initialize matrices
    success_rate = zeros(num_r, num_m);
    avg_error_mat = zeros(num_r, num_m);
    
    for r_idx = 1:num_r
        if ~isempty(results_mu{r_idx})
            res = results_mu{r_idx};
            success_rate(r_idx, :) = res.success_count / config.trial_num;
            avg_error_mat(r_idx, :) = log10(res.avg_error + 1e-10);  % Log scale
        end
    end
    
    %% Plot Success Rate Phase Diagram
    figure('Position', [100, 100, 1200, 500]);
    
    subplot(1, 2, 1);
    imagesc(m_vals, r_vals, success_rate);
    colorbar;
    colormap(jet);
    caxis([0, 1]);
    xlabel('Number of Measurements (m)');
    ylabel('Rank (r)');
    title(sprintf('Success Rate (mu=%.4f)', mu));
    set(gca, 'YDir', 'normal');
    
    % Add contour for 50% success
    hold on;
    contour(m_vals, r_vals, success_rate, [0.5, 0.5], 'k-', 'LineWidth', 2);
    
    % Add theoretical boundary: m = r(2d-r)
    d = config.d;
    m_theory = r_vals .* (2*d - r_vals);
    plot(m_theory, r_vals, 'w--', 'LineWidth', 2, 'DisplayName', 'm = r(2d-r)');
    legend('Location', 'best');
    
    %% Plot Average Error Phase Diagram
    subplot(1, 2, 2);
    imagesc(m_vals, r_vals, avg_error_mat);
    colorbar;
    colormap(jet);
    xlabel('Number of Measurements (m)');
    ylabel('Rank (r)');
    title(sprintf('Log10 Average Error (mu=%.4f)', mu));
    set(gca, 'YDir', 'normal');
    
    % Add contour for error threshold
    hold on;
    contour(m_vals, r_vals, avg_error_mat, [log10(1e-3)], 'k-', 'LineWidth', 2);
    
    % Save figure
    fig_file = fullfile(data_dir, sprintf('phase_diagram_mu_%.4f.png', mu));
    saveas(gcf, fig_file);
    fprintf('Saved phase diagram: %s\n', fig_file);
end

%% Compare Different Step Sizes (if multiple)
if num_mu > 1
    figure('Position', [100, 100, 1400, 500]);
    
    % Choose a specific rank to compare
    r_compare = config.r_star;
    r_idx = find(config.r_grid == r_compare, 1);
    
    if ~isempty(r_idx)
        for mu_idx = 1:num_mu
            results_mu = all_results{mu_idx};
            if ~isempty(results_mu{r_idx})
                res = results_mu{r_idx};
                
                subplot(1, 2, 1);
                plot(res.m_values, res.success_count / config.trial_num, ...
                     'o-', 'LineWidth', 2, 'DisplayName', sprintf('mu=%.4f', mu_values(mu_idx)));
                hold on;
                
                subplot(1, 2, 2);
                semilogy(res.m_values, res.avg_error, ...
                        'o-', 'LineWidth', 2, 'DisplayName', sprintf('mu=%.4f', mu_values(mu_idx)));
                hold on;
            end
        end
        
        subplot(1, 2, 1);
        xlabel('Number of Measurements (m)');
        ylabel('Success Rate');
        title(sprintf('Success Rate vs m (r=%d)', r_compare));
        legend('show');
        grid on;
        
        subplot(1, 2, 2);
        xlabel('Number of Measurements (m)');
        ylabel('Average Error');
        title(sprintf('Error vs m (r=%d)', r_compare));
        legend('show');
        grid on;
        
        % Save comparison figure
        fig_file = fullfile(data_dir, sprintf('mu_comparison_r_%d.png', r_compare));
        saveas(gcf, fig_file);
        fprintf('Saved comparison figure: %s\n', fig_file);
    end
end

fprintf('\n=== Visualization Complete ===\n');
