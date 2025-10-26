%%%%%%%%%% Batch Visualize Multiple Tensor Phase Diagrams
% Script to visualize multiple mu values or compare different experiments

clear; clc;

%% Configuration
base_dir = '../data_f/err_data_d1_20_d2_20_rmax_2_kappa_2_rstar_1_prob_2_alg_PGD';

% Find all mu subdirectories
mu_dirs = dir(fullfile(base_dir, 'mu_*'));
mu_dirs = mu_dirs([mu_dirs.isdir]);

fprintf('=== Batch Tensor Phase Diagram Visualization ===\n');
fprintf('Found %d mu directories\n', length(mu_dirs));

if isempty(mu_dirs)
    error('No mu_* directories found in: %s', base_dir);
end

%% Process each mu directory
all_results = cell(length(mu_dirs), 1);

for i = 1:length(mu_dirs)
    mu_dir = fullfile(base_dir, mu_dirs(i).name);
    fprintf('\n--- Processing: %s ---\n', mu_dirs(i).name);
    
    % Extract mu value from directory name
    mu_str = regexp(mu_dirs(i).name, 'mu_([\d.]+)', 'tokens');
    if ~isempty(mu_str)
        mu_value = str2double(mu_str{1}{1});
    else
        mu_value = NaN;
    end
    
    % Load results
    results = load_tensor_experiment_results(mu_dir);
    results.mu = mu_value;
    all_results{i} = results;
    
    % Create individual figure for this mu
    fig = figure('Position', [100 + (i-1)*50, 100 + (i-1)*50, 1200, 500]);
    
    % Plot 1: Success Rate
    subplot(1, 2, 1);
    imagesc(results.m_values, results.r_values, results.success_matrix);
    set(gca, 'YDir', 'normal');
    colorbar;
    colormap(gca, flipud(gray));
    xlabel('Measurements (m)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Rank (r)', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('Success Rate (\\mu=%.4f)', mu_value), 'FontSize', 12);
    caxis([0, 1]);
    grid on;
    
    % Add annotations
    hold on;
    for ri = 1:length(results.r_values)
        for mi = 1:length(results.m_values)
            if ~isnan(results.success_matrix(ri, mi))
                val = results.success_matrix(ri, mi);
                text(results.m_values(mi), results.r_values(ri), ...
                     sprintf('%.0f%%', val*100), ...
                     'HorizontalAlignment', 'center', ...
                     'FontSize', 9, 'FontWeight', 'bold', ...
                     'Color', val > 0.5 ? 'k' : 'w');
            end
        end
    end
    hold off;
    
    % Plot 2: Log Error
    subplot(1, 2, 2);
    log_error = log10(results.error_matrix + 1e-10);
    imagesc(results.m_values, results.r_values, log_error);
    set(gca, 'YDir', 'normal');
    colorbar;
    colormap(gca, 'jet');
    xlabel('Measurements (m)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Rank (r)', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('Log_{10} Error (\\mu=%.4f)', mu_value), 'FontSize', 12);
    grid on;
    
    sgtitle(sprintf('Phase Diagram: %s', mu_dirs(i).name), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    % Save figure
    output_file = fullfile(mu_dir, sprintf('phase_diagram_mu_%.4f.png', mu_value));
    saveas(fig, output_file);
    fprintf('Saved: %s\n', output_file);
end

%% Comparison plot (if multiple mu values)
if length(all_results) > 1
    figure('Position', [200, 200, 1000, 600]);
    
    % Compare success rates for a specific rank
    rank_to_plot = all_results{1}.r_values(1);  % First rank
    
    subplot(2, 1, 1);
    for i = 1:length(all_results)
        results = all_results{i};
        r_idx = find(results.r_values == rank_to_plot, 1);
        if ~isempty(r_idx)
            plot(results.m_values, results.success_matrix(r_idx, :), ...
                 '-o', 'LineWidth', 2, 'DisplayName', sprintf('\\mu=%.4f', results.mu));
            hold on;
        end
    end
    hold off;
    xlabel('Measurements (m)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Success Rate', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('Success Rate Comparison (r=%d)', rank_to_plot), 'FontSize', 12);
    legend('Location', 'best');
    grid on;
    ylim([0, 1.1]);
    
    % Compare final errors
    subplot(2, 1, 2);
    for i = 1:length(all_results)
        results = all_results{i};
        r_idx = find(results.r_values == rank_to_plot, 1);
        if ~isempty(r_idx)
            semilogy(results.m_values, results.error_matrix(r_idx, :), ...
                     '-s', 'LineWidth', 2, 'DisplayName', sprintf('\\mu=%.4f', results.mu));
            hold on;
        end
    end
    hold off;
    xlabel('Measurements (m)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Final Error', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('Error Comparison (r=%d)', rank_to_plot), 'FontSize', 12);
    legend('Location', 'best');
    grid on;
    
    % Save comparison
    output_file = fullfile(base_dir, 'comparison_across_mu.png');
    saveas(gcf, output_file);
    fprintf('\nComparison saved: %s\n', output_file);
end

fprintf('\n=== Batch visualization complete! ===\n');
