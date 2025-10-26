%%%%%%%%%% Visualize Tensor Phase Diagram Results
% Simple script to load and visualize tensor experiment results
% Uses the new utility functions for cleaner code

clear; clc;

%% Configuration
% Specify the data directory (mu subdirectory)
data_dir = '../data_f/err_data_d1_20_d2_20_rmax_2_kappa_2_rstar_1_prob_2_alg_PGD/mu_0.0100';

fprintf('=== Tensor Phase Diagram Visualization ===\n');

%% Load results using utility function
results = load_tensor_experiment_results(data_dir);

%% Create visualizations
figure('Position', [100, 100, 1200, 500]);

% Plot 1: Success Rate Heatmap
subplot(1, 2, 1);
imagesc(results.m_values, results.r_values, results.success_matrix);
set(gca, 'YDir', 'normal');
colorbar;
colormap(gca, flipud(gray));
xlabel('Number of Measurements (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Rank (r)', 'FontSize', 12, 'FontWeight', 'bold');
title('Success Rate', 'FontSize', 14, 'FontWeight', 'bold');
caxis([0, 1]);
grid on;
set(gca, 'FontSize', 11);

% Add text annotations
hold on;
for i = 1:length(results.r_values)
    for j = 1:length(results.m_values)
        if ~isnan(results.success_matrix(i, j))
            val = results.success_matrix(i, j);
            % Choose text color based on background
            if val > 0.5
                text_color = 'k';
            else
                text_color = 'w';
            end
            text(results.m_values(j), results.r_values(i), ...
                 sprintf('%.0f%%', val*100), ...
                 'HorizontalAlignment', 'center', ...
                 'FontSize', 10, 'FontWeight', 'bold', ...
                 'Color', text_color);
        end
    end
end
hold off;

% Plot 2: Log Error Heatmap
subplot(1, 2, 2);
log_error = log10(results.error_matrix + 1e-10);
imagesc(results.m_values, results.r_values, log_error);
set(gca, 'YDir', 'normal');
colorbar;
colormap(gca, 'jet');
xlabel('Number of Measurements (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Rank (r)', 'FontSize', 12, 'FontWeight', 'bold');
title('Log_{10} Final Error', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);

% Add text annotations
hold on;
for i = 1:length(results.r_values)
    for j = 1:length(results.m_values)
        if ~isnan(results.error_matrix(i, j))
            text(results.m_values(j), results.r_values(i), ...
                 sprintf('%.1f', log_error(i,j)), ...
                 'HorizontalAlignment', 'center', ...
                 'FontSize', 9, 'Color', 'w', 'FontWeight', 'bold');
        end
    end
end
hold off;

% Overall title
sgtitle('Tensor Phase Diagram Results', 'FontSize', 16, 'FontWeight', 'bold');

%% Save Figure
[~, folder_name] = fileparts(data_dir);
output_file = fullfile(data_dir, sprintf('phase_diagram_%s.png', folder_name));
saveas(gcf, output_file);
fprintf('\nFigure saved to: %s\n', output_file);

%% Optional: Plot error curves for specific points
if length(results.points) > 0 && ~isempty(results.points{1})
    figure('Position', [150, 150, 800, 600]);
    
    % Plot error evolution for each (r, m) pair
    legend_entries = {};
    for i = 1:length(results.points)
        if ~isempty(results.points{i})
            point = results.points{i};
            plot(1:length(point.e), point.e, 'LineWidth', 2);
            hold on;
            legend_entries{end+1} = sprintf('r=%d, m=%d', point.r, point.m);
        end
    end
    
    hold off;
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Error', 'FontSize', 12, 'FontWeight', 'bold');
    title('Error Evolution Curves', 'FontSize', 14, 'FontWeight', 'bold');
    legend(legend_entries, 'Location', 'best');
    grid on;
    set(gca, 'YScale', 'log');
    
    % Save error curves figure
    output_file2 = fullfile(data_dir, sprintf('error_curves_%s.png', folder_name));
    saveas(gcf, output_file2);
    fprintf('Error curves saved to: %s\n', output_file2);
end

fprintf('\nVisualization complete!\n');
