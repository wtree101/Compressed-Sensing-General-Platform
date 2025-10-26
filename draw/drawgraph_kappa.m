% Load the saved data from the .mat file
loaded_data = load('results/d1_20_d2_15_r_3_trial_50.mat');
results = loaded_data.results;

% Preallocate arrays for r and transition points
kappa_values = [results.kappa]; % Extract all r values
transition_points = [results.output]; % Preallocate for transition points



% Plot transition point vs r
fig = figure('Visible', 'off'); % Open a new figure
plot(kappa_values, transition_points, '-o', 'LineWidth', 2, 'MarkerSize', 8); % Line with markers
grid on; % Add grid for better visualization
xlabel('kappa', 'FontSize', 12); % X-axis label
ylabel('Transition Point', 'FontSize', 12); % Y-axis label
title('Transition Point vs kappa', 'FontSize', 14); % Title of the graph

% Save the plot as a file
saveas(fig, 'figures/TransitionPoint_vs_kappa.png'); % Save the figure
disp('Graph saved as TransitionPoint_vs_kappa.png');