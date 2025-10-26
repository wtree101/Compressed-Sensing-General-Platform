
% Set up parameters
dist = 'data_inits/err_data_d1_60_d2_60_rmax_20_kappa_2_rstar_1_prob_2_alg_GD/';
%dist = 'data3/err_data_d1_60_d2_60_rmax_20_kappa_2_rstar_8/err_data_d1_60_d2_60_rmax_20_kappa_2_rstar_8/';
data_dir = [dist,'0']; % Adjust as needed
m_all = load([dist,'mgrid.mat']);
m_all = sort(m_all.m_all);
r_max = 20;
%r_grid = [1,2,5,10];
%r_grid = 1:10;
r_grid = 1:20;
T = 500;

% Choose the m value you want to plot
m_target = m_all(4); % Change this to your desired m
err_vs_r = nan(length(r_grid), T);

for id = 1:length(r_grid)
    r = r_grid(id);
    filename = sprintf('r_%d_m_%d_t_10.mat', r, m_target); % Adjust t_10 if needed
    file_path = fullfile(data_dir, filename);
    if exist(file_path, 'file')
        data = load(file_path);
        err_vs_r(id,:) = data.point.e(1:T); % points.e is assumed to be a vector
    end
end

% Plot
figure;
semilogy(err_vs_r(1,:), '-','LineWidth',1); % Plot the first error value for each r
hold on
for i = 2:length(r_grid)
    semilogy(err_vs_r(i,:), '-','LineWidth',1); % Plot the residual (last error value) for each r
end
hold off
%set(gca, 'YScale', 'log');
legend(arrayfun(@(r) sprintf('r = %d', r), r_grid, 'UniformOutput', false), 'Location', 'best');
xlabel('Iteration');
ylabel('Error (mean of points.e)');
title(['Error vs r for m = ', num2str(m_target)]);
grid on;

if ~exist('figures', 'dir')
    mkdir('figures');
end
% Sanitize dist for filename
dist_name = regexprep(dist, '[\/\\]', '_');
fig_name = sprintf('figures_curves/%s_m_%d.png', dist_name, m_target);
saveas(gcf, fig_name);