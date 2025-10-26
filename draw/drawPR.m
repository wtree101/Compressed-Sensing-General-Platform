% Load data
dist = 'data_f/err_data_d1_20_d2_20_rmax_2_kappa_2_rstar_1_prob_2_alg_PGD/';
%dist = 'data3/err_data_d1_60_d2_60_rmax_20_kappa_2_rstar_8/';
data_dir = [dist,'/mu_0.0100']; % Replace with your path
%data_dir = [dist]; % Replace with your path
m_all = load([dist,'mgrid.mat']);
m_all = sort(m_all.m_all);
r_max = 2;
r_grid = 1:r_max;

file_list = dir(fullfile(data_dir, '*.mat'));
P = zeros(r_max,length(m_all));
m_map = containers.Map(m_all, 1:length(m_all));


for i = 1:length(file_list)
    filename = file_list(i).name;
    tokens = regexp(filename, 'r_(\d+)_m_(\d+)_t_\d+\.mat', 'tokens');
    if ~isempty(tokens)
        
        data = load(fullfile(data_dir, filename));
        p = data.point.p;
        r = data.point.r;
        m = data.point.m;
        if r <= r_max && isKey(m_map, m)
            m_idx = m_map(m);
            P(r, m_idx) = p;
        end
    end
end

% Plot heat-map
P = P';

% Create an invisible figure
%fig = figure('Visible', 'off');
figure(1);
imagesc(r_grid, m_all, P(:,1:r_max));
set(gca, 'YDir', 'normal');
colorbar;
colormap('gray'); % Black (0) to white (1)
xlabel('r');
ylabel('m');
title('Heat-map of Success Probability');

% % Save the figure quietly
% saveas(fig, [dist,'/success_probability_heatmap.png']);
% 
% % Close the figure
% close(fig);