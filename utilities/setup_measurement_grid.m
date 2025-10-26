function [m_all, data_dir] = setup_measurement_grid(params)
    % Generate measurement grid and create data directory
    % Inputs:
    %   params - Structure with fields:
    %            .d1, .d2      - Matrix dimensions
    %            .r_max        - Maximum rank for experiments
    %            .kappa        - Condition number
    %            .problem_flag - Problem type
    %            .alg_name     - Algorithm name
    %            .scale_num    - Number of scale levels
    % Outputs:
    %   m_all - Array of measurement numbers to test
    %   data_dir - Directory path for saving results
    
    % Extract parameters
    d1 = params.d1;
    d2 = params.d2;
    r_max = params.r_max;
    kappa = params.kappa;
    problem_flag = params.problem_flag;
    alg_name = params.alg_name;
    scale_num = params.scale_num;
    
    % Calculate grid parameters
        
        %approx = (d1 + d2)*r_max*6;
        %approx = (d1 + d2)*r_max*3;
        %approx = floor((d1 + d2)*10);
        % record: selection for scale
        %approx = floor((d1 + d2)*10); for r_star = 01
    approx = floor(d1^2 * 2);
    Max_scale = round(log2(approx));
    m_max = 2^Max_scale;
    
    % Grid levels (coarse to fine)
    levels = Max_scale-1:-1:Max_scale-scale_num;
    m_grids = cell(length(levels), 1);
    
    % Generate grids for each level
    for i = 1:length(levels)
        n = levels(i);
        scale_gap = 2^n;
        m_grids{i} = 0:scale_gap:m_max;
    end
    
    % Combine all unique points
    m_all = [];
    for i = 1:length(levels)
        m_current = m_grids{i};
        if i == 1
            m_all = [];
        else
            new_coords = setdiff(m_current, m_all);
            m_all = [m_all, new_coords];
        end
    end
    
    % Remove zero measurements and handle special cases
    m_all = m_all(m_all ~= 0);
    if problem_flag == 3
        m_all = [d1*d1]; % Richard's example
    end
    
    % Create data directory
    data_file = sprintf('err_data_d1_%d_d2_%d_rmax_%d_kappa_%d_prob_%d_alg_%s', ...
        d1, d2, r_max, kappa, problem_flag, alg_name);
    data_dir = fullfile('data_f', data_file);
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    
    % Save grid information
    save(fullfile(data_dir, 'mgrid.mat'), "m_all");
    
    fprintf('Measurement Grid Setup:\n');
    fprintf('  Total unique points to compute: %d\n', length(m_all));
    fprintf('  Range: [%d, %d]\n', min(m_all), max(m_all));
    fprintf('  Data directory: %s\n', data_dir);
end
