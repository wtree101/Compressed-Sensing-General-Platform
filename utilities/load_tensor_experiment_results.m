function results_summary = load_tensor_experiment_results(data_dir, r_grid, m_grid)
    % LOAD_TENSOR_EXPERIMENT_RESULTS Load and aggregate tensor experiment results
    %
    % Inputs:
    %   data_dir - Directory containing result files (r_*_m_*_t_*.mat)
    %   r_grid   - Array of rank values to load (optional)
    %   m_grid   - Array of measurement values to load (optional)
    %
    % Outputs:
    %   results_summary - Structure containing:
    %       .r_values      - Unique rank values found
    %       .m_values      - Unique measurement values found
    %       .success_matrix - Success rate matrix (rank × measurements)
    %       .error_matrix   - Final error matrix (rank × measurements)
    %       .trial_count    - Number of trials matrix (rank × measurements)
    %       .points         - Cell array of all point structures
    %       .data_dir       - Source directory
    
    fprintf('Loading results from: %s\n', data_dir);
    
    %% Find all result files
    files = dir(fullfile(data_dir, 'r_*_m_*_t_*.mat'));
    
    if isempty(files)
        error('No result files found in: %s', data_dir);
    end
    
    fprintf('Found %d result files\n', length(files));
    
    %% Load all points
    points = cell(length(files), 1);
    r_values = [];
    m_values = [];
    
    for i = 1:length(files)
        filepath = fullfile(data_dir, files(i).name);
        data = load(filepath);
        
        if isfield(data, 'point')
            points{i} = data.point;
            r_values = [r_values; data.point.r];
            m_values = [m_values; data.point.m];
        end
    end
    
    % Get unique sorted values
    r_unique = unique(r_values);
    m_unique = unique(m_values);
    
    % Filter by provided grids if specified
    if nargin >= 2 && ~isempty(r_grid)
        r_unique = intersect(r_unique, r_grid);
    end
    if nargin >= 3 && ~isempty(m_grid)
        m_unique = intersect(m_unique, m_grid);
    end
    
    fprintf('Ranks: [%s]\n', num2str(r_unique'));
    fprintf('Measurements: [%s]\n', num2str(m_unique'));
    
    %% Build matrices
    n_r = length(r_unique);
    n_m = length(m_unique);
    
    success_matrix = NaN(n_r, n_m);
    error_matrix = NaN(n_r, n_m);
    trial_count = zeros(n_r, n_m);
    
    for i = 1:length(points)
        if isempty(points{i})
            continue;
        end
        
        point = points{i};
        r_idx = find(r_unique == point.r);
        m_idx = find(m_unique == point.m);
        
        if ~isempty(r_idx) && ~isempty(m_idx)
            success_matrix(r_idx, m_idx) = point.p;
            error_matrix(r_idx, m_idx) = point.e(end);  % Final error
            trial_count(r_idx, m_idx) = point.trial_num;
        end
    end
    
    %% Package output
    results_summary = struct();
    results_summary.r_values = r_unique;
    results_summary.m_values = m_unique;
    results_summary.success_matrix = success_matrix;
    results_summary.error_matrix = error_matrix;
    results_summary.trial_count = trial_count;
    results_summary.points = points;
    results_summary.data_dir = data_dir;
    
    %% Print summary
    fprintf('\n=== Results Summary ===\n');
    fprintf('%-5s | ', 'r\m');
    for j = 1:n_m
        fprintf('%7d ', m_unique(j));
    end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 10 + 8*n_m));
    
    for i = 1:n_r
        fprintf('%-5d | ', r_unique(i));
        for j = 1:n_m
            if ~isnan(success_matrix(i, j))
                fprintf('%6.0f%% ', success_matrix(i,j)*100);
            else
                fprintf('   N/A  ');
            end
        end
        fprintf('\n');
    end
    
    fprintf('\nTotal trials per point:\n');
    fprintf('%-5s | ', 'r\m');
    for j = 1:n_m
        fprintf('%7d ', m_unique(j));
    end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 10 + 8*n_m));
    
    for i = 1:n_r
        fprintf('%-5d | ', r_unique(i));
        for j = 1:n_m
            fprintf('%7d ', trial_count(i,j));
        end
        fprintf('\n');
    end
end
