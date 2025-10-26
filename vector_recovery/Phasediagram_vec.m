%%%%%%%%%% General Framework for Sparse Vector Recovery (linear / non-linear) 
d1 = 100; % Vector dimension
kappa = 2; % Signal strength parameter
trial_num = 10; 
verbose = 0;
add_flag = 0;
problem_flag = 0; % 0 for sensing, 1 for phase retrieval, 4 for Fourier sensing
alg_flag = 0; % 0 for GD_vec (can be extended for other vector algorithms)

% Parameters for vector recovery
sparsity_star = 5; % True sparsity level
sparsity_max = 30; % Maximum sparsity to test
sparsity_grid = 1:2:30; % Sparsity levels to test

% Measurement scaling - for vectors, typically need m > 2*s*log(d1/s) for s-sparse vectors
approx = floor(d1 * 8); % More conservative scaling for vectors
% record: selection for scale
T = 1000;
Max_scale = round(log2(approx));

% lambda_list for regularization (L1 penalty for sparsity)
lambda_list = [0, 1e-4, 1e-3, 1e-2];

scale_num = 3; % Number of grid levels
m_max = 2^Max_scale;

% Set algorithm name and solver for vectors
alg_name = 'GD_vec'; % For now, only gradient descent for vectors
params.alg = @solve_GD_vec;

% Set initialization method for vectors
params.init_flag = 1; % Use random initialization
params.init = @init_random_vector;

% Set nonlinear function based on problem type
nonlinear_flag = 0; % 0 for linear, 1 for magnitude (phase retrieval)
params.nonlinear_flag = nonlinear_flag;
if nonlinear_flag == 1
    params.nonlinear_func = @abs; % Phase retrieval
else
    params.nonlinear_func = [];
end

% Create data directory for vector experiments
data_file = sprintf('err_data_vec_d1_%d_smax_%d_kappa_%d_sstar_%d_prob_%d_alg_%s', ...
                   d1, sparsity_max, kappa, sparsity_star, problem_flag, alg_name);
full_path = fullfile('data_f', data_file);
if ~exist(full_path, 'dir')
    mkdir(full_path);
end

%%%%%%%%%%%%%%%%%%%%%% Grid Setup

% Grid levels (coarse to fine)
levels = Max_scale-1:-1:Max_scale-scale_num;
m_grids = cell(length(levels), 1);

for i = 1:length(levels)
    n = levels(i);
    scale_gap = 2^n;
    m_grids{i} = 0:scale_gap:m_max;
end

m_all = [];
for i = 1:length(levels)
    m_current = m_grids{i};    
    % Exclude points already computed (from higher levels)
    if i == 1
        m_all = [];
    else
        new_coords = setdiff(m_current, m_all);
        m_all = [m_all, new_coords];
    end
end

% Total unique m
m_all = m_all(m_all ~= 0);

% For specific problem types, adjust m_all
if problem_flag == 3
   m_all = [d1]; % Custom case
end

points_num = length(m_all);
disp(['Total unique points to compute: ', num2str(points_num)]);

save([full_path,'/mgrid.mat'],"m_all")

%%%%%%%%%%%%%%%%%%%%%% Set ground truth and parameters

% Generate sparse ground truth vector
xstar = zeros(d1, 1);
support = randperm(d1, sparsity_star);
xstar(support) = randn(sparsity_star, 1);
xstar = xstar / norm(xstar); % Normalize

% Store ground truth
params.xstar = xstar;

%%%%%%%%%%%%%%%%%%%%%%%% Main Experiment Loop
for lambda_idx = 1:length(lambda_list)
    lambda = lambda_list(lambda_idx);
    dist = fullfile(full_path, num2str(lambda),'/');
    if ~exist(dist, 'dir')
        mkdir(dist);
    end
    
    for sparsity = sparsity_grid
        points_s = cell(points_num, 1);
        
        % Display the parameters being used for debugging
        disp(['Running Vector Recovery with ', ...
            'sparsity = ', num2str(sparsity), ...
            ', d1 = ', num2str(d1), ...
            ', trials = ', num2str(trial_num), ...
            ', kappa = ', num2str(kappa), ...
            ', lambda = ', num2str(lambda), ...
            ', T = ', num2str(T), ...
            ', s^* = ', num2str(sparsity_star), ...
            ', m points = ', num2str(points_num)]);
        tic;
        
        % Set up parameters for this sparsity level
        params.d1 = d1;
        params.sparsity = sparsity;
        params.kappa = kappa;
        params.trial_num = trial_num;
        params.lambda = lambda;
        params.T = T;
        params.init_scale = 1e-3;
        params.verbose = verbose;
        params.problem_flag = problem_flag;
        params.apply_thresholding = (lambda > 0); % Apply soft thresholding if lambda > 0
        
        parfor i = 1:points_num
            m = m_all(i); 
            
            % Add current m, sparsity, kappa to params for this trial
            params_trial = params;
            params_trial.m = m;
            params_trial.sparsity = sparsity;
            params_trial.kappa = kappa;

            [err_list, p_list, err_list_f] = multipletrial(params_trial, @onetrial_vec);
            points = struct();
            points.sparsity = sparsity; 
            points.m = m; 
            points.e = err_list; 
            points.p = p_list;
            points.e_f = err_list_f; % Store final error
            points.trial_num = trial_num;
            
            points_s{i} = points;
        end
        
        t = toc;
        disp(['Execution time: ', num2str(t), ' seconds']);
        
        % Save results
        for i = 1:points_num
            point = points_s{i};
            point_name = sprintf('s_%d_m_%d_t_%d.mat', point.sparsity, point.m, point.trial_num);
            if add_flag == 0
                save([dist, point_name], "point");
                disp(['Save ', point_name])
            else
                % Load existing point data and combine
                old_data = load([dist, point_name], 'point');
                data_old = old_data.point;
                
                % Update probability: weighted average
                p_new = (data_old.p * data_old.trial_num + point.p * point.trial_num) / ...
                        (data_old.trial_num + point.trial_num);
                
                % Update the point structure
                point.p = p_new;
                point.trial_num = data_old.trial_num + point.trial_num;
                
                % Save updated point back to the same file
                save([dist, point_name], 'point');
                disp(['Updated ', point_name, ' with p = ', num2str(p_new)]);
            end
        end
    end
end

disp('Vector recovery phase diagram computation completed!');

%%%%%%%%%%%%%%%%%
