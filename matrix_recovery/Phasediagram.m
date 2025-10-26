%%%%%%%%%% General Framework for Low-rank matrix recovery (linear / non-linear) 
d1 = 60; d2 = 60;
kappa = 2;
trial_num = 10; verbose = 0;
add_flag = 0;
problem_flag = 2; % 0 for sensing, 1 for phase retrieval, 2 for symmetric Gaussian sensing; 3 for Richard's example; 4 for PSD full rank
alg_flag = 3; % 0 for GD, 1 for RGD, 2 for SGD, 3 for SubGD
% Parameters
%Max_scale = 14;
r_star = 1;
r_max = 20; % Adjust r range as needed (e.g., r in [0, 1])
r_grid = 1:1:20;
%approx = (d1 + d2)*r_max*6;
%approx = (d1 + d2)*r_max*3;
approx = floor((d1 + d2)*10);
% record: selection for scale
%approx = floor((d1 + d2)*10); for r_star = 01
T =1000;
Max_scale = round(log2(approx));

% lambda_list = [1e-16,1e-15,1e-15,1e-13,1e-12,1e-11,1e-10,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2];
lambda_list = [0];

scale_num = 3; %3 8 points 4 17 points
m_max = 2^Max_scale; % 16384

% Set algorithm name and solver
[alg_name, alg_handle] = set_solver(alg_flag);
params.alg = alg_handle;

% Set initialization method
params.init_flag = 1; % Use random initialization (set this before calling set_init)
[init_name, init_handle] = set_init(params.init_flag);
params.init = init_handle;

% Set nonlinear function based on problem type
nonlinear_flag = 0;
params.nonlinear_flag = nonlinear_flag;
[nonlinear_name, nonlinear_handle] = set_nonlinear(nonlinear_flag);
params.nonlinear_func = nonlinear_handle;



% Create data directory
data_file = sprintf('err_data_d1_%d_d2_%d_rmax_%d_kappa_%d_rstar_%d_prob_%d_alg_%s', d1, d2, r_max, kappa, r_star, problem_flag, alg_name);
full_path = fullfile('data_f', data_file);
if ~exist(full_path, 'dir')
    mkdir(full_path);
end




%%%%%%%%%%%%%%%%%%%%%%

% Grid levels (coarse to fine)
levels = Max_scale-1:-1:Max_scale-scale_num;
%levels = [Max_scale-1, Max_scale-2, Max_scale-3, Max_scale-4]; % Start at 2^13, go down to 2^10 (adjust as needed)
m_grids = cell(length(levels), 1);
%%%%%%%%%%%%%%%%%%%%%%%



for i = 1:length(levels)
    n = levels(i);
    scale_gap = 2^n; % e.g., 8193, 4097, 2049, 1025
    m_grids{i} = 0:scale_gap:m_max;
  
end
m_all = [];

for i = 1:length(levels)
    m_current = m_grids{i};    
    % Exclude points already computed (from higher levels)
    if i == 1
        m_all = [];
    else
        % Check for uniqueness (points not in previous levels)
        
        new_coords = setdiff(m_current, m_all);
        m_all = [m_all, new_coords];
      
    end
end

% Total unique m
m_all = m_all(m_all ~= 0);
if problem_flag == 3
   m_all = [d1*d1];
end
points_num = length(m_all);
disp(['Total unique points to compute: ', num2str(points_num)]);



save([full_path,'/mgrid.mat'],"m_all")

%parpool(16);
%%%%%%%%%%%%%%%%%%%%%% set ground truth and some params

%params.mu = 0.2;  different default mu for different algs, not set here

% special case, not random A,
if problem_flag == 3
    % Richard's example
    [A, Xstar] = prob3(d1, 100, r_star, r_star);
    Xstar = Xstar / norm(Xstar, 'fro'); % Normalize Xstar
    params.A = A;
    params.mu = 20; % smaller scale
else
    Xstar = groundtruth(d1,d2,r_star,kappa,1); %share the same ground truth with r^* fixed.
end


%%%%%%%%%%%%%%%%%%%%%%%% Main Experiment Loop
for lambda_idx = 1:length(lambda_list)
    lambda = lambda_list(lambda_idx);
    dist = fullfile(full_path, num2str(lambda),'/');
    if ~exist(dist, 'dir')
        mkdir(dist);
    end
    % Place the rest of your experiment code here if needed
    for r = r_grid
        points_r = cell(points_num, 1);
        % Display the parameters being used for debugging
        disp(['Running PhaseTransition with ', ...
            'r = ', num2str(r), ...
            ', d1 = ', num2str(d1), ...
            ', d2 = ', num2str(d2), ...
            ', trials = ', num2str(trial_num), ...
            ', kappa = ', num2str(kappa), ...
            ', lambda = ', num2str(lambda), ...
            ', T = ', num2str(T), ...
            ', r^* = ', num2str(r_star), ...
            ', m points = ', num2str(points_num)]);
        tic;
       
        
        params.d1 = d1;
        params.d2 = d2;
        params.r = r;
        params.kappa = kappa;
        params.trial_num = trial_num;
        params.Xstar = Xstar;
        params.lambda = lambda;
        params.T = T;
        params.init_scale = 1e-10;
        params.verbose = verbose;
        params.problem_flag = problem_flag; %
        parfor i = 1:points_num
            m = m_all(i); 
            
            % Add current m, r, kappa, lambda to params for this trial
            params_trial = params;
            params_trial.m = m;
            params_trial.r = r;
            params_trial.kappa = kappa;
            params_trial.lambda = lambda;
            
            [err_list,p_list,err_list_f] = multipletrial(params_trial);
            points = struct();
            points.r = r; points.m = m; points.e = err_list; points.p = p_list;
            points.e_f = err_list_f; % Store final error
            %points.Xstar = Xstar; 
            points.trial_num = trial_num;
            
            points_r{i} = points;
            %save the data for graph
            
        end
        t = toc;
            disp(['Execution time: ', num2str(t), ' seconds']);
        for i = 1:points_num
            point = points_r{i};
            point_name = sprintf('r_%d_m_%d_t_%d.mat', point.r, point.m, point.trial_num);
            if add_flag==0
                save([dist, point_name], "point");
                disp(['Save ', point_name])
            else
                    % Load existing point data
                old_data = load([dist, point_name], 'point');
                data_old = old_data.point;
                
                % Update probability: weighted average
                p_new = (data_old.p * data_old.trial_num + point.p * point.trial_num) / ...
                        (data_old.trial_num + point.trial_num);
                
                % Update the point structure
                point.p = p_new;
                point.trial_num = data_old.trial_num + point.trial_num; % Combine trial counts
                
                % Save updated point back to the same file
                save([dist, point_name], 'point');
                disp(['Updated ', point_name, ' with p = ', num2str(p_new)]);
            end
        
        end

    end
end



%delete(gcp('nocreate'));

%%%%%%%%%%%%%%%%%




