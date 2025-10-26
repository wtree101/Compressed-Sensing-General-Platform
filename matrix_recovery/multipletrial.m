function [err_list,success_rate] = multipletrial(params)
    % Extract parameters from the struct 'params'
    
    % Required parameters
    
    % Extract core parameters
    trial_num = params.trial_num;

    if isfield(params, 'T')
        T = params.T;
    else
        disp('Warning: Please set the number of iterations T in params')
        T = 200; % Default value
    end
    
    % Optional lambda with default
    if ~isfield(params, 'lambda')
        params.lambda = 0;
    end
    
    % Check if parallel processing is enabled (default: false)
    if ~isfield(params, 'use_parallel')
        params.use_parallel = false;
    end
    
    onetrial = params.onetrial; % Get the function handle for onetrial
   
    err_list = zeros(T,1);
    success_rate = 0; 
    
    % Use parfor only if use_parallel is true AND trial_num > 1
    if params.use_parallel && trial_num > 1
        parfor i = 1:trial_num
            [output, is_success] = onetrial(params); % Pass entire params struct
            err_list = err_list + output.Error_Stand;  % Access Error_Stand from output struct
            success_rate = success_rate + is_success; % Count successes
        end
    else
        for i = 1:trial_num
            [output, is_success] = onetrial(params); % Pass entire params struct
            err_list = err_list + output.Error_Stand;  % Access Error_Stand from output struct
            success_rate = success_rate + is_success; % Count successes
        end
    end

    err_list = err_list / trial_num; % Average error over trials
    success_rate = success_rate / trial_num; % Average success rate over trials
%err_avg = mean(err_list);
end

