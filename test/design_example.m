%% Example: Recommended Parameter Structure Design

% 1. CLEAN FUNCTION SIGNATURES
function [err_list, success_list] = multipletrial_clean(params)
    % All parameters in struct - clean and extensible
    
    % Required parameters with validation
    required_fields = {'m', 'r', 'kappa', 'trial_num'};
    for i = 1:length(required_fields)
        if ~isfield(params, required_fields{i})
            error('Required parameter %s is missing', required_fields{i});
        end
    end
    
    % Extract with defaults
    m = params.m;
    r = params.r;
    kappa = params.kappa;
    trial_num = params.trial_num;
    
    % Optional parameters with defaults
    if ~isfield(params, 'lambda'), params.lambda = 0; end
    if ~isfield(params, 'T'), params.T = 200; end
    if ~isfield(params, 'verbose'), params.verbose = false; end
    
    % Main computation
    err_list = zeros(params.T, 1);
    success_list = zeros(params.T, 1);
    
    for i = 1:trial_num
        output_list = onetrial_clean(params);
        err_list = err_list + output_list;
        success_list = success_list + (output_list < params.lambda * 10);
    end
    
    err_list = err_list / trial_num;
    success_list = success_list / trial_num;
end

function [Error_Stand, Error_function] = onetrial_clean(params)
    % All parameters in struct
    
    % Required parameter validation
    required_fields = {'m', 'r', 'kappa', 'd1', 'd2'};
    for i = 1:length(required_fields)
        if ~isfield(params, required_fields{i})
            error('Required parameter %s is missing', required_fields{i});
        end
    end
    
    % Extract parameters
    m = params.m;
    r = params.r;
    kappa = params.kappa;
    d1 = params.d1;
    d2 = params.d2;
    
    % Rest of your onetrial logic...
    % ...existing code...
end

% 2. USAGE EXAMPLES

% Easy to create parameter sets
params_base = struct();
params_base.d1 = 60;
params_base.d2 = 60;
params_base.trial_num = 10;
params_base.T = 1000;
params_base.verbose = false;

% Easy to create variants
params_experiment1 = params_base;
params_experiment1.m = 100;
params_experiment1.r = 5;
params_experiment1.kappa = 2;
params_experiment1.lambda = 0.001;

params_experiment2 = params_base;
params_experiment2.m = 200;
params_experiment2.r = 10;
params_experiment2.kappa = 1.5;
params_experiment2.lambda = 0;

% Clean function calls
[err1, success1] = multipletrial_clean(params_experiment1);
[err2, success2] = multipletrial_clean(params_experiment2);

% 3. PARAMETER VALIDATION HELPER
function validate_params(params, required_fields, optional_defaults)
    % Validate required fields
    for i = 1:length(required_fields)
        if ~isfield(params, required_fields{i})
            error('Required parameter %s is missing', required_fields{i});
        end
    end
    
    % Set optional defaults
    if nargin > 2
        field_names = fieldnames(optional_defaults);
        for i = 1:length(field_names)
            field = field_names{i};
            if ~isfield(params, field)
                params.(field) = optional_defaults.(field);
            end
        end
    end
end
