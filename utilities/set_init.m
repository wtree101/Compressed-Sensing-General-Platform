function [init_name, init_handle] = set_init(init_flag)
    % SET_INIT Returns initialization function handle based on flag
    %
    % Inputs:
    %   init_flag - 0: Standard (SVD), 1: Random, 2: Power method, 3: Tensor lift
    %
    % Outputs:
    %   init_name   - String name of initialization method
    %   init_handle - Function handle for initialization
    %
    % Note: All initialization functions now use unified signature:
    %       [X0, U0, history] = func(y, operator, d1, d2, params)
    
    switch init_flag
        case 0
            init_name = 'Standard_Init';
            init_handle = @Initialization;
        case 1
            init_name = 'Random_Init';
            init_handle = @Initialization_random;
        case 2
            init_name = 'Power_Method_Init';
            init_handle = @initialize_power_method;
        case 3
            init_name = 'Tensor_Lift_Init';
            init_handle = @initialize_tensor_lift;
        otherwise
            init_name = 'Unknown_Init';
            error('Unknown init_flag value. Use 0 for standard, 1 for random, 2 for power method, 3 for tensor lift.');
    end
end