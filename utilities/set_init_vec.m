function [init_name, init_handle] = set_init_vec(init_flag)
    % Set initialization method for sparse vector recovery
    % Inputs:
    %   init_flag - Initialization method selector
    %               0: Zero initialization (most common)
    %               1: Small random initialization  
    %               2: Matching pursuit initialization
    %               3: Least squares + sparsification
    % Outputs:
    %   init_name - Name of the initialization method
    %   init_handle - Function handle for initialization
    
    switch init_flag
        case 0
            init_name = 'Zero';
            init_handle = @init_zero_vector;
            
        case 1
            init_name = 'Random';
            init_handle = @init_random_vector;
            
        case 2
            init_name = 'MatchingPursuit';
            init_handle = @init_matching_pursuit_vector;
            
        case 3
            init_name = 'LeastSquares';
            init_handle = @init_random_vector; % Uses the updated version with LS
            
        otherwise
            warning('Unknown init_flag %d, using zero initialization', init_flag);
            init_name = 'Zero';
            init_handle = @init_zero_vector;
    end
    
    fprintf('Vector initialization method: %s\n', init_name);
end
