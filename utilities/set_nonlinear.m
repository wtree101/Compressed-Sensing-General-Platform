function [nonlinear_name, nonlinear_handle] = set_nonlinear(nonlinear_flag)
    switch nonlinear_flag
        case 0
            nonlinear_name = 'Identity';
            nonlinear_handle = @identity_func;
        case 1
            nonlinear_name = 'Absolute_Value';
            nonlinear_handle = @absolute_value_func;
        case 2
            nonlinear_name = 'Square';
            nonlinear_handle = @square_func;
        case 3
            nonlinear_name = 'Sign';
            nonlinear_handle = @sign_func;
        otherwise
            nonlinear_name = 'Identity';
            nonlinear_handle = @identity_func;
            warning('Unknown nonlinear_flag value. Using identity function as default.');
    end
end

function y_out = identity_func(y)
    % Identity function - no transformation
    y_out = y;
end

function y_out = absolute_value_func(y)
    % Absolute value function - for phase retrieval
    y_out = abs(y);
end

function y_out = square_func(y)
    % Square function - element-wise squaring
    y_out = y.^2;
end

function y_out = sign_func(y)
    % Sign function - returns sign of each element
    y_out = sign(y);
end