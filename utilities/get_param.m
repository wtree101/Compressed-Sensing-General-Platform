function val = get_param(params, field, default_val)
% GET_PARAM Get parameter value with default fallback
%
% Safely retrieves a parameter from a struct with a default value if the
% field doesn't exist. This is useful for handling optional parameters.
%
% Inputs:
%   params      - Parameter structure
%   field       - Field name to retrieve (string)
%   default_val - Default value to return if field doesn't exist
%
% Output:
%   val - Value of params.field if it exists, otherwise default_val
%
% Usage:
%   params = struct('alpha', 0.5);
%   alpha = get_param(params, 'alpha', 1.0);   % Returns 0.5
%   beta = get_param(params, 'beta', 0.1);     % Returns 0.1 (default)

    if isfield(params, field) && ~isempty(params.(field))
        val = params.(field);
    else
        val = default_val;
    end
end
