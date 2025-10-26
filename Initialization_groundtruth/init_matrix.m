function [X0, U0, history] = init_matrix(y, operator, d1, d2, r, m, method, scale)
% INIT_MATRIX Unified interface for matrix initialization methods
%
% Provides a single interface to call different initialization methods
% with consistent input/output signatures.
%
% Inputs:
%   y        - Measurement vector (m x 1)
%   operator - Struct with fields:
%              .A: Forward operator @(X) A*X(:)
%              .A_star: Adjoint operator @(y) reshape(A'*y, [d1,d2])
%   d1       - Matrix row dimension
%   d2       - Matrix column dimension
%   r        - Target rank for initialization
%   m        - Number of measurements
%   method   - String specifying initialization method:
%              'spectral' or 'svd': Truncated SVD initialization
%              'random': Random initialization
%              'power': Power method initialization
%              'zero': Zero initialization
%   scale    - (Optional) Scaling factor, default depends on method
%
% Outputs:
%   X0       - Initialized matrix (d1 x d2)
%   U0       - Factor matrix (d1 x r) such that X0 ≈ U0*U0'
%   history  - Convergence history (only non-empty for 'power' method)
%
% Examples:
%   [X0, U0] = init_matrix(y, operator, 20, 20, 2, 100, 'spectral');
%   [X0, U0, ~] = init_matrix(y, operator, 20, 20, 2, 100, 'random', 0.1);
%   [X0, U0, history] = init_matrix(y, operator, 20, 20, 2, 100, 'power');

if nargin < 8
    % Default scaling depends on method
    if strcmpi(method, 'random')
        scale = 0.1;
    else
        scale = 1.0;
    end
end

% Call appropriate initialization method
switch lower(method)
    case {'spectral', 'svd', 'truncated_svd'}
        [X0, U0, history] = Initialization(y, operator, d1, d2, r, m, scale);
        
    case {'random', 'rand', 'randn'}
        [X0, U0, history] = Initialization_random(y, operator, d1, d2, r, m, scale);
        
    case {'power', 'power_method', 'spectral_power'}
        [X0, U0, history] = initialize_power_method(y, operator, d1, d2, r, m, scale);
        
    case {'zero', 'zeros'}
        X0 = zeros(d1, d2);
        U0 = zeros(d1, r);
        history = struct();
        
    otherwise
        error('Unknown initialization method: %s. Options: spectral, random, power, zero', method);
end

end
