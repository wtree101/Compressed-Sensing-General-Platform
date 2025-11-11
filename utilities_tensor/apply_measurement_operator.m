function y = apply_measurement_operator(A_operator, T, varargin)
% APPLY_MEASUREMENT_OPERATOR Apply forward measurement operator A to tensor T
%
% Computes measurements y = A(T) where:
%   y_i = <A_i ⊗ A_i, T> / scaling
%
% For phase retrieval with tensor lift T = X ⊗ X:
%   y_i = <A_i ⊗ A_i, T> / sqrt(m)
%
% Inputs:
%   A_operator - TuckerOperator or struct with A_cells field
%   T          - 4th-order tensor (d×d×d×d)
%
% Optional Name-Value Pairs:
%   'scaling'      - Scaling factor (default: 1/sqrt(m))
%   'use_tucker'   - If true and T is TuckerTensor, use efficient operator
%                    (default: true)
%
% Output:
%   y - Measurement vector (m × 1)
%
% Examples:
%   % Apply to full tensor
%   T = create_tensor_from_matrix(X, d);
%   y = apply_measurement_operator(operator, T);
%   
%   % Apply to Tucker tensor (efficient)
%   T_tucker = TuckerTensor([d,d,d,d], [r,r,r,r]);
%   y = apply_measurement_operator(operator, T_tucker);
%   
%   % Custom scaling
%   y = apply_measurement_operator(operator, T, 'scaling', 1);

% Parse inputs
p = inputParser;
addRequired(p, 'A_operator');
addRequired(p, 'T');
addParameter(p, 'scaling', []);  % Auto-detect if empty
addParameter(p, 'use_tucker', true, @islogical);
parse(p, A_operator, T, varargin{:});

use_tucker = p.Results.use_tucker;

% Extract A_cells and m
if isa(A_operator, 'TuckerOperator')
    A_cells = A_operator.A_cells;
    m = A_operator.m;
else
    A_cells = A_operator.A_cells;
    m = length(A_cells);
end

% Set default scaling
if isempty(p.Results.scaling)
    scaling = 1 / sqrt(m);
else
    scaling = p.Results.scaling;
end

% Choose computation method based on T type
if isa(T, 'TuckerTensor') && use_tucker && isa(A_operator, 'TuckerOperator')
    % Efficient Tucker tensor computation
    y = A_operator.forward(T) * scaling;
    
elseif isa(T, 'TuckerTensor')
    % Convert Tucker to full tensor first
    T_full = T.full();
    y = apply_to_full_tensor(A_cells, T_full, scaling, m);
    
else
    % T is a full tensor array
    y = apply_to_full_tensor(A_cells, T, scaling, m);
end

end

%% Helper function for full tensor
function y = apply_to_full_tensor(A_cells, T, scaling, m)
    % Apply measurement operator to full tensor
    % y_i = <A_i ⊗ A_i, T> * scaling
    
    % Get tensor dimensions
    sz = size(T);
    if length(sz) ~= 4
        error('Tensor must be 4th-order');
    end
    d = sz(1);
    
    % Verify all dimensions match
    if ~all(sz == d)
        error('Tensor must be d×d×d×d');
    end
    
    % Compute measurements
    y = zeros(m, 1);
    
    % Flatten tensor for efficient computation
    T_vec = T(:);  % d^4 × 1 vector
    
    for i = 1:m
        Ai = A_cells{i};
        
        % Compute A_i ⊗ A_i efficiently
        % <A_i ⊗ A_i, T> = vec(A_i)' * kron(A_i, A_i) * vec(T)
        %                 = vec(A_i)' * (vec(A_i) * vec(A_i)') * vec(T)
        %                 = (vec(A_i)' * vec(T))^2
        
        Ai_vec = Ai(:);  % d^2 × 1 vector
        
        % For 4th-order: need to compute proper Kronecker structure
        % A_i ⊗ A_i is (d^2 × d^2) tensor viewed as (d^4 × 1) when flattened
        AiAi = Ai_vec * Ai_vec';  % d^2 × d^2 outer product
        AiAi_vec = AiAi(:);  % d^4 × 1 vector
        
        % Inner product
        y(i) = (AiAi_vec' * T_vec) * scaling;
    end
end
