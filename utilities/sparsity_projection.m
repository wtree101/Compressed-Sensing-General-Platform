
function xl = sparsity_projection(xl, params)
    if isfield(params, 'sparsity')
        s = params.sparsity;
    else
        s = length(xl); % Default to no sparsity constraint
        warning('Parameter sparsity not specified for sparsity projection. No projection applied.');
    end
    [~, idx] = sort(abs(xl), 'descend');
    xl_proj = zeros(size(xl));
    xl_proj(idx(1:s)) = xl(idx(1:s));
    xl = xl_proj;
end