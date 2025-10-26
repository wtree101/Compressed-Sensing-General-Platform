function Xl = rank_projection(Xl, r)
 
    [U, S, V] = svd(Xl);
    S_proj = S;
    S_proj(r+1:end, r+1:end) = 0; % Truncate to rank r
    Xl = U * S_proj * V';
end
