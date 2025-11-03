function y = set_zero_outside_range_tensor(y_in)
    % Zero out measurements outside 3 standard deviations from the mean
    lower = 0 / sqrt(length(y_in));
    upper = 20 / sqrt(length(y_in));
    y = y_in;
    y(y_in < lower | y_in > upper) = 0;
    %y = y.^2; % seems that we should make a square for spectral method, otherwise very large when m is large.
    % though scale does not effect. but y^2 would be more natual than |y|?
end