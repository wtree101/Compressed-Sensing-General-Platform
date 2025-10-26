function y = set_zero_outside_range(y_in)
    % Zero out measurements outside 3 standard deviations from the mean
    lower = 1 / sqrt(length(y_in));
    upper = 5 / sqrt(length(y_in));
    y = y_in;
    y(y_in < lower | y_in > upper) = 0;
end