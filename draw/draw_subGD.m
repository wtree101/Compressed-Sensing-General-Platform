r_values = [1,2,5,10];
T = 1000;
params.T = T;
results = zeros(T,4);

for i = 1:length(r_values)
    r = r_values(i);
    [results(:,i),ss] = onetrial_GD(256, r, kappa, lambda, params);
end

figure;
title("Init 1e-12")
semilogy(results, '-');
legend(arrayfun(@(r) sprintf('r = %d', r), r_values, 'UniformOutput', false));
grid on;