function [grad] = gradient(x, y, Omega, sigma, n1, n2)
    grad = zeros(n1, n2);
    grad(Omega) = (x(Omega) - y(Omega)) .* exp(- ((x(Omega) - y(Omega)) .^ 2) / (sigma ^ 2));
end