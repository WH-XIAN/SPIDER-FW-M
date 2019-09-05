function [output] = Object(x, y, Omega, sigma)
    output = 0.5 * sigma ^ 2 * sum(1 - exp(- ((x(Omega) - y(Omega)) .^ 2) / (sigma ^ 2)));
end