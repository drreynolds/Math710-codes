function [x, y] = LMM_stability(thetas, a, b)
    % Usage: x, y = LMM_stability(thetas,a,b):
    %
    % This function evaluates the boundary of the stability region for the LMM
    % defined by the arrays a and b, for a given set of input angles, thetas.
    %
    % Inputs:   thetas - angles in the complex plane
    %           a, b - LMM arrays alpha_i and beta_i
    % Outputs:  x, y - coordinate locations in the complex plane: x+i*y
    %
% Function to generate and plot the linear stability regions for linear multistep
% methods.  Includes a "main" that uses this function to plot overlaid stability
% regions for Adams-Bashforth, Adams-Moulton, and Backwards Differentiation Formulas.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

thetas = thetas(:);
a = a(:).';
b = b(:).';

% Allocate one complex stability-boundary sample for each input angle.
x = zeros(size(thetas));
y = zeros(size(thetas));

n = numel(a) - 1;
m = numel(b) - 1;
k = max(n, m);

for ith = 1:numel(thetas)
    % Parameterize the root condition with xi on the unit circle.
    xi = exp(1i*thetas(ith));

    % Evaluate the numerator rho(xi) and denominator sigma(xi).
    num = 0.0;
    for j = 0:n
        num = num + a(j+1)*xi^(k-j);
    end

    den = 0.0;
    for j = 0:m
        den = den + b(j+1)*xi^(k-j);
    end

    % The stability boundary satisfies eta = rho(xi)/sigma(xi).
    eta = num / den;
    x(ith) = real(eta);
    y(ith) = imag(eta);
end
end
