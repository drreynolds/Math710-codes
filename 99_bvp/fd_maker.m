function [coeffs, errorterm] = fd_maker(stencil, deriv)
    % Usage: coeffs, errorterm = fd_maker(stencil, deriv)
    %
    % Utility to compute classical finite difference approximation to a requested derivative
    % using a specified set of nodes.  We assume that all nodes are evenly spaced, with a
    % spacing of 'h', and that the derivative is requested at the node "0", meaning that the
    % location of each node may be uniquely specified by a "stencil" of offsets from the
    % derivative location.
    %
    % Inputs:  stencil = array of integer offsets from node "0" that
    %                     will be used in approximation, e.g. [-1, 0, 1]
    %                     for f(x-h), f(x) and f(x+h)
    %          deriv = integer specifying the desired derivative
    %
    % Outputs: coeffs = row vector of finite-difference coefficients s.t.
    %                      f^(deriv) \approx \sum coeffs(i)*f(x+stencil(i)*h)
    %          errorterm = leading error term in derivative approximation
    %
% fd_maker.m
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
n = numel(stencil);
% Need at least deriv+1 stencil entries to match derivatives through this order.
if deriv > n-1
    error('fd_maker:InvalidStencil', 'not enough stencil entries for requested derivative');
end

stencil = stencil(:)';
% Moment equations enforce exactness on monomials up through degree n-1.
A = zeros(n, n);
for i = 1:n
    A(i, :) = stencil.^(i-1);
end

rhs = zeros(n, 1);
rhs(deriv+1) = factorial(deriv);
coeffs = (A \ rhs)';

errorterm = struct('power', [], 'coefficient', []);
% The first unmet moment condition gives the leading truncation-error term.
for power = 0:(2*n-1)
    moment = sum(coeffs .* stencil.^power);
    target = 0.0;
    if power == deriv
        target = factorial(deriv);
    end
    residual = moment - target;
    if abs(residual) > 1e-10*max(1.0, norm(coeffs, inf))
        errorterm.power = power - deriv;
        errorterm.coefficient = residual/factorial(power);
        return;
    end
end
end
