function [X, Y, R] = RK_stability(B, box, N)
    % Usage: X,Y = RK_stability(B, box, N)
    %
    %        Inputs:
    %          B is a Butcher table, with components:
    %             B.A -- the Butcher table matrix
    %             B.b -- the solution coefficients
    %          box = [xl, xr, yl, yr] is the bounding box for the sub-region
    %              of the complex plane in which to perform the test
    %          N is optional, specifying how many sample sub-region points to use
    %
    %        Outputs:
    %          (X, Y) where X is an array of real components of the stability boundary
    %            and Y is an array of imaginary components of the stability boundary
    %
    %        We consider the RK stability function
    %          R(eta) = 1 + eta * dot(b, inv(I-eta*A)*e)
    %
    %        We sample the values in 'box' within the complex plane, plugging
    %        each value into |R(eta)|, and plot the contour of this function
    %        having value 1.
    %
% Function to generate and plot the linear stability regions for Runge--Kutta
% methods.  Includes a simple "main" that uses this function to plot the
% stability region for forward and backward Euler (when posed as RK methods).
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
if nargin < 3 || isempty(N)
    N = 1000;
end

A = B.A;
b = B.b(:);
s = numel(b);
e = ones(s, 1);
I = eye(s);

% Sample the requested rectangle in the complex eta-plane.
x = linspace(box(1), box(2), N);
y = linspace(box(3), box(4), N);
R = zeros(N, N);

for j = 1:N
    for i = 1:N
        eta = x(i) + 1i*y(j);
        % Evaluate the RK stability function without explicitly forming an inverse.
        R(j,i) = abs(1.0 + eta*(b.'*((I - eta*A) \ e)));
    end
end

% Return the longest contour segment satisfying |R(eta)| = 1.
C = contourc(x, y, R, [1.0, 1.0]);
[X, Y] = longestContourSegment(C);
end

function [X, Y] = longestContourSegment(C)
    X = [];
    Y = [];
    bestLen = 0;
    col = 1;
    while col < size(C, 2)
        npts = C(2, col);
        pts = C(:, col+1:col+npts);
        if npts > bestLen
            bestLen = npts;
            X = pts(1, :).';
            Y = pts(2, :).';
        end
        col = col + npts + 1;
    end
end
