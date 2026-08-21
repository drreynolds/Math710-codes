function stencil_driver(lam, quickMode)
% Main routine to run a stencil-based finite-difference method for solution of a
% second-order, scalar-valued BVP:
%
%    u'' = p(t)*u' + q(t)*u + r(t),  a<t<b,
%    u(a) = ua,  u(b) = ub
%
% where the problem has stiffness that may be adjusted using
% the real-valued parameter lambda<0 [read from the command line]
%
% This driver attempts to solve the problem using a second-order, stencil-based
% finite-difference approximation.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
% get optional inputs, otherwise use default values
if nargin < 1 || isempty(lam)
    lam = -10.0;
end
if nargin < 2 || isempty(quickMode)
    quickMode = false;
end

bvp = BVP(lam);
% loop over spatial or temporal resolutions for tests
Nvals = [100, 1000, 10000];
if quickMode
    Nvals = [100, 1000];
end

% run each requested resolution
for n = Nvals
    fprintf('\nStencil-based FD method for BVP with lambda = %.1f,  N = %d\n', lam, n);

    t = linspace(bvp.a, bvp.b, n+1)';
    h = t(2) - t(1);
    % compute and store the analytical solution
    utrue = bvp.utrue(t);

    % create matrix and right-hand-side storage
    rhs = zeros(n+1, 1);

    % set up the linear system
    rhs(1) = bvp.ua;
    rhs(n+1) = bvp.ub;

    % Interior rows contain the centered second-order stencil for
    % u'' = p(t)u' + q(t)u + r(t); boundary rows enforce u(a), u(b).
    lower = zeros(n+1, 1);
    diagv = zeros(n+1, 1);
    upper = zeros(n+1, 1);
    diagv([1, n+1]) = 1.0;
    j = (2:n).';
    lower(j-1) = -1.0 - 0.5*h*bvp.p(t(j));
    diagv(j) = 2.0 + h*h*bvp.q(t(j));
    upper(j+1) = -1.0 + 0.5*h*bvp.p(t(j));
    rhs(j) = -h*h*bvp.r(t(j));

    % create sparse matrix from accumulated diagonal entries
    A = spdiags([lower, diagv, upper], [-1, 0, 1], n+1, n+1);
    % solve linear system for BVP solution
    u = A \ rhs;
    % output maximum error against the analytical solution
    uerr = abs(u - utrue);
    fprintf('  Maximum BVP solution error = %.4e\n', norm(uerr, inf));
end
end
