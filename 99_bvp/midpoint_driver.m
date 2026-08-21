function midpoint_driver(lam, quickMode)
% Main routine to run an implicit midpoint finite-difference method for solution
% of a second-order, scalar-valued BVP:
%
%    u'' = p(t)*u' + q(t)*u + r(t),  a<t<b,
%    u(a) = ua,  u(b) = ub
%
% where the problem has stiffness that may be adjusted using
% the real-valued parameter lambda<0 [read from the command line]
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
uidx = @(j) 2*j + 1;
vidx = @(j) 2*j + 2;

% loop over spatial or temporal resolutions for tests
Nvals = [100, 1000, 10000];
if quickMode
    Nvals = [100, 1000];
end

% run each requested resolution
for n = Nvals
    fprintf('\nImplicit Midpoint FD method for BVP with lambda = %.1f,  N = %d\n', lam, n);

    t = zeros(n+1, 1);
    t(1) = 0.0;
    t(end) = 1.0;
    for j = 1:(n-1)
        t(j+1) = 0.5*(1-cos((2*j-1)*pi/(2*(n-1))));
    end
    % compute and store the analytical solution
    utrue = bvp.utrue(t);

    M = 2*(n+1);
    maxEntries = 8*(n+1);
    % create matrix and right-hand-side storage
    Arows = zeros(maxEntries, 1);
    Acols = zeros(maxEntries, 1);
    Avals = zeros(maxEntries, 1);
    rhs = zeros(M, 1);

    % set up the linear system
    % y = [u_0, u'_0, u_1, u'_1, ..., u_N, u'_N].
    % The first two rows enforce the Dirichlet boundary data.
    idx = 1;
    [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, 1, uidx(0), 1.0);
    rhs(1) = bvp.ua;

    [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, 2, uidx(n), 1.0);
    rhs(2) = bvp.ub;

    for j = 1:n
        h = t(j+1) - t(j);
        thalf = 0.5*(t(j+1) + t(j));
        alpha = -h*bvp.q(thalf);
        beta = h*bvp.p(thalf);
        gamma = 2*h*bvp.r(thalf);

        % Enforce midpoint consistency:
        % 2*u_j - 2*u_{j-1} - h*u'_j - h*u'_{j-1} = 0.
        row = 2*j + 1;
        cols = [uidx(j), uidx(j-1), vidx(j), vidx(j-1)];
        vals = [2.0, -2.0, -h, -h];
        [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, row, cols, vals);
        rhs(row) = 0.0;

        % Enforce the differential equation at the interval midpoint.
        row = 2*j + 2;
        cols = [uidx(j), uidx(j-1), vidx(j), vidx(j-1)];
        vals = [alpha, alpha, 2.0-beta, -(2.0+beta)];
        [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, row, cols, vals);
        rhs(row) = gamma;
    end

    % create sparse matrix from accumulated entries
    A = sparse(Arows(1:idx-1), Acols(1:idx-1), Avals(1:idx-1), M, M);
    % solve linear system for the numerical solution
    y = A \ rhs;
    u = y(1:2:end);
    % output maximum error against the analytical solution
    uerr = abs(u - utrue);
    fprintf('  Maximum BVP solution error = %.4e\n', norm(uerr, inf));
end
end

function [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, row, cols, vals)
    nvals = numel(vals);
    entries = idx:(idx+nvals-1);
    Arows(entries) = row;
    Acols(entries) = cols(:);
    Avals(entries) = vals(:);
    idx = idx + nvals;
end
