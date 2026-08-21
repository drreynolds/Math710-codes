function lobatto_driver(lam, quickMode)
% Main routine to run an implicit 3-node Lobatto finite-difference method
% for solution of a second-order, scalar-valued BVP:
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
% Map from physical/component space to linear algebra index space.
% interval: physical interval index, location: 0=left, 1=midpoint, 2=right,
% component: 0=u or 1=u'.
index = @(interval, location, component) 4*(interval-1) + 2*location + component + 1;

fprintf("Test output from 'index' function for N = 3, M = 14:\n");
for interval = 1:3
    fprintf('\n  interval  %d , (loc,comp,idx):\n', interval);
    for location = 0:2
        for component = 0:1
            fprintf('  ( %d ,  %d ,  %d )\n', location, component, index(interval, location, component)-1);
        end
    end
end

% loop over spatial or temporal resolutions for tests
Nvals = [100, 1000, 10000];
if quickMode
    Nvals = [100, 1000];
end

% run each requested resolution
for n = Nvals
    fprintf('\nImplicit Lobatto-3 FD method for BVP with lambda = %.1f,  N = %d\n', lam, n);

    t = zeros(n+1, 1);
    t(1) = 0.0;
    t(end) = 1.0;
    for j = 1:(n-1)
        t(j+1) = 0.5*(1-cos((2*j-1)*pi/(2*(n-1))));
    end
    % compute and store the analytical solution
    utrue = bvp.utrue(t);

    M = 4*n + 2;
    maxEntries = 22*n + 2;
    % create matrix and right-hand-side storage
    Arows = zeros(maxEntries, 1);
    Acols = zeros(maxEntries, 1);
    Avals = zeros(maxEntries, 1);
    rhs = zeros(M, 1);

    % set up the linear system
    % index(interval,location,component) maps interval data into the global
    % algebraic vector, with location 0=left, 1=midpoint, 2=right and
    % component 0=u, 1=u'.
    idx = 1;
    [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, 1, index(1, 0, 0), 1.0);
    rhs(1) = bvp.ua;
    [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, 2, index(n, 2, 0), 1.0);
    rhs(2) = bvp.ub;

    irow = 3;
    for j = 1:n
        tl = t(j);
        tr = t(j+1);
        th = 0.5*(tl+tr);
        h = tr-tl;

        % First interpolation equation for this Lobatto interval.
        cols = [index(j, 0, 0), index(j, 0, 1), index(j, 1, 0), index(j, 1, 1), index(j, 2, 1)];
        vals = [-24.0, -5.0*h, 24.0, -8.0*h, h];
        [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, irow, cols, vals);
        irow = irow + 1;

        % Enforce the differential equation using the left/mid/right Lobatto data.
        cols = [index(j, 0, 0), index(j, 0, 1), index(j, 1, 0), index(j, 1, 1), index(j, 2, 0), index(j, 2, 1)];
        vals = [-5.0*h*bvp.q(tl), -(24.0 + 5.0*h*bvp.p(tl)), -8.0*h*bvp.q(th), ...
            24.0 - 8.0*h*bvp.p(th), h*bvp.q(tr), h*bvp.p(tr)];
        [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, irow, cols, vals);
        rhs(irow) = h*(5.0*bvp.r(tl) + 8.0*bvp.r(th) - bvp.r(tr));
        irow = irow + 1;

        % Second interpolation equation for this Lobatto interval.
        cols = [index(j, 0, 0), index(j, 0, 1), index(j, 1, 1), index(j, 2, 0), index(j, 2, 1)];
        vals = [-6.0, -h, -4.0*h, 6.0, -h];
        [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, irow, cols, vals);
        irow = irow + 1;

        % Enforce the differential equation with Simpson-like endpoint weights.
        cols = [index(j, 0, 0), index(j, 0, 1), index(j, 1, 0), index(j, 1, 1), index(j, 2, 0), index(j, 2, 1)];
        vals = [-h*bvp.q(tl), -(6.0 + h*bvp.p(tl)), -4.0*h*bvp.q(th), ...
            -4.0*h*bvp.p(th), -h*bvp.q(tr), 6.0 - h*bvp.p(tr)];
        [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, irow, cols, vals);
        rhs(irow) = h*(bvp.r(tl) + 4.0*bvp.r(th) + bvp.r(tr));
        irow = irow + 1;
    end

    % create sparse matrix from accumulated entries
    A = sparse(Arows(1:idx-1), Acols(1:idx-1), Avals(1:idx-1), M, M);
    % solve linear system for the numerical solution
    y = A \ rhs;

    u = y(index(1:n+1, 0, 0));
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
