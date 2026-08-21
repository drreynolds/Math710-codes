function hermite_driver(lam, quickMode)
% Main routine to run a piecewise Hermite finite-difference method
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
% interval: physical interval index, location: 0=left or 1=right,
% component: 0=u or 1=u'.
index = @(interval, location, component) 2*(interval-1) + 2*location + component + 1;

fprintf("Test output from 'index' function for N = 3, M = 8:\n");
for interval = 1:3
    fprintf('\n  interval  %d , (loc,comp,idx):\n', interval);
    for location = 0:1
        for component = 0:1
            fprintf('  ( %d ,  %d ,  %d )\n', location, component, index(interval, location, component)-1);
        end
    end
end

testHermiteBasis();

% loop over spatial or temporal resolutions for tests
Nvals = [100, 1000, 10000];
if quickMode
    Nvals = [100, 1000];
end

% run each requested resolution
for n = Nvals
    fprintf('\nPiecewise Hermite FD method for BVP with lambda = %.1f,  N = %d\n', lam, n);

    t = zeros(n+1, 1);
    t(1) = 0.0;
    t(end) = 1.0;
    for j = 1:(n-1)
        t(j+1) = 0.5*(1-cos((2*j-1)*pi/(2*(n-1))));
    end
    % compute and store the analytical solution
    utrue = bvp.utrue(t);

    M = 2*n + 2;
    maxEntries = 8*n + 2;
    % create matrix and right-hand-side storage
    Arows = zeros(maxEntries, 1);
    Acols = zeros(maxEntries, 1);
    Avals = zeros(maxEntries, 1);
    rhs = zeros(M, 1);

    % set up the linear system
    % index(interval,location,component) maps interval data into the global
    % algebraic vector, with location 0=left, 1=right and component 0=u, 1=u'.
    idx = 1;
    [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, 1, index(1, 0, 0), 1.0);
    rhs(1) = bvp.ua;
    [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, 2, index(n, 1, 0), 1.0);
    rhs(2) = bvp.ub;

    irow = 3;
    for j = 1:n
        tl = t(j);
        tr = t(j+1);
        h = tr-tl;
        eta1 = 0.5*(tr+tl) - h/(2.0*sqrt(3.0));
        eta2 = 0.5*(tr+tl) + h/(2.0*sqrt(3.0));
        q1 = bvp.q(eta1);
        q2 = bvp.q(eta2);
        p1 = bvp.p(eta1);
        p2 = bvp.p(eta2);

        % Enforce the differential equation at the left Gauss point.
        [Arows, Acols, Avals, idx] = addHermiteEntry(Arows, Acols, Avals, idx, irow, j, tl, h, eta1, p1, q1, index);
        rhs(irow) = bvp.r(eta1);
        irow = irow + 1;

        % Enforce the differential equation at the right Gauss point.
        [Arows, Acols, Avals, idx] = addHermiteEntry(Arows, Acols, Avals, idx, irow, j, tl, h, eta2, p2, q2, index);
        rhs(irow) = bvp.r(eta2);
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

function [Arows, Acols, Avals, idx] = addHermiteEntry(Arows, Acols, Avals, idx, row, j, tl, h, t, pval, qval, index)
    % Apply u'' - p(t)u' - q(t)u = r(t) to each Hermite basis function.
    vals = [
        ddphi1(tl, h, t) - pval*dphi1(tl, h, t) - qval*phi1(tl, h, t);
        ddphi2(tl, h, t) - pval*dphi2(tl, h, t) - qval*phi2(tl, h, t);
        ddphi3(tl, h, t) - pval*dphi3(tl, h, t) - qval*phi3(tl, h, t);
        ddphi4(tl, h, t) - pval*dphi4(tl, h, t) - qval*phi4(tl, h, t)
    ];
    cols = [
        index(j, 0, 0);
        index(j, 0, 1);
        index(j, 1, 0);
        index(j, 1, 1)
    ];
    [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, row, cols, vals);
end

function testHermiteBasis()
    delta = 1e-8;
    tl = 0.5;
    tr = 0.6;
    tt = 0.53;
    h = tr-tl;

    fprintf('\n\nOsculatory interpolation tests:\n');
    failed = false;
    failed = checkValue(failed, phi1(tl,h,tl) - 1.0, 1e-4, '    phi1(tleft) error, value = %.16e\n', phi1(tl,h,tl));
    failed = checkValue(failed, phi1(tl,h,tr), 1e-4, '    phi1(tright) error, value = %.16e\n', phi1(tl,h,tr));
    failed = checkValue(failed, (phi1(tl,h,tl+delta)-phi1(tl,h,tl))/delta, 1e-4, "    phi1'(tleft) error, value = %.16e\n", (phi1(tl,h,tl+delta)-phi1(tl,h,tl))/delta);
    failed = checkValue(failed, (phi1(tl,h,tr+delta)-phi1(tl,h,tr))/delta, 1e-4, "    phi1'(tright) error, value = %.16e\n", (phi1(tl,h,tr+delta)-phi1(tl,h,tr))/delta);
    failed = checkValue(failed, phi2(tl,h,tl), 1e-4, '    phi2(tleft) error, value = %.16e\n', phi2(tl,h,tl));
    failed = checkValue(failed, phi2(tl,h,tr), 1e-4, '    phi2(tright) error, value = %.16e\n', phi2(tl,h,tr));
    dtest = (phi2(tl,h,tl+delta)-phi2(tl,h,tl))/delta - 1.0;
    failed = checkValue(failed, dtest, 1e-4, "    phi2'(tleft) error, value = %.16e\n", dtest + 1.0);
    failed = checkValue(failed, (phi2(tl,h,tr+delta)-phi2(tl,h,tr))/delta, 1e-4, "    phi2'(tright) error, value = %.16e\n", (phi2(tl,h,tr+delta)-phi2(tl,h,tr))/delta);
    failed = checkValue(failed, phi3(tl,h,tl), 1e-4, '    phi3(tleft) error, value = %.16e\n', phi3(tl,h,tl));
    failed = checkValue(failed, phi3(tl,h,tr) - 1.0, 1e-4, '    phi3(tright) error, value = %.16e\n', phi3(tl,h,tr));
    failed = checkValue(failed, (phi3(tl,h,tl+delta)-phi3(tl,h,tl))/delta, 1e-4, "    phi3'(tleft) error, value = %.16e\n", (phi3(tl,h,tl+delta)-phi3(tl,h,tl))/delta);
    failed = checkValue(failed, (phi3(tl,h,tr+delta)-phi3(tl,h,tr))/delta, 1e-4, "    phi3'(tright) error, value = %.16e\n", (phi3(tl,h,tr+delta)-phi3(tl,h,tr))/delta);
    failed = checkValue(failed, phi4(tl,h,tl), 1e-4, '    phi4(tleft) error, value = %.16e\n', phi4(tl,h,tl));
    failed = checkValue(failed, phi4(tl,h,tr), 1e-4, '    phi4(tright) error, value = %.16e\n', phi4(tl,h,tr));
    failed = checkValue(failed, (phi4(tl,h,tl+delta)-phi4(tl,h,tl))/delta, 1e-4, "    phi4'(tleft) error, value = %.16e\n", (phi4(tl,h,tl+delta)-phi4(tl,h,tl))/delta);
    dtest = (phi4(tl,h,tr+delta)-phi4(tl,h,tr))/delta - 1.0;
    failed = checkValue(failed, dtest, 1e-4, "    phi4'(tright) error, value = %.16e\n", dtest + 1.0);
    if ~failed
        fprintf('  all tests pass\n');
    end

    fprintf('Derivative tests:\n');
    failed = false;
    failed = checkValue(failed, (phi1(tl,h,tt+delta)-phi1(tl,h,tt))/delta - dphi1(tl,h,tt), 1e-4, '  dphi1 error, value = %.16e\n', dphi1(tl,h,tt));
    failed = checkValue(failed, (dphi1(tl,h,tt+delta)-dphi1(tl,h,tt))/delta - ddphi1(tl,h,tt), 1e-4, '  ddphi1 error, value = %.16e\n', ddphi1(tl,h,tt));
    failed = checkValue(failed, (phi2(tl,h,tt+delta)-phi2(tl,h,tt))/delta - dphi2(tl,h,tt), 1e-4, '  dphi2 error, value = %.16e\n', dphi2(tl,h,tt));
    failed = checkValue(failed, (dphi2(tl,h,tt+delta)-dphi2(tl,h,tt))/delta - ddphi2(tl,h,tt), 1e-4, '  ddphi2 error, value = %.16e\n', ddphi2(tl,h,tt));
    failed = checkValue(failed, (phi3(tl,h,tt+delta)-phi3(tl,h,tt))/delta - dphi3(tl,h,tt), 1e-4, '  dphi3 error, value = %.16e\n', dphi3(tl,h,tt));
    failed = checkValue(failed, (dphi3(tl,h,tt+delta)-dphi3(tl,h,tt))/delta - ddphi3(tl,h,tt), 1e-4, '  ddphi3 error, value = %.16e\n', ddphi3(tl,h,tt));
    failed = checkValue(failed, (phi4(tl,h,tt+delta)-phi4(tl,h,tt))/delta - dphi4(tl,h,tt), 1e-4, '  dphi4 error, value = %.16e\n', dphi4(tl,h,tt));
    failed = checkValue(failed, (dphi4(tl,h,tt+delta)-dphi4(tl,h,tt))/delta - ddphi4(tl,h,tt), 1e-4, '  ddphi4 error, value = %.16e\n', ddphi4(tl,h,tt));
    if ~failed
        fprintf('  all tests pass\n');
    end
end

function failed = checkValue(failed, err, tol, msg, val)
    if abs(err) > tol
        failed = true;
        fprintf(msg, val);
    end
end

function val = phi1(tleft, h, t)
    val = 2*((t-tleft)/h)^3 - 3*((t-tleft)/h)^2 + 1;
end

function val = dphi1(tleft, h, t)
    val = (6/h)*((t-tleft)/h)^2 - 6*(t-tleft)/(h*h);
end

function val = ddphi1(tleft, h, t)
    val = 12*(t-tleft)/(h*h*h) - 6/(h*h);
end

function val = phi2(tleft, h, t)
    val = h*((t-tleft)/h)^3 - 2*h*((t-tleft)/h)^2 + (t-tleft);
end

function val = dphi2(tleft, h, t)
    val = 3*((t-tleft)/h)^2 - 4*(t-tleft)/h + 1;
end

function val = ddphi2(tleft, h, t)
    val = 6*(t-tleft)/(h*h) - 4/h;
end

function val = phi3(tleft, h, t)
    val = -2*((t-tleft)/h)^3 + 3*((t-tleft)/h)^2;
end

function val = dphi3(tleft, h, t)
    val = (-6/h)*((t-tleft)/h)^2 + 6*(t-tleft)/(h*h);
end

function val = ddphi3(tleft, h, t)
    val = -12*(t-tleft)/(h*h*h) + 6/(h*h);
end

function val = phi4(tleft, h, t)
    val = h*((t-tleft)/h)^3 - h*((t-tleft)/h)^2;
end

function val = dphi4(tleft, h, t)
    val = 3*((t-tleft)/h)^2 - 2*(t-tleft)/h;
end

function val = ddphi4(tleft, h, t)
    val = 6*(t-tleft)/(h*h) - 2/h;
end

function [Arows, Acols, Avals, idx] = addEntries(Arows, Acols, Avals, idx, row, cols, vals)
    nvals = numel(vals);
    entries = idx:(idx+nvals-1);
    Arows(entries) = row;
    Acols(entries) = cols(:);
    Avals(entries) = vals(:);
    idx = idx + nvals;
end
