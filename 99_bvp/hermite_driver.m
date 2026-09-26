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
clear

% get lambda from the command line, otherwise set to -10
lam = str2double(input('Enter the stiffness parameter lambda < 0 [default -10]: ', 's'));
if ~isfinite(lam) || ~isreal(lam) || lam >= 0
    fprintf('Invalid or missing lambda, using the default value -10\n');
    lam = -10.0;
end

bvp = BVP(lam);
% Map from physical/component space to linear algebra index space.
% interval: physical interval index, location: 0=left or 1=right,
% component: 0=u or 1=u'.
index = @(interval, location, component) 2*(interval-1) + 2*location + component + 1;

% Hermite basis functions and derivatives
phi1 = @(tleft, h, t) 2*((t-tleft)/h)^3 - 3*((t-tleft)/h)^2 + 1;
dphi1 = @(tleft, h, t) (6/h)*((t-tleft)/h)^2 - 6*(t-tleft)/(h*h);
ddphi1 = @(tleft, h, t) 12*(t-tleft)/(h*h*h) - 6/(h*h);

phi2 = @(tleft, h, t) h*((t-tleft)/h)^3 - 2*h*((t-tleft)/h)^2 + (t-tleft);
dphi2 = @(tleft, h, t) 3*((t-tleft)/h)^2 - 4*(t-tleft)/h + 1;
ddphi2 = @(tleft, h, t) 6*(t-tleft)/(h*h) - 4/h;

phi3 = @(tleft, h, t) -2*((t-tleft)/h)^3 + 3*((t-tleft)/h)^2;
dphi3 = @(tleft, h, t) (-6/h)*((t-tleft)/h)^2 + 6*(t-tleft)/(h*h);
ddphi3 = @(tleft, h, t) -12*(t-tleft)/(h*h*h) + 6/(h*h);

phi4 = @(tleft, h, t) h*((t-tleft)/h)^3 - h*((t-tleft)/h)^2;
dphi4 = @(tleft, h, t) 3*((t-tleft)/h)^2 - 2*(t-tleft)/h;
ddphi4 = @(tleft, h, t) 6*(t-tleft)/(h*h) - 2/h;

fprintf("Test output from 'index' function for N = 3, M = 8:\n");
for interval = 1:3
    fprintf('\n  interval  %d , (loc,comp,idx):\n', interval);
    for location = 0:1
        for component = 0:1
            fprintf('  ( %d ,  %d ,  %d )\n', location, component, index(interval, location, component)-1);
        end
    end
end

% test basis functions
%   check {1,0} properties
delta = 1e-8;
tl = 0.5;
tr = 0.6;
tt = 0.53;
h = tr-tl;
fprintf('\n\nOsculatory interpolation tests:\n');
failed = false;

dtest = phi1(tl,h,tl) - 1;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi1(tleft) error, value = %g\n', dtest + 1);
end
dtest = phi1(tl,h,tr);
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi1(tright) error, value = %g\n', dtest);
end
dtest = (phi1(tl,h,tl+delta) - phi1(tl,h,tl))/delta;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi1''(tleft) error, value = %g\n', dtest);
end
dtest = (phi1(tl,h,tr+delta) - phi1(tl,h,tr))/delta;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi1''(tright) error, value = %g\n', dtest);
end

dtest = phi2(tl,h,tl);
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi2(tleft) error, value = %g\n', dtest);
end
dtest = phi2(tl,h,tr);
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi2(tright) error, value = %g\n', dtest);
end
dtest = (phi2(tl,h,tl+delta) - phi2(tl,h,tl))/delta - 1;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi2''(tleft) error, value = %g\n', dtest + 1);
end
dtest = (phi2(tl,h,tr+delta) - phi2(tl,h,tr))/delta;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi2''(tright) error, value = %g\n', dtest);
end

dtest = phi3(tl,h,tl);
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi3(tleft) error, value = %g\n', dtest);
end
dtest = phi3(tl,h,tr) - 1;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi3(tright) error, value = %g\n', dtest + 1);
end
dtest = (phi3(tl,h,tl+delta) - phi3(tl,h,tl))/delta;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi3''(tleft) error, value = %g\n', dtest);
end
dtest = (phi3(tl,h,tr+delta) - phi3(tl,h,tr))/delta;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi3''(tright) error, value = %g\n', dtest);
end

dtest = phi4(tl,h,tl);
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi4(tleft) error, value = %g\n', dtest);
end
dtest = phi4(tl,h,tr);
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi4(tright) error, value = %g\n', dtest);
end
dtest = (phi4(tl,h,tl+delta) - phi4(tl,h,tl))/delta;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi4''(tleft) error, value = %g\n', dtest);
end
dtest = (phi4(tl,h,tr+delta) - phi4(tl,h,tr))/delta - 1;
if abs(dtest) > 1e-4
    failed = true;
    fprintf('    phi4''(tright) error, value = %g\n', dtest + 1);
end
if ~failed
    fprintf('  all tests pass\n');
end

%   check analytical derivative tests
fprintf('Derivative tests:\n');
failed = false;

dtest = (phi1(tl,h,tt+delta)-phi1(tl,h,tt))/delta;
if abs(dtest - dphi1(tl,h,tt)) > 1e-4
    failed = true;
    fprintf('  dphi1 error, value = %g, approx = %g\n', dphi1(tl,h,tt), dtest);
end
dtest = (dphi1(tl,h,tt+delta)-dphi1(tl,h,tt))/delta;
if abs(dtest - ddphi1(tl,h,tt)) > 1e-4
    failed = true;
    fprintf('  ddphi1 error, value = %g, approx = %g\n', ddphi1(tl,h,tt), dtest);
end

dtest = (phi2(tl,h,tt+delta)-phi2(tl,h,tt))/delta;
if abs(dtest - dphi2(tl,h,tt)) > 1e-4
    failed = true;
    fprintf('  dphi2 error, value = %g, approx = %g\n', dphi2(tl,h,tt), dtest);
end
dtest = (dphi2(tl,h,tt+delta)-dphi2(tl,h,tt))/delta;
if abs(dtest - ddphi2(tl,h,tt)) > 1e-4
    failed = true;
    fprintf('  ddphi2 error, value = %g, approx = %g\n', ddphi2(tl,h,tt), dtest);
end

dtest = (phi3(tl,h,tt+delta)-phi3(tl,h,tt))/delta;
if abs(dtest - dphi3(tl,h,tt)) > 1e-4
    failed = true;
    fprintf('  dphi3 error, value = %g, approx = %g\n', dphi3(tl,h,tt), dtest);
end
dtest = (dphi3(tl,h,tt+delta)-dphi3(tl,h,tt))/delta;
if abs(dtest - ddphi3(tl,h,tt)) > 1e-4
    failed = true;
    fprintf('  ddphi3 error, value = %g, approx = %g\n', ddphi3(tl,h,tt), dtest);
end

dtest = (phi4(tl,h,tt+delta)-phi4(tl,h,tt))/delta;
if abs(dtest - dphi4(tl,h,tt)) > 1e-4
    failed = true;
    fprintf('  dphi4 error, value = %g, approx = %g\n', dphi4(tl,h,tt), dtest);
end
dtest = (dphi4(tl,h,tt+delta)-dphi4(tl,h,tt))/delta;
if abs(dtest - ddphi4(tl,h,tt)) > 1e-4
    failed = true;
    fprintf('  ddphi4 error, value = %g, approx = %g\n', ddphi4(tl,h,tt), dtest);
end
if ~failed
    fprintf('  all tests pass\n');
end

% loop over spatial resolutions for tests
N = [100, 1000, 10000];
for n = N

    % output problem information
    fprintf('\nPiecewise Hermite FD method for BVP with lambda = %.1f,  N = %d\n', lam, n);

    % compute/store analytical solution
    t = zeros(n+1, 1);
    t(1) = 0;
    t(n+1) = 1;
    for j = 1:n-1
        t(j+1) = 0.5*(1-cos((2*j-1)*pi/(2*(n-1))));
    end
    utrue = zeros(n+1, 1);
    for j = 1:n+1
        utrue(j) = bvp.utrue(t(j));
    end

    % set integer for overall linear algebra problem size
    M = 2*n+2;

    % create matrix and right-hand side vectors
    Arows = zeros(8*n+2, 1);
    Acols = zeros(8*n+2, 1);
    Avals = zeros(8*n+2, 1);
    b = zeros(M, 1);

    % set up linear system:
    %    recall 'index' usage: index(interval,location,component)
    %      interval:  physical interval index [1 <= interval <= N]
    %      location:  location in interval [0=left, 1=right]
    %      component: solution component at this location [0=u, 1=u']
    %    recall [dd]phiN usage: [dd]phiN(tleft, h, t)
    idx = 1;
    Arows(idx) = 1;        % A(1,index(1,0,0))
    Acols(idx) = index(1,0,0);
    Avals(idx) = 1;
    idx = idx + 1;
    b(1) = bvp.ua;

    Arows(idx) = 2;        % A(2,index(n,1,0))
    Acols(idx) = index(n,1,0);
    Avals(idx) = 1;
    idx = idx + 1;
    b(2) = bvp.ub;

    irow = 3;
    for j = 1:n

        % setup interval-specific information
        tl = t(j);
        tr = t(j+1);
        h = tr-tl;
        eta1 = 0.5*(tr+tl) - h/2/sqrt(3);
        eta2 = 0.5*(tr+tl) + h/2/sqrt(3);
        q1 = bvp.q(eta1);
        q2 = bvp.q(eta2);
        p1 = bvp.p(eta1);
        p2 = bvp.p(eta2);

        % setup first equation for this interval: enforce ODE at eta1
        Arows(idx) = irow;        % A(irow,index(j,0,0))
        Acols(idx) = index(j,0,0);
        Avals(idx) = ddphi1(tl,h,eta1) - p1*dphi1(tl,h,eta1) - q1*phi1(tl,h,eta1);
        idx = idx + 1;

        Arows(idx) = irow;        % A(irow,index(j,0,1))
        Acols(idx) = index(j,0,1);
        Avals(idx) = ddphi2(tl,h,eta1) - p1*dphi2(tl,h,eta1) - q1*phi2(tl,h,eta1);
        idx = idx + 1;

        Arows(idx) = irow;        % A(irow,index(j,1,0))
        Acols(idx) = index(j,1,0);
        Avals(idx) = ddphi3(tl,h,eta1) - p1*dphi3(tl,h,eta1) - q1*phi3(tl,h,eta1);
        idx = idx + 1;

        Arows(idx) = irow;        % A(irow,index(j,1,1))
        Acols(idx) = index(j,1,1);
        Avals(idx) = ddphi4(tl,h,eta1) - p1*dphi4(tl,h,eta1) - q1*phi4(tl,h,eta1);
        idx = idx + 1;

        b(irow) = bvp.r(eta1);
        irow = irow + 1;

        % setup second equation for this interval: enforce ODE at eta2
        Arows(idx) = irow;        % A(irow,index(j,0,0))
        Acols(idx) = index(j,0,0);
        Avals(idx) = ddphi1(tl,h,eta2) - p2*dphi1(tl,h,eta2) - q2*phi1(tl,h,eta2);
        idx = idx + 1;

        Arows(idx) = irow;        % A(irow,index(j,0,1))
        Acols(idx) = index(j,0,1);
        Avals(idx) = ddphi2(tl,h,eta2) - p2*dphi2(tl,h,eta2) - q2*phi2(tl,h,eta2);
        idx = idx + 1;

        Arows(idx) = irow;        % A(irow,index(j,1,0))
        Acols(idx) = index(j,1,0);
        Avals(idx) = ddphi3(tl,h,eta2) - p2*dphi3(tl,h,eta2) - q2*phi3(tl,h,eta2);
        idx = idx + 1;

        Arows(idx) = irow;        % A(irow,index(j,1,1))
        Acols(idx) = index(j,1,1);
        Avals(idx) = ddphi4(tl,h,eta2) - p2*dphi4(tl,h,eta2) - q2*phi4(tl,h,eta2);
        idx = idx + 1;

        b(irow) = bvp.r(eta2);
        irow = irow + 1;

    end

    A = sparse(Arows(1:idx-1), Acols(1:idx-1), Avals(1:idx-1), M, M);

    % solve linear system for BVP solution
    y = A \ b;

    % output maximum error
    u = zeros(n+1, 1);
    for j = 1:n+1
        u(j) = y(index(j,0,0));
    end
    uerr = abs(u-utrue);
    fprintf('  Maximum BVP solution error = %.4e\n', max(uerr));

end

