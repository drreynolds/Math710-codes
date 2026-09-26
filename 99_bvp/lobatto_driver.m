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
clear
% get lambda from the command line, otherwise set to -10
lam = str2double(input('Enter the stiffness parameter lambda < 0 [default -10]: ', 's'));
if ~isfinite(lam) || ~isreal(lam) || lam >= 0
    fprintf('Invalid or missing lambda, using the default value -10\n');
    lam = -10.0;
end

% create BVP object
bvp = BVP(lam);

% utility routine to map from physical/component space to linear algebra index space
%    interval:  physical interval index [1 <= interval <= N]
%    location:  location in interval [0=left, 1=midpoint, 2=right]
%    component: solution component at this location [0=u, 1=u']
index = @(interval, location, component) 4*(interval-1) + 2*location + component + 1;

% test 'index' function by outputting mapping for small N
fprintf("Test output from 'index' function for N = 3, M = 14:\n");
for interval = 1:3
    fprintf('\n  interval  %d , (loc,comp,idx):\n', interval);
    for location = 0:2
        for component = 0:1
            fprintf('  ( %d ,  %d ,  %d )\n', location, component, index(interval,location,component)-1);
        end
    end
end

% loop over spatial resolutions for tests
N = [100, 1000, 10000];
for n = N

    % output problem information
    fprintf('\nImplicit Lobatto-3 FD method for BVP with lambda = %.1f,  N = %i\n', lam, n);

    % compute/store analytical solution
    t = zeros(n+1, 1);
    t(1) = 0;
    t(n+1) = 1.0;
    for j = 1:n-1
        t(j+1) = 0.5*(1-cos((2*j-1)*pi/(2*(n-1))));
    end
    utrue = bvp.utrue(t);

    % set integer for overall linear algebra problem size
    M = 4*n+2;

    % create matrix and right-hand side vectors
    Arows = zeros(22*n+2, 1);
    Acols = zeros(22*n+2, 1);
    Avals = zeros(22*n+2, 1);
    b = zeros(M, 1);

    % set up linear system
    %    recall 'index' usage: index(interval,location,component)
    %      interval:  physical interval index [1 <= interval <= N]
    %      location:  location in interval [0=left, 1=midpoint, 2=right]
    %      component: solution component at this location [0=u, 1=u']
    idx = 1;
    Arows(idx) = 1;       % A(1,index(1,0,0))
    Acols(idx) = index(1,0,0);
    Avals(idx) = 1.0;
    idx = idx + 1;
    b(1) = bvp.ua;

    Arows(idx) = 2;       % A(2,index(n,2,0))
    Acols(idx) = index(n,2,0);
    Avals(idx) = 1.0;
    idx = idx + 1;
    b(2) = bvp.ub;

    irow = 3;
    for j = 1:n

        % setup interval-specific information
        tl = t(j);
        tr = t(j+1);
        th = 0.5*(tl+tr);
        h = tr-tl;

        % setup first equation for this interval:
        %    -24*y_{j-1,0} - 5*h*y_{j-1,1} + 24*y_{j-1/2,0} - 8*h*y_{j-1/2,1} + h*y_{j,1} = 0
        Arows(idx) = irow;       % A(irow,index(j,0,0))
        Acols(idx) = index(j,0,0);
        Avals(idx) = -24;
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,0,1))
        Acols(idx) = index(j,0,1);
        Avals(idx) = -5*h;
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,1,0))
        Acols(idx) = index(j,1,0);
        Avals(idx) = 24;
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,1,1))
        Acols(idx) = index(j,1,1);
        Avals(idx) = -8*h;
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,2,1))
        Acols(idx) = index(j,2,1);
        Avals(idx) = h;
        idx = idx + 1;

        b(irow) = 0;
        irow = irow + 1;

        % setup second equation for this interval:
        %    -5*h*q_{j-1}*y_{j-1,0} - (24+5*h*p_{j-1})*y_{j-1,1} - 8*h*q_{j-1/2}*y_{j-1/2,0}
        %      + (24-8*h*p_{j-1/2})*y_{j-1/2,1} + h*q_{j}*y_{j,0} + h*p_{j}*y_{j,1} = h*(5*r_{j-1}+8*r_{j-1/2}-r_{j})
        Arows(idx) = irow;       % A(irow,index(j,0,0))
        Acols(idx) = index(j,0,0);
        Avals(idx) = -5*h*bvp.q(tl);
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,0,1))
        Acols(idx) = index(j,0,1);
        Avals(idx) = -(24 + 5*h*bvp.p(tl));
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,1,0))
        Acols(idx) = index(j,1,0);
        Avals(idx) = -8*h*bvp.q(th);
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,1,1))
        Acols(idx) = index(j,1,1);
        Avals(idx) = (24-8*h*bvp.p(th));
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,2,0))
        Acols(idx) = index(j,2,0);
        Avals(idx) = h*bvp.q(tr);
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,2,1))
        Acols(idx) = index(j,2,1);
        Avals(idx) = h*bvp.p(tr);
        idx = idx + 1;

        b(irow) = h*(5*bvp.r(tl) + 8*bvp.r(th) - bvp.r(tr));
        irow = irow + 1;

        % setup third equation for this interval:
        %    -6*y_{j-1,0} - h*y_{j-1,1} - 4*h*y_{j-1/2,1} + 6*y_{j,0} - h*y_{j,1} = 0
        Arows(idx) = irow;       % A(irow,index(j,0,0))
        Acols(idx) = index(j,0,0);
        Avals(idx) = -6;
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,0,1))
        Acols(idx) = index(j,0,1);
        Avals(idx) = -h;
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,1,1))
        Acols(idx) = index(j,1,1);
        Avals(idx) = -4*h;
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,2,0))
        Acols(idx) = index(j,2,0);
        Avals(idx) = 6;
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,2,1))
        Acols(idx) = index(j,2,1);
        Avals(idx) = -h;
        idx = idx + 1;

        b(irow) = 0;
        irow = irow + 1;

        % setup fourth equation for this interval:
        %    -h*q_{j-1}*y_{j-1,0} - (6+h*p_{j-1})*y_{j-1,1} - 4*h*q_{j-1/2}*y_{j-1/2,0} - 4*h*p_{j-1/2}*y_{j-1/2,1}
        %       - h*q_j*y_{j,0} + (6-h*p_j)*y_{j,1} = h*(r_{j-1} + 4*r_{j-1/2} + r_{j})
        Arows(idx) = irow;       % A(irow,index(j,0,0))
        Acols(idx) = index(j,0,0);
        Avals(idx) = -h*bvp.q(tl);
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,0,1))
        Acols(idx) = index(j,0,1);
        Avals(idx) = -(6 + h*bvp.p(tl));
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,1,0))
        Acols(idx) = index(j,1,0);
        Avals(idx) = -4*h*bvp.q(th);
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,1,1))
        Acols(idx) = index(j,1,1);
        Avals(idx) = -4*h*bvp.p(th);
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,2,0))
        Acols(idx) = index(j,2,0);
        Avals(idx) = -h*bvp.q(tr);
        idx = idx + 1;

        Arows(idx) = irow;       % A(irow,index(j,2,1))
        Acols(idx) = index(j,2,1);
        Avals(idx) = (6-h*bvp.p(tr));
        idx = idx + 1;

        b(irow) = h*(bvp.r(tl) + 4*bvp.r(th) + bvp.r(tr));
        irow = irow + 1;
    end

    A = sparse(Arows, Acols, Avals, M, M);

    % solve linear system for BVP solution
    y = A \ b;

    % output maximum error
    u = y(index(1:n+1,0,0));
    uerr = abs(u-utrue);
    fprintf('  Maximum BVP solution error = %.4e\n', max(uerr));
end
