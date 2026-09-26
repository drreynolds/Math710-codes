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
clear
% get lambda from the command line, otherwise set to -10
lam = str2double(input('Enter the stiffness parameter lambda < 0 [default -10]: ', 's'));
if ~isfinite(lam) || ~isreal(lam) || lam >= 0
    fprintf('Invalid or missing lambda, using the default value -10\n');
    lam = -10.0;
end

% create BVP object
bvp = BVP(lam);


% loop over spatial resolutions for tests
N = [100, 1000, 10000];
for n = N

    % output problem information
    fprintf('\nImplicit Midpoint FD method for BVP with lambda = %.1f,  N = %i\n', lam, n);

    % compute/store analytical solution
    t = zeros(n+1, 1);
    t(1) = 0.0;
    t(n+1) = 1.0;
    for j = 1:n-1
        t(j+1) = 0.5*(1-cos((2*j-1)*pi/(2*(n-1))));
    end
    utrue = bvp.utrue(t);

    % create matrix and right-hand side vectors
    Arows = zeros(8*(n+1), 1);
    Acols = zeros(8*(n+1), 1);
    Avals = zeros(8*(n+1), 1);
    b = zeros(2*(n+1), 1);

    % set up linear system:
    %   note: y = [y_{0,1} y_{0,2} y_{1,1} y_{1,2} ... y_{N,1} y_{N,2}]
    idx = 1;
    Arows(idx) = 1;   % A(1,1) = 1.0
    Acols(idx) = 1;
    Avals(idx) = 1;
    idx = idx + 1;
    b(1) = bvp.ua;

    Arows(idx) = 2;   % A(2,2*n+1) = 1.0
    Acols(idx) = 2*n+1;
    Avals(idx) = 1;
    idx = idx + 1;
    b(2) = bvp.ub;

    for j = 1:n

        % setup interval-specific information
        h = t(j+1)-t(j);
        thalf = 0.5*(t(j+1)+t(j));
        alpha = -h*bvp.q(thalf);
        beta = h*bvp.p(thalf);
        gamma = 2*h*bvp.r(thalf);

        % setup eqn in row 2*j+1:
        %    2*y_{j,1} - 2*y_{j-1,1} - h*y_{j,2} - h*y_{j-1,2} = 0
        Arows(idx) = 2*j+1;      % A(2*j+1,2*j+1)
        Acols(idx) = 2*j+1;
        Avals(idx) = 2.0;
        idx = idx + 1;

        Arows(idx) = 2*j+1;      % A(2*j+1,2*j-1)
        Acols(idx) = 2*j-1;
        Avals(idx) = -2.0;
        idx = idx + 1;

        Arows(idx) = 2*j+1;      % A(2*j+1,2*j+2)
        Acols(idx) = 2*j+2;
        Avals(idx) = -h;
        idx = idx + 1;

        Arows(idx) = 2*j+1;      % A(2*j+1,2*j)
        Acols(idx) = 2*j;
        Avals(idx) = -h;
        idx = idx + 1;

        b(2*j+1) = 0.0;

        % setup eqn in row 2*j+2:
        %     alpha*y_{j,1} + alpha*y_{j-1,1} + (2-beta)*y_{j,2} - (2+beta)*y_{j-1,2} = gamma_j
        Arows(idx) = 2*j+2;    % A(2*j+2,2*j+1)
        Acols(idx) = 2*j+1;
        Avals(idx) = alpha;
        idx = idx + 1;

        Arows(idx) = 2*j+2;    % A(2*j+2,2*j-1)
        Acols(idx) = 2*j-1;
        Avals(idx) = alpha;
        idx = idx + 1;

        Arows(idx) = 2*j+2;     % A(2*j+2,2*j+2)
        Acols(idx) = 2*j+2;
        Avals(idx) = 2.0-beta;
        idx = idx + 1;

        Arows(idx) = 2*j+2;    % A(2*j+2,2*j)
        Acols(idx) = 2*j;
        Avals(idx) = -(2.0+beta);
        idx = idx + 1;

        b(2*j+2) = gamma;
    end

    A = sparse(Arows(1:idx-1), Acols(1:idx-1), Avals(1:idx-1), 2*(n+1), 2*(n+1));

    % solve linear system for BVP solution
    y = A \ b;

    % output maximum error
    u = y(1:2:end);
    uerr = abs(u-utrue);
    fprintf('  Maximum BVP solution error = %.4e\n', max(uerr));
end
