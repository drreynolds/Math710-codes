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
    fprintf('\nStencil-based FD method for BVP with lambda = %.1f,  N = %i\n', lam, n);

    % compute/store analytical solution
    t = linspace(bvp.a, bvp.b, n+1)';
    h = t(2)-t(1);
    utrue = bvp.utrue(t);

    % create matrix diagonals and right-hand side vector
    %   note: spdiags indexes each diagonal by column, so A(j,j-1) is stored
    %   in lower(j-1), A(j,j) in diagv(j), and A(j,j+1) in upper(j+1)
    lower = zeros(n+1, 1);
    diagv = zeros(n+1, 1);
    upper = zeros(n+1, 1);
    b = zeros(n+1, 1);

    % set up linear system
    diagv(1) = 1;     % A(1,1) = 1.0
    b(1) = bvp.ua;

    diagv(n+1) = 1;   % A(n+1,n+1) = 1.0
    b(n+1) = bvp.ub;

    j = (2:n)';
    lower(j-1) = -1 - 0.5*h*bvp.p(t(j));     % A(j,j-1)
    diagv(j) = 2 + h*h*bvp.q(t(j));          % A(j,j)
    upper(j+1) = -1 + 0.5*h*bvp.p(t(j));     % A(j,j+1)
    b(j) = -h*h*bvp.r(t(j));
    A = spdiags([lower, diagv, upper], [-1, 0, 1], n+1, n+1);

    % solve linear system for BVP solution
    u = A \ b;

    % output maximum error
    uerr = abs(u-utrue);
    fprintf('  Maximum BVP solution error = %.4e\n', max(uerr));
end
