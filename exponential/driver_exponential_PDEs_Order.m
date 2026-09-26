% driver_exponential_PDEs_Order.m
%
% We consider the following PDE:
%     U_t = U_xx + 1/(1+U^2) + phi(t,x) over [0,1]x[0,1] subject to
%     homogeneous Dirichlet boundary conditions.
% phi(t,x) is chosen so that the exact solution is U(t,x) = x(1-x)*exp(t).
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
clear

% Space interval.
a = 0;
b = 1;
t0 = 0;
t_end = 1;

% problem-defining functions
u_true = @(x, t) (x - x.^2) * exp(t);

m = 100;
NTS = [2, 4, 8, 16, 32] * 5;
delta_x = (b - a) / m;
x = linspace(a, b, m + 1).';
X = x(2:end-1);

% Initial vector (length m-1).
U0 = X - X.^2;

% Define the second derivative operator.
e = ones(m - 1, 1);
A_matrix = spdiags([e/delta_x^2, -2*e/delta_x^2, e/delta_x^2], [-1, 0, 1], m-1, m-1);

% Rewrite the problem in autonomous form.
A = blkdiag(sparse(0), A_matrix);
U0 = [t0; U0];  % prepend t0 to U0.

% Convergence plot data.
Err_ExpRK2 = zeros(numel(NTS), 1);
Err_ExpRK3 = zeros(numel(NTS), 1);
Err_ExpRK4s5 = zeros(numel(NTS), 1);
Err_ExpRB2 = zeros(numel(NTS), 1);
Err_ExpRB3 = zeros(numel(NTS), 1);
Err_ExpRB4s2 = zeros(numel(NTS), 1);
CPU_time_ExpRK2 = zeros(numel(NTS), 1);
CPU_time_ExpRK3 = zeros(numel(NTS), 1);
CPU_time_ExpRK4s5 = zeros(numel(NTS), 1);
CPU_time_ExpRB2 = zeros(numel(NTS), 1);
CPU_time_ExpRB3 = zeros(numel(NTS), 1);
CPU_time_ExpRB4s2 = zeros(numel(NTS), 1);

gfun = @(U) g(U, X);
Ffun = @(U) A * U(:) + gfun(U);
Jfun = @(U, Xin) J(U, Xin);

for i = 1:numel(NTS)
    timer = tic;
    [~, sol_ExpRK2] = expRK2s2a(Ffun, A, gfun, t0, t_end, U0, NTS(i));
    CPU_time_ExpRK2(i) = toc(timer);

    timer = tic;
    [~, sol_ExpRK3] = expRK3s3a(Ffun, A, gfun, t0, t_end, U0, NTS(i));
    CPU_time_ExpRK3(i) = toc(timer);

    timer = tic;
    [~, sol_ExpRK4s5] = expRK4s5(Ffun, A, gfun, t0, t_end, U0, NTS(i));
    CPU_time_ExpRK4s5(i) = toc(timer);

    timer = tic;
    [~, sol_ExpRB2] = expRB2(Ffun, A, Jfun, gfun, t0, t_end, U0, NTS(i), X);
    CPU_time_ExpRB2(i) = toc(timer);

    timer = tic;
    [~, sol_ExpRB3] = expRB3s3(Ffun, A, Jfun, gfun, t0, t_end, U0, NTS(i), X, 0.5);
    CPU_time_ExpRB3(i) = toc(timer);

    timer = tic;
    [~, sol_ExpRB4s2] = expRB3s3(Ffun, A, Jfun, gfun, t0, t_end, U0, NTS(i), X, 0.75);
    CPU_time_ExpRB4s2(i) = toc(timer);

    Err_ExpRK2(i) = norm(sol_ExpRK2(2:end) - u_true(X, 1));
    Err_ExpRK3(i) = norm(sol_ExpRK3(2:end) - u_true(X, 1));
    Err_ExpRK4s5(i) = norm(sol_ExpRK4s5(2:end) - u_true(X, 1));
    Err_ExpRB2(i) = norm(sol_ExpRB2(2:end) - u_true(X, 1));
    Err_ExpRB3(i) = norm(sol_ExpRB3(2:end) - u_true(X, 1));
    Err_ExpRB4s2(i) = norm(sol_ExpRB4s2(2:end) - u_true(X, 1));
end

fprintf('Exponential order errors, finest run:\n');
fprintf('  ExpRK2   %.6e   CPU %.3fs\n', Err_ExpRK2(end), CPU_time_ExpRK2(end));
fprintf('  ExpRK3   %.6e   CPU %.3fs\n', Err_ExpRK3(end), CPU_time_ExpRK3(end));
fprintf('  ExpRK4s5 %.6e   CPU %.3fs\n', Err_ExpRK4s5(end), CPU_time_ExpRK4s5(end));
fprintf('  ExpRB2   %.6e   CPU %.3fs\n', Err_ExpRB2(end), CPU_time_ExpRB2(end));
fprintf('  ExpRB3   %.6e   CPU %.3fs\n', Err_ExpRB3(end), CPU_time_ExpRB3(end));
fprintf('  ExpRB4   %.6e   CPU %.3fs\n', Err_ExpRB4s2(end), CPU_time_ExpRB4s2(end));

% Order convergence plot.
figure;
plot(log10(1./NTS), log10(Err_ExpRK2), 'gp-'); hold on;
plot(log10(1./NTS), log10(Err_ExpRK3), 'bx-');
plot(log10(1./NTS), log10(Err_ExpRK4s5), 'yx-');
plot(log10(1./NTS), log10(Err_ExpRB2), 'rp-');
plot(log10(1./NTS), log10(Err_ExpRB3), 'bp-');
plot(log10(1./NTS), log10(Err_ExpRB4s2), 'yp-');
plot(log10(1./NTS), 2*log10(1./NTS*5), 'rx--');
plot(log10(1./NTS), 3*log10(1./NTS*2), 'rx--');
plot(log10(1./NTS), 4*log10(1./NTS), 'rx--');
legend('ExpRK2', 'ExpRK3', 'ExpRK4s5', 'ExpRB2', 'ExpRB3', 'ExpRB4', 'Slope 2', 'Slope 3', 'Slope 4', 'Location', 'best');
xlabel('Log(dt)');
ylabel('Log(Error)');
title('Convergence order');
saveas(gcf, 'order_conv_plot.png');

function g_val = g(U, X)
% Define the nonlinear part.

    % U(1) = t, U(2:end) = spatial part.
    U = U(:);
    t = U(1);
    u_x = U(2:end);
    part1 = (2 + X - X.^2) * exp(t);
    part2 = 1 ./ (1 + (X .* (1 - X) * exp(t)).^2);
    part3 = 1 ./ (1 + u_x.^2);
    g_val = [1; part1 - part2 + part3];
end

function Jac = J(U, X)
% Define the Jacobian.

    U = U(:);
    n = numel(U);
    t = U(1);
    u_x = U(2:end);

    % Diagonal block (m-1, m-1).
    K = spdiags(-2*u_x ./ (1 + u_x.^2).^2, 0, n-1, n-1);

    % Block diagonal: top left 0, lower block K, result is (m,m).
    Jac = blkdiag(sparse(0), K);

    % Assign to (2:end,1).
    extra = (2 + X - X.^2) * exp(t) + ...
        2 * (X .* (1 - X) * exp(t)).^2 ./ (1 + (X .* (1 - X) * exp(t)).^2).^2;
    Jac(2:end,1) = extra;
end
