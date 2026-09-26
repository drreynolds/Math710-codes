% driver_exponential_PDEs_CPU.m
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
NTS = [2, 4, 8, 16];
NTS_ExpRK2 = NTS * 100;
NTS_ExpRB4s2 = NTS * 10;
NTS_RK2 = NTS * 5000;
NTS_RK4 = NTS * 1000;

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

% Vector for storing errors.
Err_ExpRK2 = zeros(numel(NTS), 1);
Err_ExpRB4s2 = zeros(numel(NTS), 1);
Err_RK2 = zeros(numel(NTS), 1);
Err_RK4 = zeros(numel(NTS), 1);

% Vector for storing CPU time.
CPU_time_ExpRK2 = zeros(numel(NTS), 1);
CPU_time_ExpRB4s2 = zeros(numel(NTS), 1);
CPU_time_RK2 = zeros(numel(NTS), 1);
CPU_time_RK4 = zeros(numel(NTS), 1);

gfun = @(U) g(U, X);
Ffun = @(U) A * U(:) + gfun(U);
Jfun = @(U, Xin) J(U, Xin);

for i = 1:numel(NTS)
    timer = tic;
    [~, sol_ExpRK2] = expRK2s2a(Ffun, A, gfun, t0, t_end, U0, NTS_ExpRK2(i));
    CPU_time_ExpRK2(i) = toc(timer);

    timer = tic;
    [~, sol_ExpRB4s2] = expRB3s3(Ffun, A, Jfun, gfun, t0, t_end, U0, NTS_ExpRB4s2(i), X, 0.75);
    CPU_time_ExpRB4s2(i) = toc(timer);

    timer = tic;
    [~, sol_RK2] = RK2(Ffun, t0, t_end, U0, NTS_RK2(i));
    CPU_time_RK2(i) = toc(timer);

    timer = tic;
    [~, sol_RK4] = RK4(Ffun, t0, t_end, U0, NTS_RK4(i));
    CPU_time_RK4(i) = toc(timer);

    Err_ExpRK2(i) = norm(sol_ExpRK2(2:end) - u_true(X, 1));
    Err_ExpRB4s2(i) = norm(sol_ExpRB4s2(2:end) - u_true(X, 1));
    Err_RK2(i) = norm(sol_RK2(2:end) - u_true(X, 1));
    Err_RK4(i) = norm(sol_RK4(2:end) - u_true(X, 1));
end

fprintf('CPU time comparison:\n');
fprintf('  ExpRK2:  %s\n', mat2str(CPU_time_ExpRK2.', 4));
fprintf('  RK2:     %s\n', mat2str(CPU_time_RK2.', 4));

% Order convergence plot.
figure;
plot(log10(CPU_time_ExpRK2), log10(Err_ExpRK2), 'gp-'); hold on;
plot(log10(CPU_time_ExpRB4s2), log10(Err_ExpRB4s2), 'yp-');
plot(log10(CPU_time_RK2), log10(Err_RK2), 'rp-');
plot(log10(CPU_time_RK4), log10(Err_RK4), 'yx-');
legend('ExpRK2', 'ExpRB42', 'RK2', 'RK4', 'Location', 'best');
xlabel('Log(CPU time)');
ylabel('Log(Error)');
title('CPU comparison');
saveas(gcf, 'CPU_plot.png');

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
