function driver()
% Main routine to test the backward Euler, trapezoidal, and
% forward Euler methods on the scalar-valued ODE problem
%    y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
%    y(0) = 1.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
addpath('../shared');
addpath('../02_forward_euler');

% Problem time interval.
t0 = 0.0;
tf = 5.0;

% problem-defining functions
ytrue = @(t) sin(t) + cos(t);
f = @(t,y,lam) lam*y(1) + (1.0-lam)*cos(t) - (1.0+lam)*sin(t);
J = @(t,y,lam) lam;

% Use the same Newton/linear-solver object for both implicit methods.
solver = ImplicitSolver(@(t,y,lam) J(t,y,lam), 20, 1e-9, 1e-12, 2);

% shared testing data
Nout = 6;
% set output times for the experiment
tspan = linspace(t0, tf, Nout).';
% compute and store the analytical solution
Ytrue = zeros(Nout, 1);
for i = 1:Nout
    Ytrue(i) = ytrue(tspan(i));
end
y0 = Ytrue(1);
% set requested time step sizes for convergence tests
hvals = [1.0, 0.1, 0.01, 0.001];
% store errors for convergence-rate estimates
errs = zeros(size(hvals));

% create solvers
BE = BackwardEuler(f, solver);
Tr = Trapezoidal(f, solver);
FE = ForwardEuler(f);

for lam = [-1.0, -10.0, -50.0]

    % Pass lambda to each RHS/Jacobian evaluation through the args cell array.
    fprintf('\nbackward Euler tests:\n');
    for idx = 1:numel(hvals)
        h = hvals(idx);
        fprintf('  h = %.3g,  lambda = %.1f:\n', h, lam);
        BE.reset();
        BE.sol.reset();

        [Y, success] = BE.Evolve(tspan, y0, h, {lam});
        Yerr = abs(Y(:,1) - Ytrue);
        errs(idx) = norm(Yerr, inf);
        if success
            fprintf('       t        y(t)       |err(t)|\n');
            for i = 1:Nout
                fprintf('      %.1f  %10.2e  %.2e\n', tspan(i), Y(i,1), Yerr(i));
            end
            fprintf('  overall:  steps = %4d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                BE.get_num_steps(), BE.sol.get_total_iters(), BE.sol.get_total_setups(), errs(idx));
        end
    end
    orders = log(errs(1:end-2)./errs(2:end-1)) ./ log(hvals(1:end-2)./hvals(2:end-1));
    fprintf('estimated order: max = %.4f, avg = %.4f\n', max(orders), mean(orders));

    fprintf('\ntrapezoidal tests:\n');
    for idx = 1:numel(hvals)
        h = hvals(idx);
        fprintf('  h = %.3g,  lambda = %.1f:\n', h, lam);
        Tr.reset();
        Tr.sol.reset();

        [Y, success] = Tr.Evolve(tspan, y0, h, {lam});
        Yerr = abs(Y(:,1) - Ytrue);
        errs(idx) = norm(Yerr, inf);
        if success
            fprintf('       t        y(t)       |err(t)|\n');
            for i = 1:Nout
                fprintf('      %.1f  %10.2e  %.2e\n', tspan(i), Y(i,1), Yerr(i));
            end
            fprintf('  overall:  steps = %4d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                Tr.get_num_steps(), Tr.sol.get_total_iters(), Tr.sol.get_total_setups(), errs(idx));
        end
    end
    orders = log(errs(1:end-2)./errs(2:end-1)) ./ log(hvals(1:end-2)./hvals(2:end-1));
    fprintf('estimated order: max = %.4f, avg = %.4f\n', max(orders), mean(orders));

    fprintf('\nforward Euler tests:\n');
    for idx = 1:numel(hvals)
        h = hvals(idx);
        fprintf('  h = %.3g,  lambda = %.1f:\n', h, lam);
        FE.reset();

        [Y, success] = FE.Evolve(tspan, y0, h, {lam});
        Yerr = abs(Y(:,1) - Ytrue);
        errs(idx) = norm(Yerr, inf);
        if success
            fprintf('       t        y(t)       |err(t)|\n');
            for i = 1:Nout
                fprintf('      %.1f  %10.2e  %.2e\n', tspan(i), Y(i,1), Yerr(i));
            end
            fprintf('  overall:  steps = %4d  abserr = %8.2e\n', FE.get_num_steps(), errs(idx));
        end
    end
    orders = log(errs(1:end-2)./errs(2:end-1)) ./ log(hvals(1:end-2)./hvals(2:end-1));
    fprintf('estimated order: max = %.4f, avg = %.4f\n', max(orders), mean(orders));
end
end
