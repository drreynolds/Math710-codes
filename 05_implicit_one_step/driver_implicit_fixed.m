% Main routine to test various DIRK and IRK methods on the
% scalar-valued ODE problem
%    y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
%    y(0) = 1.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../utilities');
addpath('../shared');
addpath('../03_simple_implicit');

t0 = 0.0;
tf = 5.0;

% problem-defining functions
ytrue = @(t) sin(t) + cos(t);
f = @(t, y, lam) lam*y(1) + (1.0-lam)*cos(t) - (1.0+lam)*sin(t);
J = @(t, y, lam) lam;

solver = ImplicitSolver(J, 20, 1e-9, 1e-12, 2);

Nout = 6;   % includes initial condition
% set output times for the experiment
tspan = linspace(t0, tf, Nout).';
% compute and store the analytical solution
Ytrue = zeros(Nout, 1);
for i = 1:Nout
    Ytrue(i,:) = ytrue(tspan(i));
end
y0 = ytrue(t0);
% set problem parameters for the experiments
lambdas = [-1.0, -10.0, -50.0];
% set requested time step sizes for convergence tests
hvals = 1.0 ./ linspace(1, 7, 7);

% Compare simple one-step implicit methods, DIRK methods, and fully implicit RK methods.
BE = BackwardEuler(f, solver);
runTest(BE, 'Backward Euler', lambdas, hvals, Ytrue, tspan);

Alex3 = DIRK(f, solver, DIRK.Alexander3());
runTest(Alex3, 'Alexander-3', lambdas, hvals, Ytrue, tspan);

CR3 = DIRK(f, solver, DIRK.CrouzeixRaviart3());
runTest(CR3, 'Crouzeix & Raviart-3', lambdas, hvals, Ytrue, tspan);

SD5 = DIRK(f, solver, DIRK.SDIRK5());
runTest(SD5, 'SDIRK5', lambdas, hvals, Ytrue, tspan);

RIIA2 = IRK(f, solver, IRK.RadauIIA2());
runTest(RIIA2, 'RadauIIA-2', lambdas, hvals, Ytrue, tspan);

GL2 = IRK(f, solver, IRK.GaussLegendre2());
runTest(GL2, 'Gauss-Legendre-2', lambdas, hvals, Ytrue, tspan);

RIIA3 = IRK(f, solver, IRK.RadauIIA3());
runTest(RIIA3, 'RadauIIA-3', lambdas, hvals, Ytrue, tspan);

GL3 = IRK(f, solver, IRK.GaussLegendre3());
runTest(GL3, 'Gauss-Legendre-3', lambdas, hvals, Ytrue, tspan);

GL6 = IRK(f, solver, IRK.GaussLegendre6());
runTest(GL6, 'Gauss-Legendre-6', lambdas, hvals, Ytrue, tspan);

function runTest(stepper, name, lambdas, hvals, Ytrue, tspan)
    % store errors for convergence-rate estimates
    errs = zeros(size(hvals));

    fprintf('\n%s tests:\n', name);
    for lam = lambdas
        fprintf('  lambda = %.1f:\n', lam);
        for idx = 1:numel(hvals)
            h = hvals(idx);
            fprintf('    h = %.3f:', h);
            stepper.reset();
            stepper.sol.reset();
            % Pass lambda through the args cell array to both f and J.
            [Y, success] = stepper.Evolve(tspan, Ytrue(1,:).', h, {lam});
            Yerr = abs(Y - Ytrue);
            errs(idx) = norm(Yerr, inf);
            if success
                fprintf('  solves = %4d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                    stepper.get_num_solves(), stepper.sol.get_total_iters(), ...
                    stepper.sol.get_total_setups(), errs(idx));
            else
                fprintf('  solve failed\n');
            end
        end
        orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
        fprintf('    estimated order:  max = %.2f,  avg = %.2f\n', max(orders), mean(orders));
    end
end
