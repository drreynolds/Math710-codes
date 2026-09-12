function driver(quickMode)
% Main routine to test various DIRK and IRK methods on the
% scalar-valued ODE problem
%    y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
%    y(0) = 1.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'shared'));

% get optional inputs, otherwise use default values
if nargin < 1 || isempty(quickMode)
    quickMode = false;
end

t0 = 0.0;
tf = 5.0;

% problem-defining functions
ytrue = @(t) sin(t) + cos(t);
f = @(t, y, lam) lam*y(1) + (1.0-lam)*cos(t) - (1.0+lam)*sin(t);
J = @(t, y, lam) lam;

% The implicit LMM families use the shared Newton solver and the same Jacobian.
solver = ImplicitSolver(J, 20, 1e-9, 1e-12, 2);

Nout = 6;
% set output times for the experiment
tspan = linspace(t0, tf, Nout).';
% compute and store the analytical solution
Ytrue = zeros(Nout, 1);
for i = 1:Nout
    Ytrue(i,:) = ytrue(tspan(i));
end

% set problem parameters for the experiments
lambdas = [-1.0, -10.0, -50.0, -100.0];
% set requested time step sizes for convergence tests
hvals = [0.1, 0.05, 0.01, 0.005, 0.001];
if quickMode
    lambdas = [-1.0, -50.0];
    hvals = [0.1, 0.01, 0.005];
end

% Adams-Bashforth methods are explicit and only require previous RHS values.
[alpha, beta] = Explicit_LMM.AdamsBashforth1();
RunTest(Explicit_LMM(f, alpha, beta), 0, 'Adams-Bashforth-1', false, lambdas, hvals, Ytrue, tspan);

[alpha, beta] = Explicit_LMM.AdamsBashforth2();
RunTest(Explicit_LMM(f, alpha, beta), 1, 'Adams-Bashforth-2', false, lambdas, hvals, Ytrue, tspan);

[alpha, beta] = Explicit_LMM.AdamsBashforth3();
RunTest(Explicit_LMM(f, alpha, beta), 2, 'Adams-Bashforth-3', false, lambdas, hvals, Ytrue, tspan);

[alpha, beta] = Explicit_LMM.AdamsBashforth4();
RunTest(Explicit_LMM(f, alpha, beta), 3, 'Adams-Bashforth-4', false, lambdas, hvals, Ytrue, tspan);

% Adams-Moulton methods use the implicit solver at the new time level.
[alpha, beta] = Implicit_LMM.AdamsMoulton1();
RunTest(Implicit_LMM(f, solver, alpha, beta), 0, 'Adams-Moulton-1', true, lambdas, hvals, Ytrue, tspan);

[alpha, beta] = Implicit_LMM.AdamsMoulton2();
RunTest(Implicit_LMM(f, solver, alpha, beta), 0, 'Adams-Moulton-2', true, lambdas, hvals, Ytrue, tspan);

[alpha, beta] = Implicit_LMM.AdamsMoulton3();
RunTest(Implicit_LMM(f, solver, alpha, beta), 1, 'Adams-Moulton-3', true, lambdas, hvals, Ytrue, tspan);

[alpha, beta] = Implicit_LMM.AdamsMoulton4();
RunTest(Implicit_LMM(f, solver, alpha, beta), 2, 'Adams-Moulton-4', true, lambdas, hvals, Ytrue, tspan);

% BDF methods are implicit and use the same test harness as Adams-Moulton.
[alpha, beta] = Implicit_LMM.BDF1();
RunTest(Implicit_LMM(f, solver, alpha, beta), 0, 'BDF-1', true, lambdas, hvals, Ytrue, tspan);

[alpha, beta] = Implicit_LMM.BDF2();
RunTest(Implicit_LMM(f, solver, alpha, beta), 1, 'BDF-2', true, lambdas, hvals, Ytrue, tspan);

[alpha, beta] = Implicit_LMM.BDF3();
RunTest(Implicit_LMM(f, solver, alpha, beta), 2, 'BDF-3', true, lambdas, hvals, Ytrue, tspan);

[alpha, beta] = Implicit_LMM.BDF4();
RunTest(Implicit_LMM(f, solver, alpha, beta), 3, 'BDF-4', true, lambdas, hvals, Ytrue, tspan);
end

function RunTest(stepper, prevsteps, name, implicit, lambdas, hvals, Ytrue, tspan)
    % store errors for convergence-rate estimates
    errs = zeros(size(hvals));
    fprintf('\n%s tests:\n', name);
    for lam = lambdas
        fprintf('  lambda = %.1f:\n', lam);
        for idx = 1:numel(hvals)
            h = hvals(idx);
            fprintf('    h = %.3f:', h);
            stepper.reset();
            if implicit
                stepper.sol.reset();
            end

            % Multistep methods need previous solution values; fill them from
            % the exact solution so that startup error does not pollute the test.
            y0 = zeros(prevsteps+1, 1);
            y0(end,:) = ytrue(tspan(1));
            for k = 1:prevsteps
                y0(end-k,:) = ytrue(tspan(1)-k*h);
            end

            % Pass lambda through the args cell array to the RHS/Jacobian.
            [Y, success] = stepper.Evolve(tspan, y0, h, {lam});
            errs(idx) = norm(abs(Y - Ytrue), inf);
            if success
                if implicit
                    fprintf('  steps = %4d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                        stepper.get_num_steps(), stepper.sol.get_total_iters(), ...
                        stepper.sol.get_total_setups(), errs(idx));
                else
                    fprintf('  steps = %4d  Nrhs = %6d  abserr = %8.2e\n', ...
                        stepper.get_num_steps(), stepper.get_num_rhs(), errs(idx));
                end
            else
                fprintf('  solve failed  abserr = %8.2e\n', errs(idx));
            end
        end

        if numel(hvals) > 2
            orders = log(errs(1:end-2)./errs(2:end-1))./log(hvals(1:end-2)./hvals(2:end-1));
            fprintf('    estimated order:  max = %.2f,  avg = %.2f\n', max(orders), mean(orders));
        end
    end
end
