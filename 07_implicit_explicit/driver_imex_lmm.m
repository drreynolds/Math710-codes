% Main routine to test various ImEx LMM methods on the
% scalar-valued additively split ODE problem
%    y' = fe(t,y) + fi(t,y), t in [0,2],
%    y(0) = 1,
% where
%    fe(t,y) = sin(y-cos(t)) + 0.3*(y-cos(t)) - 0.5*sin(t),
%    fi(t,y) = lambda*(y-cos(t)) - 0.5*sin(t),
% which has true solution y(t) = cos(t).
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear
addpath('../shared');

t0 = 0.0;
tf = 2.0;

% problem-defining functions
ytrue = @(t) cos(t);
fe = @(t, y, lam) sin(y(1) - cos(t)) + 0.3*(y(1) - cos(t)) - 0.5*sin(t);
fi = @(t, y, lam) lam*(y(1) - cos(t)) - 0.5*sin(t);
J = @(t, y, lam) lam;

% The ImEx LMM methods use the shared Newton solver with the Jacobian of fi.
solver = ImplicitSolver(J, 20, 1e-12, 1e-14);

Nout = 5;
% set output times for the experiment
tspan = linspace(t0, tf, Nout).';
% compute and store the analytical solution
Ytrue = zeros(Nout, 1);
for i = 1:Nout
    Ytrue(i,:) = ytrue(tspan(i));
end

% set problem parameters for the experiments
lambdas = [-1.0, -1000.0];
% set requested time step sizes for convergence tests
hvals = 0.01 ./ 2.0.^(0:4);

% SBDF-2
[alpha, beta, gamma] = ImEx_LMM.SBDF2();
RunTest(ImEx_LMM(fe, fi, solver, alpha, beta, gamma), 1, 'SBDF-2', lambdas, hvals, Ytrue, tspan, ytrue);

% CNAB
[alpha, beta, gamma] = ImEx_LMM.CNAB();
RunTest(ImEx_LMM(fe, fi, solver, alpha, beta, gamma), 1, 'CNAB', lambdas, hvals, Ytrue, tspan, ytrue);

% MCNAB
[alpha, beta, gamma] = ImEx_LMM.MCNAB();
RunTest(ImEx_LMM(fe, fi, solver, alpha, beta, gamma), 1, 'MCNAB', lambdas, hvals, Ytrue, tspan, ytrue);

% CNLF
[alpha, beta, gamma] = ImEx_LMM.CNLF();
RunTest(ImEx_LMM(fe, fi, solver, alpha, beta, gamma), 1, 'CNLF', lambdas, hvals, Ytrue, tspan, ytrue);

function RunTest(stepper, prevsteps, name, lambdas, hvals, Ytrue, tspan, ytrue)
    % store errors for convergence-rate estimates
    errs = zeros(size(hvals));
    fprintf('\n%s tests:\n', name);
    for lam = lambdas
        fprintf('  lambda = %.1f:\n', lam);
        for idx = 1:numel(hvals)
            h = hvals(idx);
            fprintf('    h = %.2e:', h);
            stepper.reset();
            stepper.sol.reset();

            % create initial condition vector with the required previous solution values
            y0 = zeros(prevsteps+1, 1);
            y0(end,:) = ytrue(tspan(1));
            for k = 1:prevsteps
                y0(end-k,:) = ytrue(tspan(1)-k*h);
            end

            % Pass lambda through the args cell array to the RHS/Jacobian.
            [Y, success] = stepper.Evolve(tspan, y0, h, {lam});
            errs(idx) = norm(abs(Y - Ytrue), inf);
            if success
                fprintf('  steps = %4d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                    stepper.get_num_steps(), stepper.sol.get_total_iters(), ...
                    stepper.sol.get_total_setups(), errs(idx));
            else
                fprintf('  solve failed  abserr = %8.2e\n', errs(idx));
            end
        end

        if numel(hvals) > 1
            orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
            fprintf('    estimated order:  max = %.2f,  avg = %.2f\n', max(orders), mean(orders));
        end
    end
end
