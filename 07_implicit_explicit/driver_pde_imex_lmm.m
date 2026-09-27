% Main routine to test fixed-step ImEx LMM methods on the reaction-diffusion
% problem from shared/ReactionDiffusion.m and the viscous Burgers problem from
% shared/AdvectionDiffusion.m.  In both, the diffusion and forcing terms are
% treated implicitly, while the reaction or advection term is treated
% explicitly.  Results are compared against a second-order DIRK method applied
% to the full right-hand side.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../shared');
addpath('../05_implicit_one_step');

% shared testing data
Nout = 1;
hvals = 0.01 ./ 2.0.^(0:3);

rd = LoadProblem('ReactionDiffusion');
ad = LoadProblem('AdvectionDiffusion');
RunProblem(rd, 'Reaction-diffusion', Nout, hvals);
RunProblem(ad, 'Burgers', Nout, hvals);

% test runner function for ImEx LMM methods
function RunLMMTest(problem, alphas, betas, gammas, prevsteps, name, hvals, tspan, ytrue)

    errs = zeros(size(hvals));
    fprintf('\n  %s tests:\n', name);
    solver = ImplicitSolver(problem.JI, 20, 1e-10, 1e-12, 3);
    stepper = ImEx_LMM(problem.fE, problem.fI, solver, alphas, betas, gammas);
    for idx = 1:numel(hvals)
        h = hvals(idx);
        fprintf('    h = %.5e:', h);
        stepper.reset();
        stepper.sol.reset();
        % create initial condition vector with the required previous solution values
        y0 = zeros(prevsteps+1, numel(problem.xgrid));
        y0(end,:) = ytrue(1,:);
        for k = 1:prevsteps
            y0(end-k,:) = problem.utrue(problem.xgrid, tspan(1)-k*h).';
        end
        [Y, success] = stepper.Evolve(tspan, y0, h);
        errs(idx) = max(max(abs(Y-ytrue)));
        if success
            fprintf('  steps = %4d  solves = %5d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                stepper.get_num_steps(), stepper.get_num_solves(), ...
                stepper.sol.get_total_iters(), stepper.sol.get_total_setups(), errs(idx));
        else
            fprintf('  solve failed  abserr = %8.2e\n', errs(idx));
        end
    end
    orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
    fprintf('    estimated order:  max = %.2f,  avg = %.2f\n', max(orders), mean(orders));
end

% test runner function for the full-RHS DIRK method
function RunDIRKTest(problem, B, name, hvals, tspan, ytrue)

    errs = zeros(size(hvals));
    fprintf('\n  %s tests:\n', name);
    solver = ImplicitSolver(problem.J, 20, 1e-10, 1e-12, 3);
    stepper = DIRK(problem.f, solver, B);
    y0 = ytrue(1,:).';
    for idx = 1:numel(hvals)
        h = hvals(idx);
        fprintf('    h = %.5e:', h);
        stepper.reset();
        stepper.sol.reset();
        [Y, success] = stepper.Evolve(tspan, y0, h);
        errs(idx) = max(max(abs(Y-ytrue)));
        if success
            fprintf('  steps = %4d  solves = %5d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                stepper.get_num_steps(), stepper.get_num_solves(), ...
                stepper.sol.get_total_iters(), stepper.sol.get_total_setups(), errs(idx));
        else
            fprintf('  solve failed  abserr = %8.2e\n', errs(idx));
        end
    end
    orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
    fprintf('    estimated order:  max = %.2f,  avg = %.2f\n', max(orders), mean(orders));
end

% problem runner function
function RunProblem(problem, problem_name, Nout, hvals)

    fprintf('\n%s tests:\n', problem_name);
    tspan = linspace(problem.t0, problem.tf, Nout+1).';
    ytrue = zeros(Nout+1, numel(problem.xgrid));
    for iout = 1:(Nout+1)
        ytrue(iout,:) = problem.utrue(problem.xgrid, tspan(iout)).';
    end

    % ImEx LMM tests
    [alphas, betas, gammas] = ImEx_LMM.SBDF2();
    RunLMMTest(problem, alphas, betas, gammas, 1, 'SBDF-2', hvals, tspan, ytrue);
    [alphas, betas, gammas] = ImEx_LMM.CNAB();
    RunLMMTest(problem, alphas, betas, gammas, 1, 'CNAB', hvals, tspan, ytrue);
    [alphas, betas, gammas] = ImEx_LMM.MCNAB();
    RunLMMTest(problem, alphas, betas, gammas, 1, 'MCNAB', hvals, tspan, ytrue);
    [alphas, betas, gammas] = ImEx_LMM.CNLF();
    RunLMMTest(problem, alphas, betas, gammas, 1, 'CNLF', hvals, tspan, ytrue);

    % full-RHS DIRK comparison
    RunDIRKTest(problem, DIRK.SSP222SDIRK(), 'SSP2(2,2,2)-SDIRK on full RHS', hvals, tspan, ytrue);
end

% utility function to collect a shared problem's parameters and functions
function problem = LoadProblem(cls)
    problem = feval([cls, '.problem']);
    for fname = {'utrue', 'f', 'J', 'fE', 'fI', 'JI'}
        problem.(fname{1}) = str2func([cls, '.', fname{1}]);
    end
end
