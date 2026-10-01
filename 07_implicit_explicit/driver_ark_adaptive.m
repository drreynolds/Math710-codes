% Main routine to compare adaptive ARK, DIRK, and ERK methods on the
% reaction-diffusion and viscous Burgers problems.  The ARK solver treats
% diffusion implicitly and all remaining terms explicitly.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../utilities');
addpath('../shared');
addpath('../04_explicit_one_step');
addpath('../05_implicit_one_step');

% shared testing data
rtols = 10.0.^(-(3:2:7));
atol = 1.e-12;
Nout = 20;

rd = LoadProblem('ReactionDiffusion');
ad = LoadProblem('AdvectionDiffusion');
RunProblem(rd, 'Reaction-diffusion', inf, rtols, atol, Nout);
RunProblem(ad, 'Burgers', AdvectionDiffusion.explicit_stability_limit(), rtols, atol, Nout);

% test runner function
function RunProblem(problem, problem_name, hE, rtols, atol, Nout)

    fprintf('\n%s tests:\n', problem_name);
    tspan = linspace(problem.t0, problem.tf, Nout+1).';
    ytrue = zeros(Nout+1, numel(problem.xgrid));
    for iout = 1:(Nout+1)
        ytrue(iout,:) = problem.utrue(problem.xgrid, tspan(iout)).';
    end
    y0 = ytrue(1,:).';

    % use the same component tables for all three solvers
    [BE, BI] = AdaptARK.ARK324L2SA();
    ark_iters = zeros(size(rtols));
    ark_times = zeros(size(rtols));
    ark_errs = zeros(size(rtols));
    dirk_iters = zeros(size(rtols));
    dirk_times = zeros(size(rtols));
    dirk_errs = zeros(size(rtols));
    erk_times = zeros(size(rtols));
    erk_errs = zeros(size(rtols));
    ark_history = [];

    for idx = 1:numel(rtols)
        rtol = rtols(idx);
        fprintf('  rtol = %.1e:\n', rtol);

        % adaptive ARK method on the split problem
        solverI = ImplicitSolver(problem.JI, 12, 1e-10, 1e-12, 3);
        A = AdaptARK(problem.fE, problem.fI, y0, solverI, BE, BI, ...
                     rtol, atol, [], [], [], [], [], hE, true);
        tstart = tic;
        [Y, success] = A.Evolve(tspan, y0);
        ark_times(idx) = toc(tstart);
        ark_iters(idx) = solverI.get_total_iters();
        ark_errs(idx) = max(max(abs(Y-ytrue)));
        ark_history = A.get_step_history();
        if success
            fprintf('    ARK:   steps = %6d  fails = %3d  solves = %6d  Niters = %7d  NJevals = %6d  abserr = %8.2e  runtime = %.2fs\n', ...
                A.get_num_steps(), A.get_num_error_failures(), A.get_num_solves(), ...
                solverI.get_total_iters(), solverI.get_total_setups(), ark_errs(idx), ark_times(idx));
        else
            fprintf('    ARK:   solve failed  abserr = %8.2e\n', ark_errs(idx));
        end

        % adaptive DIRK method on the full right-hand side
        solver = ImplicitSolver(problem.J, 12, 1e-10, 1e-12, 3);
        D = AdaptDIRK(problem.f, y0, solver, BI, rtol, atol);
        tstart = tic;
        [Y, success] = D.Evolve(tspan, y0);
        dirk_times(idx) = toc(tstart);
        dirk_iters(idx) = solver.get_total_iters();
        dirk_errs(idx) = max(max(abs(Y-ytrue)));
        if success
            fprintf('    DIRK:  steps = %6d  fails = %3d  solves = %6d  Niters = %7d  NJevals = %6d  abserr = %8.2e  runtime = %.2fs\n', ...
                D.get_num_steps(), D.get_num_error_failures(), D.get_num_solves(), ...
                solver.get_total_iters(), solver.get_total_setups(), dirk_errs(idx), dirk_times(idx));
        else
            fprintf('    DIRK:  solve failed  abserr = %8.2e\n', dirk_errs(idx));
        end

        % adaptive ERK method on the full right-hand side
        E = AdaptERK(problem.f, y0, BE, rtol, atol);
        tstart = tic;
        [Y, success] = E.Evolve(tspan, y0);
        erk_times(idx) = toc(tstart);
        erk_errs(idx) = max(max(abs(Y-ytrue)));
        if success
            fprintf('    ERK:   steps = %6d  fails = %3d  nrhs = %7d  abserr = %8.2e  runtime = %.2fs\n', ...
                E.get_num_steps(), E.get_num_error_failures(), E.get_num_rhs(), erk_errs(idx), erk_times(idx));
        else
            fprintf('    ERK:   solve failed  abserr = %8.2e\n', erk_errs(idx));
        end
    end

    % accuracy-versus-Newton-iterations plot
    figure;
    loglog(ark_iters, ark_errs, 'bo-', 'DisplayName', 'ARK3(2)4L[2]SA');
    hold on;
    loglog(dirk_iters, dirk_errs, 'rs-', 'DisplayName', 'DIRK component on full RHS');
    xlabel('Newton iterations');
    ylabel('error');
    title([problem_name, ' -- adaptive accuracy versus Newton iterations']);
    grid on;
    legend;
    hold off;
    exportgraphics(gcf, ['ark_adaptive_', strrep(lower(problem_name), '-', '_'), '_iters.png']);

    % accuracy-versus-runtime plot
    figure;
    loglog(ark_times, ark_errs, 'bo-', 'DisplayName', 'ARK3(2)4L[2]SA');
    hold on;
    loglog(dirk_times, dirk_errs, 'rs-', 'DisplayName', 'DIRK component on full RHS');
    loglog(erk_times, erk_errs, 'g^-', 'DisplayName', 'ERK component on full RHS');
    xlabel('runtime (s)');
    ylabel('error');
    title([problem_name, ' -- adaptive accuracy versus runtime']);
    grid on;
    legend;
    hold off;
    exportgraphics(gcf, ['ark_adaptive_', strrep(lower(problem_name), '-', '_'), '_runtime.png']);

    % ARK step history at the tightest tolerance
    figure;
    plot(ark_history.t, ark_history.h, 'b-', 'HandleVisibility', 'off');
    hold on;
    for i = 1:numel(ark_history.t)
        if ark_history.err(i) > 1.0
            plot(ark_history.t(i), ark_history.h(i), 'bx', 'HandleVisibility', 'off');
        end
    end
    if isfinite(hE)
        plot([problem.t0, problem.tf], [hE, hE], 'k--', 'DisplayName', '$h_E$');
        legend('Interpreter', 'latex');
    end
    xlabel('$t$', 'Interpreter', 'latex');
    ylabel('$h$', 'Interpreter', 'latex');
    title([problem_name, ' -- adaptive ARK step history']);
    hold off;
    exportgraphics(gcf, ['ark_adaptive_', strrep(lower(problem_name), '-', '_'), '_steps.png']);
end

% utility function to collect a shared problem's parameters and functions
function problem = LoadProblem(cls)
    problem = feval([cls, '.problem']);
    for fname = {'utrue', 'f', 'J', 'fE', 'fI', 'JI'}
        problem.(fname{1}) = str2func([cls, '.', fname{1}]);
    end
end
