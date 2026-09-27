% Main routine to test fixed-step ARK methods on the reaction-diffusion
% problem from shared/ReactionDiffusion.m.  Diffusion is treated implicitly,
% while the reaction and forcing terms are treated explicitly.  Results are
% compared against a DIRK method applied to the full right-hand side.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../shared');
addpath('../04_explicit_one_step');
addpath('../05_implicit_one_step');

% shared testing data
rd = ReactionDiffusion.problem();
Nout = 1;
tspan = linspace(rd.t0, rd.tf, Nout+1).';
ytrue = zeros(Nout+1, rd.Nx-2);
for iout = 1:(Nout+1)
    ytrue(iout,:) = ReactionDiffusion.utrue(rd.xgrid, tspan(iout)).';
end
y0 = ytrue(1,:).';
hvals = 0.01 ./ 2.0.^(0:3);

% fixed-step ARK tests
[BE, BI] = ARK.ARS122();
RunARKTest(BE, BI, 'ARS(1,2,2)', hvals, tspan, y0, ytrue);
[BE, BI] = ARK.ARS343();
RunARKTest(BE, BI, 'ARS(3,4,3)', hvals, tspan, y0, ytrue);
[BE, BI] = ARK.ARK324L2SA();
RunARKTest(BE, BI, 'ARK3(2)4L[2]SA', hvals, tspan, y0, ytrue);
[BE, BI] = ARK.ARK436L2SA();
RunARKTest(BE, BI, 'ARK4(3)6L[2]SA', hvals, tspan, y0, ytrue);

% full-RHS DIRK comparison
RunDIRKTest(DIRK.CrouzeixRaviart3(), 'Crouzeix-Raviart-3 on full RHS', hvals, tspan, y0, ytrue);

% test runner function for ARK methods
function RunARKTest(BE, BI, name, hvals, tspan, y0, ytrue)

    errs = zeros(size(hvals));
    fprintf('\n%s tests:\n', name);
    solver = ImplicitSolver(@ReactionDiffusion.JI, 20, 1e-10, 1e-12, 3);
    stepper = ARK(@ReactionDiffusion.fE, @ReactionDiffusion.fI, solver, BE, BI);
    for idx = 1:numel(hvals)
        h = hvals(idx);
        fprintf('  h = %.5e:', h);
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
    fprintf('  estimated order:  max = %.2f,  avg = %.2f\n', max(orders), mean(orders));
end

% test runner function for the full-RHS DIRK method
function RunDIRKTest(B, name, hvals, tspan, y0, ytrue)

    errs = zeros(size(hvals));
    fprintf('\n%s tests:\n', name);
    solver = ImplicitSolver(@ReactionDiffusion.J, 20, 1e-10, 1e-12, 3);
    stepper = DIRK(@ReactionDiffusion.f, solver, B);
    for idx = 1:numel(hvals)
        h = hvals(idx);
        fprintf('  h = %.5e:', h);
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
    fprintf('  estimated order:  max = %.2f,  avg = %.2f\n', max(orders), mean(orders));
end
