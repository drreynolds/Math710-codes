% Main routine to apply fractional-step (operator-splitting) methods to the
% viscous Burgers problem from shared/AdvectionDiffusion.m,
%    u' = fE(t,u) + fI(t,u),
% where fE is the (nonstiff) advection term and fI is the (stiff) diffusion
% and forcing term.  Each partition is advanced by its own sub-solver:
%
%   fixed:     advection with the fixed-step ERK method, using a step at the
%              explicit stability limit hE (so several sub-steps per fractional
%              step), and diffusion with one step of the fixed-step DIRK
%              method per fractional step;
%   adaptive:  advection with AdaptERK, and diffusion with AdaptDIRK, both at
%              a tight tolerance, so that the remaining error is essentially
%              the splitting error.
%
% Both use the explicit and implicit component tables of ARK3(2)4L[2]SA, so
% that the two sub-solver types can be compared directly.  We only use
% splittings with nonnegative sub-steps, since a backward-in-time sub-step of
% the diffusion term would be unstable.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../utilities');
addpath('../shared');
addpath('../04_explicit_one_step');
addpath('../05_implicit_one_step');

% fractional-step methods to test, from the catalogue in FractionalStep.m
methods = {'Lie-Trotter', FractionalStep.LieTrotter();
           'Strang-Marchuk', FractionalStep.StrangMarchuk();
           'OS2(2,2)-1/4', FractionalStep.OS2(0.25)};

% testing data
ad = LoadProblem('AdvectionDiffusion');
Hvals = 0.1 ./ 2.0.^(0:4);
rtol = 1e-8;
atol = 1e-12;
hE = ad.explicit_stability_limit();
[BE, BI] = ARK.ARK324L2SA();
tspan = [ad.t0; ad.tf];
y0 = ad.u0();
ytrue = ad.utrue(ad.xgrid, ad.tf);

fprintf('Burgers splitting tests, fixed sub-solvers (hE = %.2e for advection, one DIRK step for diffusion):\n', hE);
for i = 1:size(methods, 1)
    RunTest(methods{i,1}, methods{i,2}, @() FixedSolvers(ad, BE, BI, hE), Hvals, tspan, y0, ytrue);
end

fprintf('\nBurgers splitting tests, adaptive sub-solvers (rtol = %.0e):\n', rtol);
for i = 1:size(methods, 1)
    RunTest(methods{i,1}, methods{i,2}, @() AdaptiveSolvers(ad, BE, BI, y0, rtol, atol), Hvals, tspan, y0, ytrue);
end

% end of script

% utility routines to construct each type of sub-solver pair
function solvers = FixedSolvers(ad, BE, BI, hE)
    solverI = ImplicitSolver(ad.JI, 12, 1e-10, 1e-12, 3);
    solvers = {ERK(ad.fE, BE, hE), DIRK(ad.fI, solverI, BI, ad.tf-ad.t0)};
end
function solvers = AdaptiveSolvers(ad, BE, BI, y0, rtol, atol)
    solverI = ImplicitSolver(ad.JI, 12, 1e-10, 1e-12, 3);
    solvers = {AdaptERK(ad.fE, y0, BE, rtol, atol), ...
               AdaptDIRK(ad.fI, y0, solverI, BI, rtol, atol)};
end

% test runner function
function RunTest(name, S, BuildSolvers, Hvals, tspan, y0, ytrue)
    fprintf('\n  %s (splitting order %i):\n', name, S.p);
    errs = zeros(size(Hvals));
    for idx = 1:numel(Hvals)
        H = Hvals(idx);
        solvers = BuildSolvers();
        stepper = FractionalStep(S, solvers);
        tstart = tic;
        [Y, success] = stepper.Evolve(tspan, y0, H);
        runtime = toc(tstart);
        errs(idx) = max(abs(Y(end,:).' - ytrue));
        if ~success
            fprintf('    H = %.5f:  solve failed\n', H);
            continue;
        end
        fprintf('    H = %.5f:  advection steps = %6i  diffusion steps = %5i  abserr = %8.2e  runtime = %.2fs\n', ...
                H, solvers{1}.get_num_steps(), solvers{2}.get_num_steps(), errs(idx), runtime);
    end
    orders = log(errs(1:end-1)./errs(2:end))./log(Hvals(1:end-1)./Hvals(2:end));
    fprintf('    estimated orders:  %s\n', strjoin(arrayfun(@(q) sprintf('%5.2f', q), orders, 'UniformOutput', false), ' '));
end

% utility function to collect a shared problem's parameters and functions
function problem = LoadProblem(cls)
    problem = feval([cls, '.problem']);
    for fname = {'utrue', 'fE', 'fI', 'JI', 'u0', 'explicit_stability_limit'}
        problem.(fname{1}) = str2func([cls, '.', fname{1}]);
    end
end
