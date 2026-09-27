% Script that runs various adaptive implicit methods on a reaction-diffusion problem.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../shared');
addpath('../04_explicit_one_step');

rd = ReactionDiffusion.problem();
Nout = 20;
% set output times for the experiment
tspan = linspace(rd.t0, rd.tf, Nout+1).';

% Reference data are available from the manufactured exact solution.
yref = zeros(Nout+1, rd.Nx-2);
for iout = 1:(Nout+1)
    yref(iout,:) = ReactionDiffusion.utrue(rd.xgrid, tspan(iout)).';
end

solver = ImplicitSolver(@ReactionDiffusion.J, 20, 1e-9, 1e-12, 3);

rtol = 1e-6;
atol = 1e-12;

ERK_4 = ERK(@ReactionDiffusion.f, ERK.ERK4());
% set requested time step sizes for convergence tests
hvals = [1e-5, 5e-6, 2.5e-6];
% store errors for convergence-rate estimates
errs = zeros(size(hvals));
fprintf('\nExplicit RK4 solver:\n');
for idx = 1:numel(hvals)
    h = hvals(idx);
    ERK_4.reset();
    [Y_ERK, success] = ERK_4.Evolve(tspan, yref(1,:).', h);
    errs(idx) = norm(Y_ERK - yref, 1);
    fprintf('  h = %.1e:', h);
    if ~success
        fprintf('  solve failed');
    end
    fprintf('  steps = %5d  nrhs = %5d, error = %.2e\n', ...
        ERK_4.get_num_steps(), ERK_4.get_num_rhs(), errs(idx));
end
orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
fprintf('estimated order: %.2f\n', median(orders));


SDIRK_CR3 = DIRK(@ReactionDiffusion.f, solver, DIRK.CrouzeixRaviart3());
% set requested time step sizes for convergence tests
hvals = [1e-2, 1e-3, 1e-4];
errs = zeros(size(hvals));
fprintf('\nDiagonally-implicit Crouzeix-Raviart-3 solver:\n');
for idx = 1:numel(hvals)
    h = hvals(idx);
    SDIRK_CR3.reset();
    solver.reset();
    [Y_DIRK, success] = SDIRK_CR3.Evolve(tspan, yref(1,:).', h);
    errs(idx) = norm(Y_DIRK - yref, 1);
    fprintf('  h = %.1e:', h);
    if ~success
        fprintf('  solve failed');
    end
    fprintf('  steps = %5d  nsolves = %5d, error = %.2e\n', ...
        SDIRK_CR3.get_num_steps(), SDIRK_CR3.get_num_solves(), errs(idx));
end
if numel(hvals) > 1
    orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
    fprintf('estimated order: %.2f\n', median(orders));
end


fprintf('\nAdaptive Dormand-Prince solver:\n');
DP = AdaptERK(@ReactionDiffusion.f, yref(1,:).', AdaptERK.DormandPrince(), rtol, atol);
[Y_DP, success] = DP.Evolve(tspan, yref(1,:).');
err_DP = norm(Y_DP - yref, 1);
if ~success, fprintf('  solve failed\n'); end
fprintf('  steps = %5d  fails = %2d, error = %.2e\n', ...
    DP.get_num_steps(), DP.get_num_error_failures(), err_DP);


fprintf('\nAdaptive DIRK43 solver:\n');
AD43 = AdaptDIRK(@ReactionDiffusion.f, yref(1,:).', solver, AdaptDIRK.ESDIRK43(), rtol, atol);
[Y_AD43, success] = AD43.Evolve(tspan, yref(1,:).');
err_AD43 = norm(Y_AD43 - yref, 1);
if ~success, fprintf('  solve failed\n'); end
fprintf('  steps = %5d  fails = %2d, error = %.2e\n', ...
    AD43.get_num_steps(), AD43.get_num_error_failures(), err_AD43);
solver.reset();
