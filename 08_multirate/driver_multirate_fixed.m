% Script that runs various Lie-Trotter subcycling methods and MRI methods
% on the nonlinear Kvaerno Prothero and Robinson problem:
%    [u]' = [ G  e ] [(-1+u^2-r)/(2u)] + [ r'(t)/(2u) ]
%    [v]    [ e -1 ] [(-2+v^2-s)/(2v)]   [ s'(t)/(2v) ]
%         = [fs(t,y)]
%           [ff(t,y)]
% where r(t) = 0.5*cos(t),  s(t) = cos(w*t),  0 < t < 5.
% This problem has analytical solution given by
%    u(t) = sqrt(1+r(t)),  v(t) = sqrt(2+s(t)).
%
% We use the parameters:
%   e = inter-variable coupling strength (0.5)
%   G = stiffness at slow time scale (-10)
%   w = variable time-scale separation factor (100)
%
% This script uses Lie-Trotter and Strang-Marchuk subcycling methods, built from
% the fractional-step stepper in 07_implicit_explicit/FractionalStep.m, with
% components at various orders of accuracy to see if/how that affects
% accuracy.  It also runs MRI of various orders of accuracy to see how those
% compare.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../utilities');
addpath('../04_explicit_one_step');
addpath('../07_implicit_explicit');

% KPR problem parameters
Tf = 5;
Nt = 25;
tvals = linspace(0, Tf, Nt+1).';
e = 0.5;
w = 100;
G = -10;

% slow step sizes to try
Hvals = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025];

% KPR component functions
r = @(t) 0.5*cos(t);
s = @(t) cos(w*t);
rdot = @(t) -0.5*sin(t);
sdot = @(t) -w*sin(w*t);

% KPR true solution functions
utrue = @(t) sqrt(1 + r(t));
vtrue = @(t) sqrt(2 + s(t));
ytrue = @(t) [utrue(t); vtrue(t)];

% initial condition
Y0 = ytrue(0);

% true solution at each output time
Ytrue = zeros(Nt+1, 2);
for i = 1:(Nt+1)
    Ytrue(i,:) = ytrue(tvals(i)).';
end

% KPR right-hand side functions (written in terms of u = y(1) and v = y(2))
fs_uv = @(t, u, v) [G, e; 0, 0] ...
    * [(-1 + u^2 - r(t)) / (2*u); (-2 + v^2 - s(t)) / (2*v)] ...
    + [rdot(t)/(2*u); 0];
ff_uv = @(t, u, v) [0, 0; e, -1] ...
    * [(-1 + u^2 - r(t)) / (2*u); (-2 + v^2 - s(t)) / (2*v)] ...
    + [0; sdot(t)/(2*v)];
fs = @(t, y) fs_uv(t, y(1), y(2));
ff = @(t, y) ff_uv(t, y(1), y(2));

% Lie-Trotter subcycling evolves slow dynamics once per macro step and
% fast dynamics with h = H/w substeps.
runFamily('Lie-Trotter-Subcycling-1', Hvals, w, Y0, Ytrue, tvals, ...
    @(h,H) LTSubcycling(fs, ff, ERK.ERK1(), ERK.ERK1(), h, H));

runFamily('Lie-Trotter-Subcycling-2', Hvals, w, Y0, Ytrue, tvals, ...
    @(h,H) LTSubcycling(fs, ff, ERK.ERK2(), ERK.ERK2(), h, H));

% Strang-Marchuk variants symmetrize the slow/fast splitting.
runFamily('Strang-Marchuk-Subcycling-1', Hvals, w, Y0, Ytrue, tvals, ...
    @(h,H) SMSubcycling(fs, ff, ERK.ERK1(), ERK.ERK1(), h, H));

runFamily('Strang-Marchuk-2', Hvals, w, Y0, Ytrue, tvals, ...
    @(h,H) SMSubcycling(fs, ff, ERK.ERK2(), ERK.ERK2(), h, H));

runFamily('Strang-Marchuk-3', Hvals, w, Y0, Ytrue, tvals, ...
    @(h,H) SMSubcycling(fs, ff, ERK.ERK3(), ERK.ERK3(), h, H));

% MRI-GARK methods couple slow stages to a fast IVP solve over each stage interval.
runFamily('MRI-GARK-ERK22a', Hvals, w, Y0, Ytrue, tvals, ...
    @(h,H) MRIMethod(Y0, fs, ff, MRI.MRIGARKERK22a(), ERK.ERK2(), h, H));

runFamily('MRI-GARK-ERK33a', Hvals, w, Y0, Ytrue, tvals, ...
    @(h,H) MRIMethod(Y0, fs, ff, MRI.MRIGARKERK33a(), ERK.ERK3(), h, H));

runFamily('MRI-GARK-ERK45a', Hvals, w, Y0, Ytrue, tvals, ...
    @(h,H) MRIMethod(Y0, fs, ff, MRI.MRIGARKERK45a(), ERK.ERK4(), h, H));

% utility routines to construct each method from the slow and fast right-hand
% side functions and Butcher tables, with fast step size h and slow step size
% H; each returns the stepper, the slow solver, and the fast solver (so that
% their steps and right-hand side evaluations may be counted separately)
function [stepper, slow, fast] = LTSubcycling(fs, ff, Bs, Bf, h, H)
    % Lie-Trotter subcycling: one slow step over [t_n, t_n+H], followed by a
    % fast solve over the same interval
    slow = ERK(fs, Bs, H);
    fast = ERK(ff, Bf, h);
    stepper = FractionalStep(FractionalStep.LieTrotter(), {slow, fast}, H);
end
function [stepper, slow, fast] = SMSubcycling(fs, ff, Bs, Bf, h, H)
    % Strang-Marchuk subcycling: fast solves over each half of [t_n, t_n+H],
    % surrounding one slow step over the full interval (so the fast piece is
    % partition 1, which takes the two half steps)
    slow = ERK(fs, Bs, H);
    fast = ERK(ff, Bf, h);
    stepper = FractionalStep(FractionalStep.StrangMarchuk(), {fast, slow}, H);
end
function [stepper, slow, fast] = MRIMethod(Y0, fs, ff, C, Bf, h, H)
    fast = ERK(ff, Bf, h);
    stepper = MRI(Y0, fs, ff, C, fast, H);
    slow = stepper;
end

function runFamily(name, Hvals, w, Y0, Ytrue, tvals, buildStepper)
    % store errors for convergence-rate estimates
    errs = zeros(size(Hvals));
    fprintf('\n%s:\n', name);
    for idx = 1:numel(Hvals)
        H = Hvals(idx);
        h = H / w;
        [stepper, slow, fast] = buildStepper(h, H);
        fprintf('  H = %f, h = %f:\n', H, h);
        [Y, success] = stepper.Evolve(tvals, Y0);
        errs(idx) = norm(abs(Y - Ytrue), inf);
        if ~success
            fprintf('  solve failed\n');
        end
        fprintf('   steps (s,f) = (%d, %d)  nrhs (s,f) = (%d, %d)  err = %.1e\n', ...
            slow.get_num_steps(), fast.get_num_steps(), slow.get_num_rhs(), fast.get_num_rhs(), errs(idx));
    end

    if numel(Hvals) > 2
        orders = log(errs(1:end-1)./errs(2:end))./log(Hvals(1:end-1)./Hvals(2:end));
        fprintf('estimated order:  %.16g\n', mean(orders));
    end
end
