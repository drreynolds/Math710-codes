% Script that runs subcycling and MRI methods with an adaptive fast solver on
% the nonlinear Kvaerno Prothero and Robinson problem:
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
% For each of a few slow step sizes H, this script runs Lie-Trotter and
% Strang-Marchuk subcycling (built from 07_implicit_explicit/FractionalStep.m,
% with one explicit slow step per sub-step) and the MRI-GARK-ERK33a method,
% all using the same adaptive fast solver, over a range of fast relative
% tolerances.  As the fast tolerance is tightened, the errors of the
% subcycling methods level off at their splitting error, no matter how
% accurately the fast piece is solved, while the error of the MRI method
% keeps decreasing until it reaches its own (much smaller) O(H^3) error, at a
% similar cost.
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

% slow method for the subcycling methods, and adaptive fast method
slow_table = ERK.ERK2();
fast_adapt_table = AdaptERK.BogackiShampine();

% slow step sizes and fast relative tolerances to try
Hvals = [0.02, 0.01];
rtols = 10.^(-(3:10));
atol = 1e-12;

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

methods = {'Lie-Trotter subcycling', @LTSubcycling;
           'Strang-Marchuk subcycling', @SMSubcycling;
           'MRI-GARK-ERK33a', @MRIMethod};

for H = Hvals
    fprintf('\nH = %g:\n', H);
    for j = 1:size(methods, 1)
        name = methods{j,1};
        buildMethod = methods{j,2};
        fprintf('  %s:\n', name);
        for rtol = rtols
            [stepper, slow, fast] = buildMethod(fs, ff, Y0, slow_table, fast_adapt_table, H, rtol, atol);
            [Y, success] = stepper.Evolve(tvals, Y0);
            err = norm(abs(Y - Ytrue), inf);
            if ~success
                fprintf('    rtol = %.0e:  solve failed\n', rtol);
                continue
            end
            fprintf('    rtol = %.0e:  nrhs (s,f) = (%5d, %7d)  err = %.2e\n', ...
                rtol, slow.get_num_rhs(), fast.get_num_rhs(), err);
        end
    end
end

% utility routines to construct each method from the slow and fast right-hand
% side functions, initial condition, slow and adaptive fast Butcher tables,
% slow step size H, and fast tolerances rtol and atol; each returns the
% stepper, the slow solver, and the fast solver (so that their right-hand
% side evaluations may be counted separately)
function [stepper, slow, fast] = LTSubcycling(fs, ff, Y0, Bs, Bf, H, rtol, atol)
    % Lie-Trotter subcycling: one slow step over [t_n, t_n+H], followed by a
    % fast solve over the same interval
    slow = ERK(fs, Bs, H);
    fast = AdaptERK(ff, Y0, Bf, rtol, atol);
    stepper = FractionalStep(FractionalStep.LieTrotter(), {slow, fast}, H);
end
function [stepper, slow, fast] = SMSubcycling(fs, ff, Y0, Bs, Bf, H, rtol, atol)
    % Strang-Marchuk subcycling: fast solves over each half of [t_n, t_n+H],
    % surrounding one slow step over the full interval (so the fast piece is
    % partition 1, which takes the two half steps)
    slow = ERK(fs, Bs, H);
    fast = AdaptERK(ff, Y0, Bf, rtol, atol);
    stepper = FractionalStep(FractionalStep.StrangMarchuk(), {fast, slow}, H);
end
function [stepper, slow, fast] = MRIMethod(fs, ff, Y0, Bs, Bf, H, rtol, atol)
    % (Bs is unused, since the MRI method has its own slow coupling table)
    fast = AdaptERK(ff, Y0, Bf, rtol, atol);
    stepper = MRI(Y0, fs, ff, MRI.MRIGARKERK33a(), fast, H);
    slow = stepper;
end
