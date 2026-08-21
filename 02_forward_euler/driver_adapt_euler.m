function driver_adapt_euler()
% Main routine to test adaptive forward Euler method on the scalar-valued ODE problem
%    y' = -exp(-t)*y, t in [0,5],
%    y(0) = 1.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
t0 = 0.0;
tf = 5.0;

% shared testing data
Nout = 6;   % includes initial condition
% set output times for the experiment
tspan = linspace(t0, tf, Nout);

% get true solution at output times
% compute and store the analytical solution
Ytrue = zeros(Nout,1);
for i = 1:Nout
    Ytrue(i,:) = ytrue(tspan(i));
end
y0 = Ytrue(1,:).';

% set the testing tolerances
rtols = [1e-3, 1e-5, 1e-7];
atol = 1e-11;

% create adaptive forward Euler stepper object (will reset rtol before each solve)
AE = AdaptEuler(@f, y0, 1e-3, atol);

fprintf('Adaptive Euler test problem, steps and errors vs tolerances:\n');
for rtol = rtols
    fprintf('  rtol = %.1e\n', rtol);
    AE.set_rtol(rtol);
    alpha = 1.0;
    % Pass the problem parameter as a cell array; AdaptEuler expands it as
    % f(t,y,args{:}) when evaluating the RHS.
    [Y, success] = AE.Evolve(tspan, y0, 0.0, {alpha});
    if ~success
        fprintf('    solve failed at this tolerance\n');
        continue;
    end

    Yerr = abs(Y - Ytrue);
    for i = 1:Nout
        fprintf('    y(%.1f) = %9.6f,   abserr = %.2e,  relerr = %.2e\n', ...
            tspan(i), Y(i,1), Yerr(i,1), Yerr(i,1)/abs(Y(i,1)));
    end
    fprintf('  overall:  steps = %5d  fails = %2d  abserr = %9.2e  relerr = %9.2e\n\n', ...
        AE.get_num_steps(), AE.get_num_error_failures(), norm(Yerr, inf), norm(Yerr./abs(Ytrue), inf));
end
end


function val = f(t, y, alpha)
    % ODE RHS function (with example parameter alpha)

    val = -alpha*exp(-t)*y(1);
end

function val = ytrue(t)
    % Analytical solution

    val = exp(exp(-t)-1.0);
end
