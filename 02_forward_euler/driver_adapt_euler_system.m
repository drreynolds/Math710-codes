function driver_adapt_euler_system(N)
% Main routine to test adaptive forward Euler method on a system of ODEs
%    y' = f(t,y), t in [0,1],
%    y(0) = y0.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
    % get optional inputs, otherwise use default values
    if nargin < 1 || isempty(N)
        N = 5;
    end
    fprintf('\nRunning system ODE problem with N = %d\n', N);

    % Build a diagonalizable linear test problem with known modal decay rates.
    V = eye(N) + rand(N,N);
    d = -rand(N,1);
    D = diag(d);
    % Construct A = V*D*V^{-1} using right division instead of forming inv(V).
    A = V * D / V;

    if N < 10
        fprintf('\nProblem-defining matrices:\n');
        fprintf('V:\n'); disp(V);
        fprintf('V \\ eye(N):\n'); disp(V \ eye(N));
        fprintf('D:\n'); disp(D);
        fprintf('A:\n'); disp(A);
    end

    % set problem time interval and initial condition
    t0 = 0.0;
    tf = 1.0;
    y0 = rand(N,1);
    z0 = V \ y0;

    % shared testing data
    Nout = 6;
    % set output times for the experiment
    tspan = linspace(t0, tf, Nout).';

    % Compute the analytical solution at the same output times as the method.
    Ytrue = ytrue(tspan, V, d, z0, t0);

    % set testing tolerances
    rtols = [1e-3, 1e-5, 1e-7];
    atol = 1e-13;

    % create adaptive forward Euler stepper object
    AE = AdaptEuler(@(t,y) A*y, y0, 1e-3, atol);

    % loop over relative tolerances
    fprintf('\nAdaptive Euler test problem, steps and errors vs tolerances:\n');
    for k = 1:numel(rtols)
        rtol = rtols(k);

        fprintf('  rtol = %.1e\n', rtol);
        AE.set_rtol(rtol);
        AE.reset();

        [Y, success] = AE.Evolve(tspan, y0);
        if ~success
            fprintf('    solve failed at this tolerance\n');
            continue;
        end

        Yerr = abs(Y - Ytrue);
        if N < 10
            for i = 1:Nout
                fprintf('    y(%.1f) = ', tspan(i));
                disp(Y(i,:));
            end
        end

        abserr = norm(Yerr, inf);
        relerr = norm(Yerr ./ max(abs(Ytrue), 1e-14), inf);
        fprintf('  overall:  steps = %5d  fails = %2d  abserr = %9.2e  relerr = %9.2e\n\n', ...
            AE.get_num_steps(), AE.get_num_error_failures(), abserr, relerr);
    end
end

function Y = ytrue(t, V, d, z0, t0)
    % Scale each modal component by exp(d_i*(t-t0)) and transform back with V.
    Z = exp((t(:)-t0) * d(:).') .* z0(:).';
    Y = Z * V.';
end
