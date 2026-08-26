function driver_adapt_euler_system(N)
% Function to test adaptive forward Euler method on a system of ODEs
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

    % set up problem data
    V = eye(N) + rand(N,N);     % fill V,D with random numbers
    D = diag(-rand(N,1));
    Vinv = inv(V);              % Vinv = V^{-1}
    A = V * D / V;              % construct system matrix
    if N < 10
        fprintf('\nProblem-defining matrices:\n');
        fprintf('V:\n'); disp(V);
        fprintf('Vinv:\n'); disp(Vinv);
        fprintf('D:\n'); disp(D);
        fprintf('A:\n'); disp(A);
    end

    % set problem time interval and initial condition
    t0 = 0.0;
    tf = 1.0;
    y0 = rand(N,1);

    % problem-defining functions
    %   ODE RHS function
    f = @(t,y) A*y;
    %   Analytical solution
    ytrue = @(t) V * ((exp((t-t0) * diag(D)) .* (Vinv * y0)));

    % shared testing data
    Nout = 6;   % includes initial condition
    tspan = linspace(t0, tf, Nout).';

    % get true solution at output times
    Ytrue = zeros(Nout,N);
    for i = 1:Nout
        Ytrue(i,:) = ytrue(tspan(i));
    end

    % set testing tolerances
    rtols = [1e-3, 1e-5, 1e-7];
    atol = 1e-13;

    % create adaptive forward Euler stepper object (will reset rtol before each solve)
    AE = AdaptEuler(f, y0, 1e-3, atol);

    % loop over relative tolerances
    fprintf('\nAdaptive Euler test problem, steps and errors vs tolerances:\n');
    for rtol = rtols

        % set the relative tolerance, and call the solver
        fprintf('  rtol = %.1e\n', rtol);
        AE.set_rtol(rtol);
        AE.reset();
        [Y, success] = AE.Evolve(tspan, y0);
        if ~success
            fprintf('    solve failed at this tolerance\n');
            continue;
        end

        % output solution, errors, and overall error
        Yerr = abs(Y - Ytrue);
        if N < 10
            for i = 1:Nout
                fprintf('    y(%.1f) = ', tspan(i));
                for j = 1:N
                    fprintf('%9.6f ', Y(i,j));
                end
                fprintf('\n');
            end
        end

        fprintf('  overall:  steps = %5d  fails = %2d  abserr = %9.2e  relerr = %9.2e\n\n', ...
            AE.get_num_steps(), AE.get_num_error_failures(), norm(Yerr, inf), ...
            norm(Yerr ./ max(abs(Ytrue), 1e-14), inf));
    end
end
