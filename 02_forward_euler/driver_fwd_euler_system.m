function driver_fwd_euler_system(N)
% Main routine to test the forward Euler method on a system of ODEs
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

    % problem-defining functions
    f = @(t,y) A*y; %#ok<NASGU>

    % shared testing data
    Nout = 6;
    % set output times for the experiment
    tspan = linspace(t0, tf, Nout).';

    % Compute the analytical solution at the same output times as the method.
    Ytrue = ytrue(tspan, V, d, z0, t0);

    % time steps to try
    % set requested time step sizes for convergence tests
    hvals = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125];
    % store errors for convergence-rate estimates
    errs = zeros(size(hvals));

    % create forward Euler stepper object
    FE = ForwardEuler(@(t,y) A*y);

    % loop over time step sizes
    for idx = 1:numel(hvals)
        h = hvals(idx);
        fprintf('\nRunning with stepsize h = %.6g:\n', h);
        FE.reset();
        [Y, success] = FE.Evolve(tspan, y0, h);
        if ~success
            fprintf('  solve failed\n');
            continue;
        end

        Yerr = abs(Y - Ytrue);
        errs(idx) = norm(Yerr, inf);
        if N < 10
            for i = 1:Nout
                fprintf('    y(%.1f) = ', tspan(i));
                disp(Y(i,:));
            end
        end

        rel = norm(Yerr ./ max(abs(Ytrue), 1e-14), inf);
        fprintf('  overall:  steps = %5d  abserr = %9.2e  relerr = %9.2e\n', ...
            FE.get_num_steps(), errs(idx), rel);
    end

    orders = log(errs(1:end-2)./errs(2:end-1)) ./ log(hvals(1:end-2)./hvals(2:end-1));
    fprintf('estimated order: max = %.4f,  avg = %.4f\n', max(orders), mean(orders));
end

function Y = ytrue(t, V, d, z0, t0)
    % Scale each modal component by exp(d_i*(t-t0)) and transform back with V.
    Z = exp((t(:)-t0) * d(:).') .* z0(:).';
    Y = Z * V.';
end
