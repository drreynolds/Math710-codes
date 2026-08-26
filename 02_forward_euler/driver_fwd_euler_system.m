function driver_fwd_euler_system(N)
% Function to test the forward Euler method on a system of ODEs
%    y' = f(t,y), t in [0,1],
%    y(0) = y0.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

    % get optional inputs, otherwise set to 5
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

    % time steps to try
    hvals = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125];
    errs = zeros(size(hvals));

    % create forward Euler stepper object (will reset rtol before each solve)
    FE = ForwardEuler(f);

    % loop over time step sizes; call stepper and compute errors
    for idx = 1:numel(hvals)
        h = hvals(idx);

        % set initial condition and call stepper
        fprintf('\nRunning with stepsize h = %.6g:\n', h);
        FE.reset();
        [Y, success] = FE.Evolve(tspan, y0, h);

        % output solution, errors, and overall error
        Yerr = abs(Y-Ytrue);
        errs(idx) = norm(Yerr, inf);
        if N < 10
            for i = 1:Nout
                fprintf('    y(%.1f) = ', tspan(i));
                for j = 1:N
                    fprintf('%9.6f ', Y(i,j));
                end
                fprintf('\n');
            end
        end

        rel = norm(Yerr ./ max(abs(Ytrue), 1e-14), inf);
        fprintf('  overall:  steps = %5d  abserr = %9.2e  relerr = %9.2e\n', ...
            FE.get_num_steps(), errs(idx), rel);
    end
    orders = log(errs(1:end-2)./errs(2:end-1)) ./ log(hvals(1:end-2)./hvals(2:end-1));
    fprintf('estimated order: max = %.4f,  avg = %.4f\n', max(orders), mean(orders));
end
