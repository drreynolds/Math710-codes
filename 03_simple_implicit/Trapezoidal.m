classdef Trapezoidal < handle
    % Trapezoidal.m
    %
    % Fixed-stepsize trapezoidal stepper class implementation file.
    %
    % Class to perform fixed-stepsize time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using the trapezoidal time stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize trapezoidal class
    %
    % The two required arguments when constructing a Trapezoidal object
    % are a function for the IVP right-hand side, and an implicit solver to use:
    %     f = ODE RHS function with calling syntax f(t,y).
    %     sol = algebraic solver object to use [ImplicitSolver]
    %     h = (optional) input with stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        f, sol, fold
        h = 0.0
        steps = 0
    end

    methods
        function self = Trapezoidal(f, sol, h)
            if nargin < 2
                error('Trapezoidal requires f and implicit solver object.');
            end
            % required inputs
            self.f = f;
            self.sol = sol;
            if nargin >= 3 && ~isempty(h)
                % optional inputs
                self.h = h;
            end
        end

        function [t, y, success] = trapezoidal_step(self, t, y, args)
            % Usage: t, y, success = trapezoidal_step(t, y, args)
            %
            % Utility routine to take a single trapezoidal time step,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 4
                args = {};
            end
            if ~iscell(args)
                error('Trapezoidal: args must be a cell array.');
            end

            % Store the old-time RHS contribution before moving to the new time.
            self.fold = self.f(t, y, args{:});
            t = t + self.h;
            yold = y;
            % Define the trapezoidal residual using both old and new RHS values.
            F = @(ynew) ynew(:) - yold(:) - 0.5*self.h*self.fold(:) - 0.5*self.h*self.f(t, ynew(:), args{:});
            % Tell the Newton solver to use I - h/2*J for this implicit step.
            self.sol.setup_linear_solver(t, -0.5*self.h, args);

            % Solve the nonlinear residual equation for the new solution.
            [ynew, ~, success] = self.sol.solve(F, y);
            y = ynew(:);
            self.steps = self.steps + 1;
        end

        function reset(self)
            % Resets the accumulated number of steps

            % reset accumulated statistics
            self.steps = 0;
        end

        function n = get_num_steps(self)
            % Returns the accumulated number of steps
            n = self.steps;
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The fixed-step trapezoidal evolution routine
            %
            % Inputs:  tspan holds the current time interval, [t0, tf], including any
            %              intermediate times when the solution is desired, i.e.
            %              [t0, t1, ..., tf]
            %          y holds the initial condition, y(t0)
            %          h optionally holds the requested step size (if it is not
            %              provided then the stored value will be used)
            %          args holds optional equation parameters used when evaluating
            %              the RHS.
            % Outputs: Y holds the computed solution at all tspan values,
            %              [y(t0), y(t1), ..., y(tf)]
            %          success = true if the solver traversed the interval,
            %              false if an integration step failed
            %
            if nargin < 4 || isempty(h)
                h = 0.0;
            end
            if nargin < 5
                args = {};
            end
            if ~iscell(args)
                error('Trapezoidal:Evolve args must be a cell array.');
            end

            % update stored stepsize when requested by the caller
            if h ~= 0.0
                self.h = h;
            end
            % require a nonzero stepsize before evolving
            if self.h == 0.0
                error('Trapezoidal:Evolve called without specifying a nonzero step size');
            end

            % Verify that each output interval can be reached by an integer number of steps.
            tspan = tspan(:);
            for n = 1:(numel(tspan)-1)
                hn = tspan(n+1)-tspan(n);
                if abs(round(hn/self.h) - (hn/self.h)) > 100*sqrt(eps)*abs(self.h)
                    error('input values in tspan (%e,%e) are not separated by a multiple of h = %e', tspan(n), tspan(n+1), self.h);
                end
            end

            % Initialize output storage, with the first row holding the initial condition.
            y = y0(:);
            nout = numel(tspan);
            m = numel(y);
            Y = zeros(nout, m);
            Y(1,:) = y.';

            % iterate over output times, filling the solution history
            for iout = 2:nout
                % The divisibility check above allows this rounded internal step count.
                N = round((tspan(iout)-tspan(iout-1))/self.h);
                t = tspan(iout-1);

                % March internally until the next requested output time is reached.
                for n = 1:N
                    [t, y, success] = self.trapezoidal_step(t, y, args);
                    if ~success
                        fprintf('Trapezoidal::Evolve error in time step at t = %g\n', t);
                        return;
                    end
                end

                % Store the current solution as one row in the output history.
                Y(iout,:) = y.';
            end

            success = true;
        end
    end
end
