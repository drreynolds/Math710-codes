classdef ForwardEuler < handle
    % ForwardEuler.m
    %
    % Fixed-stepsize forward Euler stepper class implementation file.
    %
    % Class to perform fixed-stepsize time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using the forward Euler (explicit Euler) time stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize forward Euler class
    %
    % The one required argument when constructing a ForwardEuler object
    % is a function for the IVP right-hand side:
    %     f = ODE RHS function with calling syntax f(t,y).
    %     h = (optional) input with requested stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        f
        h = 0.0
        steps = 0
    end

    methods
        function self = ForwardEuler(f, h)
            if nargin < 1
                error('ForwardEuler requires an RHS function handle f(t,y,...)');
            end
            % required inputs
            self.f = f;
            if nargin >= 2 && ~isempty(h)
                % optional inputs
                self.h = h;
            end
        end

        function [t, y, success] = forward_euler_step(self, t, y, h, args)
            % Usage: t, y, success = forward_euler_step(t, y, h, args)
            %
            % Utility routine to take a single forward Euler time step of size h,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 5
                args = {};
            end
            rhs = self.f(t, y, args{:});
            y = y + h * rhs;
            t = t + h;
            self.steps = self.steps + 1;
            success = true;
        end

        function update_rhs(self, f)
            % Updates the RHS function (cannot change vector dimensions)
            self.f = f;
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
            % The fixed-step forward Euler evolution routine
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
                error('ForwardEuler:Evolve args must be a cell array, e.g., {alpha,beta}');
            end

            % update stored stepsize when requested by the caller
            if h ~= 0.0
                self.h = h;
            end
            % require a nonzero stepsize before evolving
            if self.h == 0.0
                error('ForwardEuler:Evolve called without specifying a nonzero step size');
            end

            % Initialize output storage, with the first row holding the initial condition.
            tspan = tspan(:);
            y = y0(:);
            nout = numel(tspan);
            m = numel(y);

            Y = zeros(nout, m);
            Y(1, :) = y.';

            % iterate over output times, filling the solution history
            for iout = 2:nout
                dt = tspan(iout) - tspan(iout - 1);
                if dt < 0
                    error('ForwardEuler:Evolve requires nondecreasing tspan');
                end

                % Choose enough internal steps that no step exceeds the requested size.
                N = ceil(dt / self.h);
                if N < 1
                    N = 1;
                end
                hcur = dt / N;
                t = tspan(iout - 1);

                % March internally until the next requested output time is reached.
                for n = 1:N
                    [t, y, success] = self.forward_euler_step(t, y, hcur, args);
                    if ~success
                        fprintf('forward_euler error in time step at t = %g\n', t);
                        return;
                    end
                end

                % Store the current solution as one row in the output history.
                Y(iout, :) = y.';
            end

            success = true;
        end
    end
end
