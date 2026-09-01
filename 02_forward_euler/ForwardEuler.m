classdef ForwardEuler < handle
    % ForwardEuler.m
    %
    % Fixed-stepsize forward Euler stepper class implementation file.
    %
    % Note that in the "classdef" line above, we specify that this is 
    % a "handle class" rather than Matlab's default "value class".  
    % This means that when we pass a ForwardEuler object to a function, 
    % the function will receive a reference to the original object 
    % rather than a copy of it.  This is important because we want to 
    % be able to update the statistics stored within this object, and 
    % for more complicated clases, we want to avoid repeated memory 
    % allocation/deallocation of class objects.
    %
    % Class to perform fixed-stepsize time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using the forward Euler (explicit Euler) time stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC

    % Stored problem data and run statistics.
    properties
        f
        h = 0.0
        steps = 0
    end

    methods
        function self = ForwardEuler(f, h)
            % The one required argument when constructing a ForwardEuler object
            % is a function for the IVP right-hand side:
            %     f = ODE RHS function with calling syntax f(t,y,<args>).
            %     h = (optional) input with requested stepsize to use for time stepping.
            %         Note that this MUST be set either here or in the Evolve call.

            if nargin < 1
                error('ForwardEuler requires an RHS function handle f(t,y,args)');
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
            y = y + h * self.f(t, y, args{:});
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
            %              the RHS.  This should be a cell array, e.g., {alpha,beta}.
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

            % raise error if step size was never set
            if self.h == 0.0
                error('ForwardEuler:Evolve called without specifying a nonzero step size');
            end

            % Initialize output storage, with the first row holding the initial condition.
            tspan = tspan(:);
            y = y0(:);
            Y = zeros(numel(tspan), numel(y));
            Y(1, :) = y.';

            % iterate over output times, filling the solution history
            for iout = 2:numel(tspan)

                % determine how many internal steps are required, and the actual step size to use
                N = max(1, ceil((tspan(iout)-tspan(iout-1)) / self.h));
                h = (tspan(iout)-tspan(iout-1)) / N;

                % reset "current" t that will be evolved internally
                t = tspan(iout - 1);

                % iterate over internal time steps to reach next output
                for n = 1:N

                    % perform forward Euler update
                    [t, y, success] = self.forward_euler_step(t, y, h, args);
                    if ~success
                        fprintf('forward_euler error in time step at t = %g\n', t);
                        return;
                    end
                end

                % store current results in output matrix
                Y(iout, :) = y.';
            end

            success = true;
        end
    end
end
