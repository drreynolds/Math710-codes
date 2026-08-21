classdef Taylor2 < handle
    % Taylor2.m
    %
    % Fixed-stepsize second-order Taylor stepper class implementation file.
    %
    % Class to perform fixed-stepsize time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using an explicit second-order Taylor time stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize second-order Taylor method class
    %
    % The three required arguments when constructing a Taylor2 object are
    % functions for the IVP right-hand side and its first partial derivatives:
    %     f = ODE RHS function with calling syntax f(t,y).
    %     f_t = t-derivative of ODE RHS function with calling syntax f_t(t,y).
    %     f_y = y-derivative of ODE RHS function with calling syntax f_y(t,y).
    %     h = (optional) input with stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        f, f_t, f_y
        h = 0.0
        steps = 0
        nrhs = 0
        fn, ft, fy
    end

    methods
        function self = Taylor2(f, f_t, f_y, h)
            if nargin < 3
                error('Taylor2 requires f, f_t, and f_y function handles.');
            end

            % Store the RHS and its first partial derivatives.
            self.f = f;
            self.f_t = f_t;
            self.f_y = f_y;
            % optional inputs
            if nargin >= 4 && ~isempty(h), self.h = h; end
        end

        function [t, y, success] = Taylor2_step(self, t, y, args)
            % Usage: t, y, success = Taylor2_step(t, y, args)
            %
            % Utility routine to take a single second-order Taylor method step,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS and its derivatives.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 4
                args = {};
            end
            if ~iscell(args)
                error('Taylor2: args must be a cell array.');
            end

            % Evaluate the RHS and derivative terms used in the second-order Taylor formula.
            self.fn = self.f(t, y, args{:});
            self.ft = self.f_t(t, y, args{:});
            self.fy = self.f_y(t, y, args{:});
            self.nrhs = self.nrhs + 3;

            % Apply y_{n+1} = y_n + h*f + h^2/2*(f_t + f_y*f).
            y = y(:) + self.h * (self.fn(:) + 0.5*self.h*(self.ft(:) + self.fy*self.fn(:)));
            t = t + self.h;
            self.steps = self.steps + 1;
            success = true;
        end

        function reset(self)
            % Resets the accumulated number of steps

            % reset accumulated statistics
            self.steps = 0;
            self.nrhs = 0;
        end

        function n = get_num_steps(self)
            % Returns the accumulated number of steps
            n = self.steps;
        end

        function n = get_num_rhs(self)
            % Returns the accumulated number of RHS evaluations
            n = self.nrhs;
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h)
            %
            % The fixed-step second-order Taylor method evolution routine
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
                error('Taylor2:Evolve args must be a cell array.');
            end

            % update stored stepsize when requested by the caller
            if h ~= 0.0
                self.h = h;
            end
            % require a nonzero stepsize before evolving
            if self.h == 0.0
                error('Taylor2:Evolve called without specifying a nonzero step size');
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

            % Allocate work arrays now that the solution dimension is known.
            self.fn = y;
            self.ft = y;
            self.fy = zeros(m, m);

            % iterate over output times, filling the solution history
            for iout = 2:nout
                % The divisibility check above allows this rounded internal step count.
                N = round((tspan(iout)-tspan(iout-1))/self.h);
                t = tspan(iout-1);

                % March internally until the next requested output time is reached.
                for n = 1:N %#ok<NASGU>
                    [t, y, success] = self.Taylor2_step(t, y, args);
                    if ~success
                        fprintf('Taylor2::Evolve error in time step at t = %g\n', t);
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
