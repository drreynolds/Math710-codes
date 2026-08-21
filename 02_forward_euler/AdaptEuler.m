classdef AdaptEuler < handle
    % AdaptEuler.m
    %
    % Adaptive forward Euler solver class implementation file.
    %
    % Class to perform adaptive time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using the forward Euler (explicit Euler) time stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Adaptive forward Euler class
    %
    % The two required arguments when constructing an AdaptEuler object
    % are a function for the IVP right-hand side, and a template vector
    % with the same shape and type as the IVP solution vector:
    %   f = ODE RHS function with calling syntax f(t,y).
    %   y = MATLAB array with m entries.
    %
    % Other optional inputs focus on specific adaptivity options:
    %   rtol    = relative solution tolerance (scalar, >= 1e-12)
    %   atol    = absolute solution tolerance (scalar or MATLAB array with m entries, all >=0)
    %   maxit   = maximum allowed number of internal steps
    %   bias    = error bias factor
    %   growth  = maximum stepsize growth factor
    %   safety  = step size safety factor

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        f
        rtol = 1e-3, atol = 1e-14, maxit = 1e6
        bias = 2.0, growth = 50.0, safety = 0.95, hmin = 10*eps
        ONEMSM = 1.0 - sqrt(eps), ONEPSM = 1.0 + sqrt(eps), p = 1
        fails = 0, steps = 0, error_norm = 0.0, h = 0.0
        w, yerr
    end

    methods
        function self = AdaptEuler(f, yTemplate, rtol, atol, maxit, bias, growth, safety, hmin)
            if nargin < 2
                error('AdaptEuler requires f and a template solution vector y.');
            end

            y = yTemplate(:);

            % Store the RHS and overwrite default adaptivity controls when supplied.
            self.f = f;
            if nargin >= 3 && ~isempty(rtol), self.rtol = rtol; end
            if nargin >= 4 && ~isempty(atol), self.atol = atol; end
            if nargin >= 5 && ~isempty(maxit), self.maxit = maxit; end
            if nargin >= 6 && ~isempty(bias), self.bias = bias; end
            if nargin >= 7 && ~isempty(growth), self.growth = growth; end
            if nargin >= 8 && ~isempty(safety), self.safety = safety; end
            if nargin >= 9 && ~isempty(hmin), self.hmin = hmin; end
            if isscalar(self.atol)
                self.atol = ones(numel(y),1)*self.atol;
            else
                self.atol = self.atol(:);
            end

            % Allocate work vectors and initialize run statistics.
            self.w = ones(numel(y),1);
            self.yerr = zeros(numel(y),1);
        end

        function w = error_weight(self, y)
            % Error weight vector utility routine

            y = y(:);
            w = self.bias ./ (self.atol + self.rtol*abs(y));
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The adaptive forward Euler time step evolution routine
            %
            % Inputs:  tspan holds the current time interval, [t0, tf], including any
            %             intermediate times when the solution is desired, i.e.
            %              [t0, t1, ..., tf]
            %          y holds the initial condition, y(t0)
            %          h optionally holds the requested initial step size
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
                error('AdaptEuler:Evolve args must be a cell array, e.g., {alpha}.');
            end

            % Store the requested initial step size; zero means estimate it below.
            self.h = h;

            y0 = y0(:);
            % Initialize output storage, with the first row holding the initial condition.
            tspan = tspan(:);
            m = numel(y0);
            N = numel(tspan)-1;

            y = y0;
            Y = zeros(N+1, m);
            Y(1,:) = y.';

            t = tspan(1);

            % Reject decreasing output times before any steps are attempted.
            for n = 1:N
                if tspan(n+1) < tspan(n)
                    error('AdaptEuler:Evolve illegal tspan');
                end
            end

            % set error weights for the initial solution
            self.w = self.error_weight(y);

            % require a nonzero stepsize before evolving
            % estimate an initial stepsize if one was not supplied
            if self.h == 0.0
                % Estimate h from the weighted RHS norm at the initial condition.
                fn = self.f(t, y, args{:});
                self.error_norm = max(norm(fn(:).*self.w, inf), 1e-8);
                self.h = max(self.hmin, self.safety/self.error_norm);
            end

            % iterate over output times, filling the solution history
            for iout = 2:(N+1)
                % Take as many adaptive internal steps as needed to hit this output time.
                while (tspan(iout)-t) > sqrt(eps*tspan(iout))
                    if (self.steps + self.fails) > self.maxit
                        fprintf('AdaptEuler: reached maximum iterations, returning with failure\n');
                        success = false;
                        return;
                    end

                    % Do not step beyond the next requested output time.
                    self.h = min(self.h, tspan(iout)-t);

                    % Compare one full Euler step against two half Euler steps.
                    y1 = y;
                    y2 = y;

                    fn = self.f(t, y, args{:});
                    y1 = y1 + self.h*fn(:);
                    y2 = y2 + (0.5*self.h)*fn(:);

                    fn = self.f(t+0.5*self.h, y2, args{:});
                    y2 = y2 + (0.5*self.h)*fn(:);

                    % The difference between the two approximations estimates local error.
                    self.yerr = y2 - y1;
                    self.error_norm = max(norm(self.yerr.*self.w, inf), 1e-8);

                    % compute the next stepsize factor
                    eta = self.safety * self.error_norm^(-1.0/(self.p+1));
                    eta = min(eta, self.growth);

                    % successful step: update solution and prepare the next trial
                    if self.error_norm < self.ONEPSM
                        t = t + self.h;
                        y = 2.0*y2 - y1;
                        self.w = self.error_weight(y);
                        self.steps = self.steps + 1;
                        self.h = self.h * eta;
                    % failed step: reduce the stepsize and retry
                    else
                        self.fails = self.fails + 1;
                        if self.h > self.hmin
                            self.h = max(self.h * eta, self.hmin);
                        else
                            fprintf('AdaptEuler: error test failed at h=hmin, returning with failure\n');
                            success = false;
                            return;
                        end
                    end
                end

                % Store the accepted solution at this requested output time.
                Y(iout,:) = y.';
            end

            success = true;
        end

        function set_rtol(self, rtol)
            % Resets the relative tolerance
            self.rtol = rtol;
        end

        function set_atol(self, atol)
            % Resets the scalar- or vector-valued absolute tolerance
            self.atol = ones(size(self.atol))*atol;
        end

        function set_maxit(self, maxit)
            % Resets the maximum allowed iterations
            self.maxit = maxit;
        end

        function set_bias(self, bias)
            % Resets the error bias factor
            self.bias = bias;
        end

        function set_growth(self, growth)
            % Resets the maximum stepsize growth factor
            self.growth = growth;
        end

        function set_safety(self, safety)
            % Resets the stepsize safety factor
            self.safety = safety;
        end

        function set_hmin(self, hmin)
            % Resets the minimum step size
            self.hmin = hmin;
        end

        function update_rhs(self, f)
            self.f = f;
        end

        function out = get_error_weight(self)
            % Returns the current error weight vector
            out = self.w;
        end

        function out = get_error_vector(self)
            % Returns the current error vector
            out = self.yerr;
        end

        function out = get_error_norm(self)
            % Returns the scaled error norm
            out = self.error_norm;
        end

        function out = get_num_error_failures(self)
            % Returns the total number of error test failures
            out = self.fails;
        end

        function out = get_num_steps(self)
            % Returns the total number of internal time steps
            out = self.steps;
        end

        function out = get_current_step(self)
            % Returns the current internal step size
            out = self.h;
        end

        function reset(self)
            % Resets the solver statistics

            self.fails = 0;
            self.steps = 0;
        end
    end
end
