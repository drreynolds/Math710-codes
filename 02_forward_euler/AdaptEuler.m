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
    % The two required arguments when constructing an AdaptEuler object
    % are a function for the IVP right-hand side, and a template vector
    % with the same shape and type as the IVP solution vector:
    %   f = ODE RHS function with calling syntax f(t,y,<args>).
    %   y = column vector with m entries.
    %
    % Other optional inputs focus on specific adaptivity options:
    %   rtol    = relative solution tolerance (scalar, >= 1e-12)
    %   atol    = absolute solution tolerance (scalar or column vector with m entries, all >=0)
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
        function self = AdaptEuler(f, y, rtol, atol, maxit, bias, growth, safety, hmin)
            if nargin < 2
                error('AdaptEuler requires f and a template solution vector y.');
            end

            % required inputs
            self.f = f;
            y = y(:);  % ensure this is a column vector

            % optional inputs
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

            % allocate work vectors
            self.w = ones(numel(y),1);
            self.yerr = zeros(numel(y),1);
        end

        function w = error_weight(self, y)
            % Error weight vector utility routine
            w = self.bias ./ (self.atol + self.rtol*abs(y(:)));
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
            %              the RHS.  This must be a cell array, e.g., {alpha,beta}
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
                error('AdaptEuler:Evolve args must be a cell array, e.g., {alpha,beta}.');
            end

            % store input step size
            self.h = h;

            % store sizes
            m = numel(y0);
            N = numel(tspan)-1;

            % initialize output
            tspan = tspan(:);  % ensure this is a column vector
            y = y0(:);
            Y = zeros(N+1, m);
            Y(1,:) = y.';

            % set current time value
            t = tspan(1);

            % check for legal time span
            if ~(all(diff(tspan) >= 0) || all(diff(tspan) <= 0))
                error('AdaptEuler:Evolve illegal tspan');
            end

            % initialize error weight vector
            self.w = self.error_weight(y);

            % estimate initial step size if not provided by user
            if self.h == 0.0
                % get ||y'(t0)||
                fn = self.f(t, y, args{:});

                % estimate initial h value via linearization, safety factor
                self.error_norm = max(norm(fn(:).*self.w, inf), 1e-8);
                self.h = max(self.hmin, self.safety/self.error_norm);
            end

            % iterate over output times
            for iout = 2:(N+1)

                % loop over internal steps to reach desired output time
                while (tspan(iout)-t) > sqrt(eps*tspan(iout))

                    % enforce maxit -- if we've exceeded attempts, return with failure
                    if (self.steps + self.fails) > self.maxit
                        fprintf('AdaptEuler: reached maximum iterations, returning with failure\n');
                        success = false;
                        return;
                    end

                    % bound internal time step to not exceed next output time
                    self.h = min(abs(self.h), abs(tspan(iout)-t)) * sign(self.h);

                    % initialize two solution approximations to current solution
                    y1 = y;
                    y2 = y;

                    % get RHS at this time, perform full/half step updates
                    fn = self.f(t, y, args{:});
                    y1 = y1 + self.h*fn(:);
                    y2 = y2 + (0.5*self.h)*fn(:);

                    % get RHS at half-step, perform half step update
                    fn = self.f(t+0.5*self.h, y2, args{:});
                    y2 = y2 + (0.5*self.h)*fn(:);

                    % compute error estimate
                    self.yerr = y2 - y1;

                    % compute error estimate success factor
                    self.error_norm = max(norm(self.yerr.*self.w, inf), 1e-8);

                    % compute error estimate success factor
                    eta = self.safety * self.error_norm^(-1.0/(self.p+1));  % step size growth factor
                    eta = min(eta, self.growth);                            % limit maximum growth

                    % check error
                    if self.error_norm < self.ONEPSM    % successful step

                        % update current time, solution, error weights, work counter, and upcoming stepsize
                        t = t + self.h;
                        y = 2.0*y2 - y1;
                        self.w = self.error_weight(y);
                        self.steps = self.steps + 1;
                        self.h = self.h * eta;

                    else                                % failed step
                        self.fails = self.fails + 1;

                        % adjust step size, enforcing minimum and returning with failure if needed
                        if self.h > self.hmin           % failure, but reduction possible
                            self.h = max(self.h * eta, self.hmin);
                        else                            % failed with no reduction possible
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
