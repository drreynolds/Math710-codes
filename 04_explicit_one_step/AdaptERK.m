classdef AdaptERK < handle
    % AdaptERK.m
    %
    % Adaptive-stepsize explicit Runge--Kutta solver class implementation file.
    %
    % Also contains functions to return specific explicit embedded Butcher tables.
    %
    % Class to perform adaptive time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using an embedded Runge--Kutta time stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Adaptive explicit Runge--Kutta class
    %
    % The three required arguments when constructing an AdaptERK object
    % are a function for the IVP right-hand side, a template vector
    % with the same shape and type as the IVP solution vector, and a Butcher
    % table with an embedding:
    %   f = ODE RHS function with calling syntax f(t,y).
    %   y = MATLAB array with m entries.
    %   B = explicit embedded Butcher table.
    % Other optional inputs focus on specific adaptivity options:
    %   rtol    = relative solution tolerance (scalar, >= 1e-12)
    %   atol    = absolute solution tolerance (scalar or MATLAB array with m entries, all >=0)
    %   maxit   = maximum allowed number of internal steps
    %   bias    = error bias factor
    %   growth  = maximum stepsize growth factor
    %   safety  = step size safety factor

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        f, A, b, c, d, minpq, s
        rtol = 1e-3, atol = 1e-14, maxit = 1e6
        bias = 1.0, growth = 50.0, safety = 0.85, hmin = 10*eps
        ONEMSM = 1.0 - sqrt(eps), ONEPSM = 1.0 + sqrt(eps)
        fails = 0, steps = 0, nrhs = 0, error_norm = 0.0, h = 0.0
        save_step_hist = false
        w, yerr, step_hist, z, yt, k
    end

    methods
        function self = AdaptERK(f, yTemplate, B, rtol, atol, maxit, bias, growth, safety, hmin, save_step_hist)
            if nargin < 3
                error('AdaptERK requires f, a solution template, and an embedded Butcher table.');
            end

            y = yTemplate(:);

            % Store the RHS and unpack the embedded Butcher table.
            self.f = f;
            self.A = B.A;
            self.b = B.b(:);
            self.c = B.c(:);
            self.d = B.d(:);
            self.minpq = min(B.p, B.q);

            % Overwrite default adaptivity controls when supplied.
            if nargin >= 4 && ~isempty(rtol), self.rtol = rtol; end
            if nargin >= 5 && ~isempty(atol), self.atol = atol; end
            if nargin >= 6 && ~isempty(maxit), self.maxit = maxit; end
            if nargin >= 7 && ~isempty(bias), self.bias = bias; end
            if nargin >= 8 && ~isempty(growth), self.growth = growth; end
            if nargin >= 9 && ~isempty(safety), self.safety = safety; end
            if nargin >= 10 && ~isempty(hmin), self.hmin = hmin; end
            if nargin >= 11 && ~isempty(save_step_hist), self.save_step_hist = save_step_hist; end
            if isscalar(self.atol)
                self.atol = ones(numel(y),1)*self.atol;
            else
                self.atol = self.atol(:);
            end

            % Allocate work vectors and initialize run statistics.
            self.w = ones(numel(y),1);
            self.yerr = zeros(numel(y),1);
            self.step_hist = struct('t', [], 'h', [], 'err', []);
            self.z = zeros(numel(y),1);
            self.yt = zeros(numel(y),1);
            self.s = numel(self.b);
            self.k = zeros(self.s, numel(y));

            % Embedded explicit RK tables need matching sizes and lower triangular A.
            if numel(self.c) ~= self.s || numel(self.d) ~= self.s || ...
                    size(self.A,1) ~= self.s || size(self.A,2) ~= self.s || ...
                    norm(self.A - tril(self.A,-1), inf) > 1e-14
                error('AdaptERK: incompatible embedded Butcher table supplied');
            end
        end

        function w = error_weight(self, y)
            % Error weight vector utility routine

            y = y(:);
            w = self.bias ./ (self.atol + self.rtol*abs(y));
        end

        function [t, y, success] = step(self, t, y, args)
            % Usage: t, y, success = step(t, y, args)
            %
            % Utility routine to take a single time step,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args holds optional parameters used when evaluating the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 4
                args = {};
            end
            if ~iscell(args)
                error('AdaptERK: args must be a cell array.');
            end

            y = y(:);
            % Compute each stage using only previously available explicit stages.
            self.k(1,:) = self.f(t, y, args{:}).';
            self.nrhs = self.nrhs + 1;
            for i = 2:self.s
                self.z = y;
                for j = 1:(i-1)
                    self.z = self.z + self.h * self.A(i,j) * self.k(j,:).';
                end
                self.k(i,:) = self.f(t + self.c(i)*self.h, self.z, args{:}).';
                self.nrhs = self.nrhs + 1;
            end

            % combine stage data to update the time-step solution
            for i = 1:self.s
                y = y + self.h * self.b(i) * self.k(i,:).';
            end

            % The embedded formulas differ only in their output weights.
            self.yerr(:) = 0.0;
            for i = 1:self.s
                self.yerr = self.yerr + self.h * (self.b(i)-self.d(i)) * self.k(i,:).';
            end
            % Measure the local error in the user-specified weighted norm.
            self.error_norm = max(norm(self.yerr.*self.w, inf), 1e-8);
            success = true;
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The adaptive ERK time step evolution routine
            %
            % Inputs:  tspan holds the current time interval, [t0, tf], including any
            %             intermediate times when the solution is desired, i.e.
            %              [t0, t1, ..., tf]; these may decrease (to integrate
            %              backward in time), but must be monotone
            %          y holds the initial condition, y(t0)
            %          h optionally holds the requested initial step size magnitude
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
                error('AdaptERK:Evolve args must be a cell array.');
            end

            % Store the requested initial step size; zero means estimate it below.
            self.h = h;

            y = y0(:);
            % Initialize output storage, with the first row holding the initial condition.
            tspan = tspan(:);
            m = numel(y);
            N = numel(tspan)-1;

            Y = zeros(N+1, m);
            Y(1,:) = y.';
            t = tspan(1);

            % determine the direction of integration from tspan (tdir = 1 forward in
            % time, tdir = -1 backward); the internal step size self.h is kept signed
            % in this direction, so that t + self.h always moves toward tspan(end)
            if tspan(end) >= tspan(1)
                tdir = 1.0;
            else
                tdir = -1.0;
            end

            % check for legal time span (monotone in the direction of integration)
            for n = 1:N
                if tdir*(tspan(n+1) - tspan(n)) < 0
                    error('AdaptERK:Evolve illegal tspan');
                end
            end

            % use the magnitude of any user-supplied step size, signed in direction tdir
            self.h = tdir*abs(self.h);

            % set error weights for the initial solution
            self.w = self.error_weight(y);

            % Estimate an initial step size from the weighted RHS norm if none was supplied.
            if self.h == 0.0
                fn = self.f(t, y, args{:});
                self.error_norm = max(norm(fn(:).*self.w, inf), 1e-8);
                self.h = tdir*max(self.hmin, self.safety/self.error_norm);
            end

            % iterate over output times, filling the solution history
            for iout = 2:(N+1)
                % Take as many adaptive internal steps as needed to hit this output time.
                while tdir*(tspan(iout)-t) > sqrt(eps*abs(tspan(iout)))
                    if (self.steps + self.fails) > self.maxit
                        fprintf('AdaptERK: reached maximum iterations, returning with failure\n');
                        success = false;
                        return;
                    end

                    % Do not step beyond the next requested output time.
                    self.h = tdir*min(abs(self.h), abs(tspan(iout)-t));

                    % Take one trial ERK step from the current accepted solution.
                    self.yt = y;
                    [~, self.yt, success] = self.step(t, self.yt, args);
                    if ~success
                        return;
                    end

                    % Convert the weighted error estimate into the next step-size factor.
                    eta = self.safety * self.error_norm^(-1.0/(self.minpq+1));
                    eta = min(eta, self.growth);

                    if self.save_step_hist
                        % Record every trial step so accepted and rejected steps can be plotted.
                        self.step_hist.t(end+1,1) = t;
                        self.step_hist.h(end+1,1) = self.h;
                        self.step_hist.err(end+1,1) = self.error_norm;
                    end

                    % successful step: update solution and prepare the next trial
                    if self.error_norm < self.ONEPSM
                        t = t + self.h;
                        y = self.yt;
                        self.w = self.error_weight(y);
                        self.steps = self.steps + 1;
                        self.h = self.h * eta;
                    % failed step: reduce the stepsize and retry
                    else
                        self.fails = self.fails + 1;
                        if abs(self.h) > self.hmin
                            self.h = tdir*max(abs(self.h) * eta, self.hmin);
                        else
                            fprintf('AdaptERK: error test failed at h=hmin, returning with failure\n');
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
            if nargin < 2
                rtol = 1e-3;
            end
            self.rtol = rtol;
        end

        function set_atol(self, atol)
            % Resets the scalar- or vector-valued absolute tolerance
            if nargin < 2
                atol = 1e-14;
            end
            self.atol = ones(size(self.atol))*atol;
        end

        function set_maxit(self, maxit)
            % Resets the maximum allowed iterations
            if nargin < 2
                maxit = 1e6;
            end
            self.maxit = maxit;
        end

        function set_bias(self, bias)
            % Resets the error bias factor
            if nargin < 2
                bias = 1.0;
            end
            self.bias = bias;
        end

        function set_growth(self, growth)
            % Resets the maximum stepsize growth factor
            if nargin < 2
                growth = 50.0;
            end
            self.growth = growth;
        end

        function set_safety(self, safety)
            % Resets the stepsize safety factor
            if nargin < 2
                safety = 0.85;
            end
            self.safety = safety;
        end

        function set_hmin(self, hmin)
            % Resets the minimum step size
            if nargin < 2
                hmin = 10*eps;
            end
            self.hmin = hmin;
        end

        function update_rhs(self, f)
            % Updates the RHS function (cannot change vector dimensions)
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

        function out = get_num_rhs(self)
            % Returns the total number of rhs calls
            out = self.nrhs;
        end

        function out = get_current_step(self)
            % Returns the current internal step size (signed, negative when integrating backward)
            out = self.h;
        end

        function out = get_step_history(self)
            % Returns the current step size history (step sizes h are signed, negative when integrating backward)
            out = self.step_hist;
        end

        function reset(self)
            % Resets the solver statistics

            self.fails = 0;
            self.error_norm = 0.0;
            self.nrhs = 0;
            self.steps = 0;
            self.step_hist = struct('t', [], 'h', [], 'err', []);
        end
    end

    methods (Static)
        function B = HeunEuler()
            % Usage: B = HeunEuler()
            %
            % Utility routine to return the embedded ERK table corresponding
            % to the Heun-Euler method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.0, 0.0; 1.0, 0.0];
            B.b = [0.5; 0.5];
            B.c = [0.0; 1.0];
            B.d = [1.0; 0.0];
            B.p = 2;
            B.q = 1;
        end

        function B = ERK32()
            % Usage: B = ERK32()
            %
            % Utility routine to return the embedded ERK table corresponding
            % to a 3rd-order ERK method with 2nd-order embedding.
            %
            % Reference: base method: Kutta, Z. Math. Phys. 46:435--453 (1901).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.0, 0.0, 0.0; 0.5, 0.0, 0.0; -1.0, 2.0, 0.0];
            B.b = [1.0/6.0; 2.0/3.0; 1.0/6.0];
            B.c = [0.0; 0.5; 1.0];
            B.d = [0.0; 1.0; 0.0];
            B.p = 3;
            B.q = 2;
        end

        function B = BogackiShampine()
            % Usage: B = BogackiShampine()
            %
            % Utility routine to return the embedded ERK table corresponding
            % to the Bogacki-Shampine embedded ERK method.
            %
            % Reference: Bogacki & Shampine, Appl. Math. Lett. 2 (1989),
            %            doi:10.1016/0893-9659(89)90079-7.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.0, 0.0, 0.0, 0.0; ...
                   0.5, 0.0, 0.0, 0.0; ...
                   0.0, 3.0/4.0, 0.0, 0.0; ...
                   2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0];
            B.b = [2.0/9.0; 1.0/3.0; 4.0/9.0; 0.0];
            B.c = [0.0; 0.5; 3.0/4.0; 1.0];
            B.d = [7.0/24.0; 1.0/4.0; 1.0/3.0; 1.0/8.0];
            B.p = 3;
            B.q = 2;
        end

        function B = DormandPrince()
            % Usage: B = DormandPrince()
            %
            % Utility routine to return the embedded ERK table corresponding
            % to the Dormand Prince method.
            %
            % Reference: Dormand & Prince, J. Comput. Appl. Math. 6 (1980),
            %            doi:10.1016/0771-050X(80)90013-3.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; ...
                   1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; ...
                   3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0; ...
                   44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0; ...
                   19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0; ...
                   9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0; ...
                   35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0];
            B.b = [35.0/384.0; 0.0; 500.0/1113.0; 125.0/192.0; -2187.0/6784.0; 11.0/84.0; 0.0];
            B.c = [0.0; 1.0/5.0; 3.0/10.0; 4.0/5.0; 8.0/9.0; 1.0; 1.0];
            B.d = [5179.0/57600.0; 0.0; 7571.0/16695.0; 393.0/640.0; -92097.0/339200.0; 187.0/2100.0; 1.0/40.0];
            B.p = 5;
            B.q = 4;
        end

        function B = Verner65()
            % Usage: B = Verner65()
            %
            % Utility routine to return the embedded ERK table corresponding
            % to the a 6th-order ERK method with 5th-order embedding by Verner.
            %
            % Reference: Hull, Enright & Jackson, User's guide for DVERK, Technical
            %            Report 100, Department of Computer Science, University of
            %            Toronto (1976).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; ...
                   1.0/6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; ...
                   4.0/75.0, 16.0/75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; ...
                   5.0/6.0, -8.0/3.0, 5.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0; ...
                   -165.0/64.0, 55.0/6.0, -425.0/64.0, 85.0/96.0, 0.0, 0.0, 0.0, 0.0; ...
                   12.0/5.0, -8.0, 4015.0/612.0, -11.0/36.0, 88.0/255.0, 0.0, 0.0, 0.0; ...
                   -8263.0/15000.0, 124.0/75.0, -643.0/680.0, -81.0/250.0, 2484.0/10625.0, 0.0, 0.0, 0.0; ...
                   3501.0/1720.0, -300.0/43.0, 297275.0/52632.0, -319.0/2322.0, 24068.0/84065.0, 0.0, 3850.0/26703.0, 0.0];
            B.b = [3.0/40.0; 0.0; 875.0/2244.0; 23.0/72.0; 264.0/1955.0; 0.0; 125.0/11592.0; 43.0/616.0];
            B.c = [0.0; 1.0/6.0; 4.0/15.0; 2.0/3.0; 5.0/6.0; 1.0; 1.0/15.0; 1.0];
            B.d = [13.0/160.0; 0.0; 2375.0/5984.0; 5.0/16.0; 12.0/85.0; 3.0/44.0; 0.0; 0.0];
            B.p = 6;
            B.q = 5;
        end
        % Additional embedded explicit Runge--Kutta tables.

        function B = Ascher222ERK()
            % Usage: B = Ascher222ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Ascher(2,2,2)-ERK method.
            %
            % Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
            %            doi:10.1016/S0168-9274(97)00056-1.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0.29289321881345243, 0, 0; ...
                   -0.70710678118654791, 1.7071067811865479, 0];
            B.b = [-0.70710678118654791; 1.7071067811865479; 0];
            B.c = [0; 0.29289321881345243; 1];
            B.d = [0; 0.59999999999999998; 0.40000000000000002];
            B.p = 2;
            B.q = 1;
        end

        function B = SSP22ERK()
            % Usage: B = SSP22ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSP(2,2)-ERK method.
            %
            % Reference: Fekete, Conde & Shadid, J. Comput. Appl. Math. 412 (2022),
            %            doi:10.1016/j.cam.2022.114325.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0; ...
                   1, 0];
            B.b = [0.5; 0.5];
            B.c = [0; 1];
            B.d = [0.69402145920762603; 0.30597854079237397];
            B.p = 2;
            B.q = 1;
        end

        function B = SSP32ERK()
            % Usage: B = SSP32ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSP(3,2)-ERK method.
            %
            % Reference: Fekete, Conde & Shadid, J. Comput. Appl. Math. 412 (2022),
            %            doi:10.1016/j.cam.2022.114325.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0.5, 0, 0; ...
                   0.5, 0.5, 0];
            B.b = [0.33333333333333331; 0.33333333333333331; 0.33333333333333331];
            B.c = [0; 0.5; 1];
            B.d = [0.44444444444444442; 0.33333333333333331; 0.22222222222222221];
            B.p = 2;
            B.q = 1;
        end

        function B = SSP42ERK()
            % Usage: B = SSP42ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSP(4,2)-ERK method.
            %
            % Reference: Fekete, Conde & Shadid, J. Comput. Appl. Math. 412 (2022),
            %            doi:10.1016/j.cam.2022.114325.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0; ...
                   0.33333333333333331, 0, 0, 0; ...
                   0.33333333333333331, 0.33333333333333331, 0, 0; ...
                   0.33333333333333331, 0.33333333333333331, 0.33333333333333331, 0];
            B.b = [0.25; 0.25; 0.25; 0.25];
            B.c = [0; 0.33333333333333331; 0.66666666666666663; 1];
            B.d = [0.3125; 0.25; 0.25; 0.1875];
            B.p = 2;
            B.q = 1;
        end

        function B = SSP102ERK()
            % Usage: B = SSP102ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSP(10,2)-ERK method.
            %
            % Reference: Fekete, Conde & Shadid, J. Comput. Appl. Math. 412 (2022),
            %            doi:10.1016/j.cam.2022.114325.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.1111111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.1111111111111111, 0.1111111111111111, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0, 0, 0, 0, 0, 0, 0; ...
                   0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0, 0, 0, 0, 0, 0; ...
                   0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0, 0, 0, 0, 0; ...
                   0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0, 0, 0, 0; ...
                   0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0, 0, 0; ...
                   0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0, 0; ...
                   0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0];
            B.b = [0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001];
            B.c = [0; 0.1111111111111111; 0.22222222222222221; 0.33333333333333331; 0.44444444444444442; 0.55555555555555558; 0.66666666666666663; 0.77777777777777768; 0.88888888888888884; 1];
            B.d = [0.11; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.089999999999999997];
            B.p = 2;
            B.q = 1;
        end

        function B = ARK232ERK()
            % Usage: B = ARK232ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % ARK(2,3,2)-ERK method.
            %
            % Reference: Giraldo, Kelly & Constantinescu, SIAM J. Sci. Comput. 35
            %            (2013), doi:10.1137/120876034.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0.58578643762690508, 0, 0; ...
                   0.028595479208968433, 0.97140452079103157, 0];
            B.b = [0.35355339059327373; 0.35355339059327373; 0.29289321881345254];
            B.c = [0; 0.58578643762690508; 1];
            B.d = [0.32322330470336313; 0.32322330470336313; 0.35355339059327373];
            B.p = 2;
            B.q = 1;
        end

        function B = SSP2332LspumERK()
            % Usage: B = SSP2332LspumERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSP2(3,3,2)-lspum-ERK method.
            %
            % Reference: Higueras, Happenhofer, Koch & Kupka, J. Comput. Appl. Math. 272
            %            (2014), doi:10.1016/j.cam.2014.05.011.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0.83333333333333337, 0, 0; ...
                   0.45833333333333331, 0.45833333333333331, 0];
            B.b = [0.43636363636363634; 0.20000000000000001; 0.36363636363636365];
            B.c = [0; 0.83333333333333337; 0.91666666666666663];
            B.d = [0.43160569105691055; 0.19718218773096821; 0.37121212121212122];
            B.p = 2;
            B.q = 1;
        end

        function B = GiraldoARK2ERK()
            % Usage: B = GiraldoARK2ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Giraldo-ARK2-ERK method.
            %
            % Reference: Giraldo, Kelly & Constantinescu, SIAM J. Sci. Comput. 35
            %            (2013), doi:10.1137/120876034.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0.58578643762690485, 0, 0; ...
                   0.028595479208968284, 0.97140452079103168, 0];
            B.b = [0.35355339059327373; 0.35355339059327373; 0.29289321881345254];
            B.c = [0; 0.58578643762690485; 1];
            B.d = [0.32322330470336313; 0.32322330470336313; 0.35355339059327373];
            B.p = 2;
            B.q = 1;
        end

        function B = ARK324L2SAERK()
            % Usage: B = ARK324L2SAERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % ARK3(2)4L[2]SA-ERK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 44 (2003),
            %            doi:10.1016/S0168-9274(02)00138-1.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0; ...
                   0.87173304301691801, 0, 0, 0; ...
                   0.52758901197630037, 0.072410988023699593, 0, 0; ...
                   0.39909600767607012, -0.43755765461351942, 1.0384616469374492, 0];
            B.b = [0.18764102434672383; -0.59529747357695495; 0.97178992772177208; 0.435866521508459];
            B.c = [0; 0.87173304301691801; 0.59999999999999998; 1];
            B.d = [0.21474028622338914; -0.4851622638849391; 0.86872500252038753; 0.40169697514116243];
            B.p = 3;
            B.q = 2;
        end

        function B = SSP43ERK()
            % Usage: B = SSP43ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSP(4,3)-ERK method.
            %
            % Reference: Fekete, Conde & Shadid, J. Comput. Appl. Math. 412 (2022),
            %            doi:10.1016/j.cam.2022.114325.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0; ...
                   0.5, 0, 0, 0; ...
                   0.5, 0.5, 0, 0; ...
                   0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0];
            B.b = [0.16666666666666666; 0.16666666666666666; 0.16666666666666666; 0.5];
            B.c = [0; 0.5; 1; 0.5];
            B.d = [0.25; 0.25; 0.25; 0.25];
            B.p = 3;
            B.q = 2;
        end

        function B = SSP93ERK()
            % Usage: B = SSP93ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSP(9,3)-ERK method.
            %
            % Reference: Fekete, Conde & Shadid, J. Comput. Appl. Math. 412 (2022),
            %            doi:10.1016/j.cam.2022.114325.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.16666666666666666, 0, 0, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0, 0, 0; ...
                   0.16666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.16666666666666666, 0, 0; ...
                   0.16666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.16666666666666666, 0.16666666666666666, 0];
            B.b = [0.16666666666666666; 0.066666666666666666; 0.066666666666666666; 0.066666666666666666; 0.066666666666666666; 0.066666666666666666; 0.16666666666666666; 0.16666666666666666; 0.16666666666666666];
            B.c = [0; 0.16666666666666666; 0.33333333333333331; 0.5; 0.66666666666666663; 0.83333333333333326; 0.49999999999999994; 0.66666666666666663; 0.83333333333333326];
            B.d = [0.1111111111111111; 0.1111111111111111; 0.1111111111111111; 0.1111111111111111; 0.1111111111111111; 0.1111111111111111; 0.1111111111111111; 0.1111111111111111; 0.1111111111111111];
            B.p = 3;
            B.q = 2;
        end

        function B = SSP163ERK()
            % Usage: B = SSP163ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSP(16,3)-ERK method.
            %
            % Reference: Fekete, Conde & Shadid, J. Comput. Appl. Math. 412 (2022),
            %            doi:10.1016/j.cam.2022.114325.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.083333333333333329, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.083333333333333329, 0.083333333333333329, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0, 0; ...
                   0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.035714285714285712, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0];
            B.b = [0.083333333333333329; 0.083333333333333329; 0.083333333333333329; 0.035714285714285712; 0.035714285714285712; 0.035714285714285712; 0.035714285714285712; 0.035714285714285712; 0.035714285714285712; 0.035714285714285712; 0.083333333333333329; 0.083333333333333329; 0.083333333333333329; 0.083333333333333329; 0.083333333333333329; 0.083333333333333329];
            B.c = [0; 0.083333333333333329; 0.16666666666666666; 0.25; 0.33333333333333331; 0.41666666666666663; 0.49999999999999994; 0.58333333333333326; 0.66666666666666663; 0.75; 0.49999999999999989; 0.58333333333333326; 0.66666666666666663; 0.75; 0.83333333333333337; 0.91666666666666674];
            B.d = [0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625; 0.0625];
            B.p = 3;
            B.q = 2;
        end

        function B = SSPRK33ShuOsherERK()
            % Usage: B = SSPRK33ShuOsherERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSPRK(3,3)-Shu-Osher-ERK method.
            %
            % References: method: Shu & Osher, J. Comput. Phys. 77 (1988),
            %             doi:10.1016/0021-9991(88)90177-5.
            %             embedding: Fekete, Conde & Shadid, J. Comput. Appl. Math. 412
            %             (2022), doi:10.1016/j.cam.2022.114325.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   1, 0, 0; ...
                   0.25, 0.25, 0];
            B.b = [0.16666666666666666; 0.16666666666666666; 0.66666666666666663];
            B.c = [0; 1; 0.5];
            B.d = [0.29148541887840901; 0.29148541887840901; 0.41702916224318198];
            B.p = 3;
            B.q = 2;
        end

        function B = ARK436L2SAERK()
            % Usage: B = ARK436L2SAERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % ARK4(3)6L[2]SA-ERK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 44 (2003),
            %            doi:10.1016/S0168-9274(02)00138-1.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.5, 0, 0, 0, 0, 0; ...
                   0.221776, 0.110224, 0, 0, 0, 0; ...
                   -0.04884659515311858, -0.177720652326401, 0.84656724747951961, 0, 0, 0; ...
                   -0.15541685842491548, -0.3567050098221991, 1.0587258798684427, 0.30339598837867193, 0, 0; ...
                   0.20142435067267633, 0.0087420578429041849, 0.15993995707168115, 0.40382906052207751, 0.22606457389066084, 0];
            B.b = [0.15791629516167136; 0; 0.18675894052400077; 0.68056529530933463; -0.27524053099500667; 0.25];
            B.c = [0; 0.5; 0.33200000000000002; 0.62; 0.84999999999999998; 1];
            B.d = [0.15471180076321217; 0; 0.18920519166068023; 0.70204537122892186; -0.31918739906357912; 0.27322503541076487];
            B.p = 4;
            B.q = 3;
        end

        function B = ARK437L2SAERK()
            % Usage: B = ARK437L2SAERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % ARK4(3)7L[2]SA-ERK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 136 (2019),
            %            doi:10.1016/j.apnum.2018.10.007.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0; ...
                   0.247, 0, 0, 0, 0, 0, 0; ...
                   0.061749999999999999, 0.35990537495307723, 0, 0, 0, 0, 0; ...
                   0.05301658458687121, 0.35949264529328429, -0.077509229880155461, 0, 0, 0, 0; ...
                   0.058417159447800002, -0.16313824817772324, -0.19732090979798411, 0.37704199852790737, 0, 0, 0; ...
                   0.53853032270810797, -0.45497746895916669, 1.2562905623429941, -0.47828452721130055, -0.16155888888063494, 0, 0; ...
                   0.23221715782277083, 0.23221715782277083, -6.809994375038098, 7.3618585524244216, -1.3748790779406981, 1.3585805849088326, 0];
            B.b = [0; 0; 0.51611072831742366; -0.14606356393857081; 0.23473048589019332; 0.27172234973095377; 0.1235];
            B.c = [0; 0.247; 0.42165537495307726; 0.33500000000000002; 0.074999999999999997; 0.69999999999999996; 1];
            B.d = [0; 0; 0.51752174615934821; -0.15173820706113939; 0.23672007870234135; 0.27544638219944978; 0.12205000000000001];
            B.p = 4;
            B.q = 3;
        end

        function B = SayfyAburub43ERK()
            % Usage: B = SayfyAburub43ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Sayfy-Aburub-4-3-ERK method.
            %
            % Reference: Sayfy & Aburub, Int. J. Comput. Math. 79 (2002),
            %            doi:10.1080/00207160212109.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.5, 0, 0, 0, 0, 0; ...
                   -1, 2, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.66666666666666663, 0.16666666666666666, 0, 0, 0; ...
                   0.13700000000000001, 0.22600000000000001, 0.13700000000000001, 0, 0, 0; ...
                   0.45200000000000001, -0.90400000000000003, -0.54800000000000004, 0, 2, 0];
            B.b = [0.16666666666666666; 0.33333333333333331; 0.083333333333333329; 0; 0.33333333333333331; 0.083333333333333329];
            B.c = [0; 0.5; 1; 1; 0.5; 1];
            B.d = [0.16666666666666666; 0.66666666666666663; 0.16666666666666666; 0; 0; 0];
            B.p = 4;
            B.q = 3;
        end

        function B = SSP104ERK()
            % Usage: B = SSP104ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % SSP(10,4)-ERK method.
            %
            % Reference: Fekete, Conde & Shadid, J. Comput. Appl. Math. 412 (2022),
            %            doi:10.1016/j.cam.2022.114325.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.16666666666666666, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0, 0, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0, 0, 0, 0, 0, 0; ...
                   0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0, 0, 0, 0, 0; ...
                   0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.16666666666666666, 0, 0, 0, 0; ...
                   0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.16666666666666666, 0.16666666666666666, 0, 0, 0; ...
                   0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0, 0; ...
                   0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.066666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0];
            B.b = [0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001; 0.10000000000000001];
            B.c = [0; 0.16666666666666666; 0.33333333333333331; 0.5; 0.66666666666666663; 0.33333333333333331; 0.5; 0.66666666666666663; 0.83333333333333326; 0.99999999999999989];
            B.d = [0.20000000000000001; 0; 0; 0.29999999999999999; 0; 0; 0.20000000000000001; 0; 0.29999999999999999; 0];
            B.p = 4;
            B.q = 3;
        end

        function B = Merson43ERK()
            % Usage: B = Merson43ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Merson-4-3-ERK method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0; ...
                   0.33333333333333331, 0, 0, 0, 0; ...
                   0.16666666666666666, 0.16666666666666666, 0, 0, 0; ...
                   0.125, 0, 0.375, 0, 0; ...
                   0.5, 0, -1.5, 2, 0];
            B.b = [0.16666666666666666; 0; 0; 0.66666666666666663; 0.16666666666666666];
            B.c = [0; 0.33333333333333331; 0.33333333333333331; 0.5; 1];
            B.d = [0.10000000000000001; 0; 0.29999999999999999; 0.40000000000000002; 0.20000000000000001];
            B.p = 4;
            B.q = 3;
        end

        function B = Zonneveld43ERK()
            % Usage: B = Zonneveld43ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Zonneveld-4-3-ERK method.
            %
            % Reference: Zonneveld, Automatic integration of ordinary differential
            %            equations, Report R743, Mathematisch Centrum, Amsterdam (1963).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0; ...
                   0.5, 0, 0, 0, 0; ...
                   0, 0.5, 0, 0, 0; ...
                   0, 0, 1, 0, 0; ...
                   0.15625, 0.21875, 0.40625, -0.03125, 0];
            B.b = [0.16666666666666666; 0.33333333333333331; 0.33333333333333331; 0.16666666666666666; 0];
            B.c = [0; 0.5; 0.5; 1; 0.75];
            B.d = [-0.5; 2.3333333333333335; 2.3333333333333335; 2.1666666666666665; -5.333333333333333];
            B.p = 4;
            B.q = 3;
        end

        function B = ARK548L2SAERK()
            % Usage: B = ARK548L2SAERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % ARK5(4)8L[2]SA-ERK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 44 (2003),
            %            doi:10.1016/S0168-9274(02)00138-1.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.40999999999999998, 0, 0, 0, 0, 0, 0, 0; ...
                   0.17753520777580992, 0.082394376672570227, 0, 0, 0, 0, 0, 0; ...
                   0.12262307902976895, 0, 0.075527407662734677, 0, 0, 0, 0, 0; ...
                   2.2901776494938124, 0, 11.244925765143737, -12.615103414637549, 0, 0, 0, 0; ...
                   0.40294451783476792, 0, 1.3540123800181454, -1.4857008988406062, -0.031255999012307065, 0, 0, 0; ...
                   1.4641384430844078, 0, 7.2304686798580153, -7.8446071229424232, -0.125, -0.125, 0, 0; ...
                   -1.6748080049977643, 0, -6.3894386455592986, 14.692200676518024, 0.094666234325682705, -7.2111573276528604, 1.4885370673662177, 0];
            B.b = [-0.09554858675139874; 0; 0; 2.3386928037652464; -0.14043175608247527; -2.0705877079565589; 0.76287524702518661; 0.20499999999999999];
            B.c = [0; 0.40999999999999998; 0.25992958444838016; 0.19815048669250362; 0.92000000000000004; 0.23999999999999999; 0.59999999999999998; 1];
            B.d = [-0.09957696480500873; 0; 0; 2.4071628799997749; -0.1601481830855136; -2.1442365964445265; 0.77956562242499827; 0.21723324191027585];
            B.p = 5;
            B.q = 4;
        end

        function B = ARK548L2SAbERK()
            % Usage: B = ARK548L2SAbERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % ARK5(4)8L[2]SAb-ERK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 136 (2019),
            %            doi:10.1016/j.apnum.2018.10.007.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.44444444444444442, 0, 0, 0, 0, 0, 0, 0; ...
                   0.1111111111111111, 0.64760301386068775, 0, 0, 0, 0, 0, 0; ...
                   0.091829866647747907, 0.035448567517799241, -0.012008999601505184, 0, 0, 0, 0, 0; ...
                   -0.34252354516023137, -0.26767785943050182, 0.11056894178117282, 0.85638959121387881, 0, 0, 0, 0; ...
                   -0.0097722828790043955, 0.21070865398661751, 0.075924120912175361, 0.20765518596381696, 0.23548432201639455, 0, 0, 0; ...
                   0.46686370681500694, 1.2903598800650855, 0.37840596884419414, -0.56345584032826157, -0.2883238346202236, -0.32884988077580141, 0, 0; ...
                   0.61439671625166914, 0.61439671625166914, 0.31747780106686158, -0.71215206239529361, 0.11498708015310211, 0.09139031575415682, -0.040496567082165244, 0];
            B.b = [0; 0; 0.17366253573581261; 0.25479166260812353; 0.24190176845094791; 0.30740485830222825; -0.19998304731933453; 0.22222222222222221];
            B.c = [0; 0.44444444444444442; 0.75871412497179891; 0.11526943456404197; 0.3567571284043185; 0.71999999999999997; 0.95499999999999996; 1];
            B.d = [0; 0; 0.062724216952707135; 0.25523315714677963; 0.23902754916001318; 0.39907952207535802; -0.14315725125850667; 0.18709280592364871];
            B.p = 5;
            B.q = 4;
        end

        function B = FehlbergERK()
            % Usage: B = FehlbergERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Fehlberg-ERK method.
            %
            % Reference: Fehlberg, NASA Technical Report R-315 (1969).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.25, 0, 0, 0, 0, 0; ...
                   0.09375, 0.28125, 0, 0, 0, 0; ...
                   0.87938097405553028, -3.2771961766044608, 3.3208921256258535, 0, 0, 0; ...
                   2.0324074074074074, -8, 7.1734892787524362, -0.20589668615984405, 0, 0; ...
                   -0.29629629629629628, 2, -1.3816764132553607, 0.45297270955165692, -0.27500000000000002, 0];
            B.b = [0.11851851851851852; 0; 0.51898635477582844; 0.50613149034201665; -0.17999999999999999; 0.036363636363636362];
            B.c = [0; 0.25; 0.375; 0.92307692307692313; 1; 0.5];
            B.d = [0.11574074074074074; 0; 0.54892787524366471; 0.53533138401559455; -0.20000000000000001; 0];
            B.p = 5;
            B.q = 4;
        end

        function B = CashKarpERK()
            % Usage: B = CashKarpERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Cash-Karp-ERK method.
            %
            % Reference: Cash & Karp, ACM Trans. Math. Software 16 (1990),
            %            doi:10.1145/79505.79507.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.20000000000000001, 0, 0, 0, 0, 0; ...
                   0.074999999999999997, 0.22500000000000001, 0, 0, 0, 0; ...
                   0.29999999999999999, -0.90000000000000002, 1.2, 0, 0, 0; ...
                   -0.20370370370370369, 2.5, -2.5925925925925926, 1.2962962962962963, 0, 0; ...
                   0.029495804398148147, 0.341796875, 0.041594328703703706, 0.40034541377314814, 0.061767578125, 0];
            B.b = [0.097883597883597878; 0; 0.40257648953301128; 0.21043771043771045; 0; 0.28910220214568039];
            B.c = [0; 0.20000000000000001; 0.29999999999999999; 0.59999999999999998; 1; 0.875];
            B.d = [0.10217737268518519; 0; 0.38390790343915343; 0.24459273726851852; 0.019321986607142856; 0.25];
            B.p = 5;
            B.q = 4;
        end

        function B = Verner65bERK()
            % Usage: B = Verner65bERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Verner-6-5b-ERK method.
            %
            % Reference: Verner, Numer. Algorithms 53 (2010),
            %            doi:10.1007/s11075-009-9290-3.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.059999999999999998, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.019239962962962962, 0.07669337037037037, 0, 0, 0, 0, 0, 0, 0; ...
                   0.035975, 0, 0.10792499999999999, 0, 0, 0, 0, 0, 0; ...
                   1.3186834152331484, 0, -5.0420580636285619, 4.2206746483954136, 0, 0, 0, 0, 0; ...
                   -41.872591664327508, 0, 159.43256216313748, -122.11921356501004, 5.5317430662000531, 0, 0, 0, 0; ...
                   -54.430156935316504, 0, 207.06725136501848, -158.61081378458999, 6.9918165859502421, -0.018597231062203231, 0, 0, 0; ...
                   -54.663741787281978, 0, 207.95280625538936, -159.28895747449951, 7.0187437407969444, -0.018338785905045722, -0.00051194849978820987, 0, 0; ...
                   0.034389578683570357, 0, 0, 0.25826245556335037, 0.4209371189673537, 4.4053964696693102, -176.48311902429865, 172.36413340141507, 0];
            B.b = [0.034389578683570357; 0; 0; 0.25826245556335037; 0.4209371189673537; 4.4053964696693102; -176.48311902429865; 172.36413340141507; 0];
            B.c = [0; 0.059999999999999998; 0.095933333333333329; 0.1439; 0.49730000000000002; 0.97250000000000003; 0.99950000000000006; 1; 1];
            B.d = [0.049099676483824899; 0; 0; 0.22511122295165242; 0.46946822530295618; 0.80657922499888679; 0; -0.6071194891777959; 0.056861139440475689];
            B.p = 6;
            B.q = 5;
        end

        function B = Verner76ERK()
            % Usage: B = Verner76ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Verner-7-6-ERK method.
            %
            % Reference: Verner, Numer. Algorithms 53 (2010),
            %            doi:10.1007/s11075-009-9290-3.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.0050000000000000001, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   -1.07679012345679, 1.1856790123456791, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.040833333333333333, 0, 0.1225, 0, 0, 0, 0, 0, 0, 0; ...
                   0.63891392362557264, 0, -2.4556726382236569, 2.2722587145980842, 0, 0, 0, 0, 0, 0; ...
                   -2.6615773750187568, 0, 10.804513886456139, -8.3539146573961993, 0.82048759495665691, 0, 0, 0, 0, 0; ...
                   6.0677414346967717, 0, -24.711273635911088, 20.427517930788895, -1.9061579788166472, 1.0061722492420679, 0, 0, 0, 0; ...
                   12.054670076253204, 0, -49.754784950468988, 41.142888638604674, -4.4617601499740038, 2.0423348222391753, -0.098348436654061067, 0, 0, 0; ...
                   10.138146522881808, 0, -42.641136031717501, 35.76384003992257, -4.3480228403929075, 2.0098622683770357, 0.34874904603382717, -0.27143900510483127, 0, 0; ...
                   -45.030072034298676, 0, 187.32724376545889, -154.02882369350186, 18.564653063475362, -7.1418096792950791, 1.3088085781613787, 0, 0, 0];
            B.b = [0.04715561848627222; 0; 0; 0.25750564298434153; 0.26216653977412624; 0.15216092656738558; 0.49399691700324849; -0.29430311714032503; 0.081317472324951109; 0];
            B.c = [0; 0.0050000000000000001; 0.10888888888888888; 0.16333333333333333; 0.45550000000000002; 0.6095094489978381; 0.88400000000000001; 0.92500000000000004; 1; 1];
            B.d = [0.044608606606341174; 0; 0; 0.26716403785713727; 0.22010183001772932; 0.21884317031431569; 0.22898717054112028; 0; 0; 0.02029518466335628];
            B.p = 7;
            B.q = 6;
        end

        function B = Verner87ERK()
            % Usage: B = Verner87ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Verner-8-7-ERK method.
            %
            % Reference: Verner, Numer. Algorithms 53 (2010),
            %            doi:10.1007/s11075-009-9290-3.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.050000000000000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   -0.0069931640624999996, 0.11355566406250001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.039960937500000002, 0, 0.1198828125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.36139756280045754, 0, -1.3415240667004928, 1.3701265039000352, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.049047202797202795, 0, 0, 0.23509720422144048, 0.18085559298135673, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.06169289044289044, 0, 0, 0.11236568314640277, -0.038850460714513667, 0.01979188712522046, 0, 0, 0, 0, 0, 0, 0; ...
                   -1.767630240222327, 0, 0, -62.5, -6.0618893773766693, 5.6508231982227635, 65.621696419376235, 0, 0, 0, 0, 0, 0; ...
                   -1.1809450665549708, 0, 0, -41.504734411143211, -4.4344383191037249, 4.2604081885861333, 43.753640224461712, 0.00787142548991231, 0, 0, 0, 0, 0; ...
                   -1.2814059994414884, 0, 0, -45.047139960139866, -4.7313620694495757, 4.5149670165938076, 47.449095571729849, 0.01059228297111661, -0.0057468422638446157, 0, 0, 0, 0; ...
                   -1.7244701342624851, 0, 0, -60.92349008483054, -5.951518376222392, 5.5565237306984567, 63.98301198033306, 0.014642028250414963, 0.064604087723582032, -0.079303231690088793, 0, 0, 0; ...
                   -3.301622667747079, 0, 0, -118.01127235975251, -10.141422388456112, 9.139311332232058, 123.37594282840428, 4.6232443788745812, -3.3832777380682018, 4.5275921003246182, -5.8284954858116231, 0, 0; ...
                   -3.0395150337663086, 0, 0, -109.26086808941763, -9.2906424974002917, 8.430504981764912, 114.20100103783314, -0.96372713421454792, -5.0348840888021895, 5.9581308240029234, 0, 0, 0];
            B.b = [0.044279894190079508; 0; 0; 0; 0; 0.3541049391724449; 0.24796921549564377; -15.694202038838085; 25.084064965558564; -31.738367786260277; 22.938283273988784; -0.23613246330715421; 0];
            B.c = [0; 0.050000000000000003; 0.1065625; 0.15984375000000001; 0.39000000000000001; 0.46500000000000002; 0.155; 0.94299999999999995; 0.90180204173585699; 0.90900000000000003; 0.93999999999999995; 1; 1];
            B.d = [0.044312615229089795; 0; 0; 0; 0; 0.35460956423432266; 0.24784804313666531; 4.4481347324757845; 19.846886366118735; -23.581623377465618; 0; 0; -0.36016794372897754];
            B.p = 8;
            B.q = 7;
        end

        function B = Fehlberg87ERK()
            % Usage: B = Fehlberg87ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Fehlberg-8-7-ERK method.
            %
            % Reference: Fehlberg, NASA Technical Report R-287 (1968).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.07407407407407407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.027777777777777776, 0.083333333333333329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.041666666666666664, 0, 0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.41666666666666669, 0, -1.5625, 1.5625, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.050000000000000003, 0, 0, 0.25, 0.20000000000000001, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   -0.23148148148148148, 0, 0, 1.1574074074074074, -2.4074074074074074, 2.3148148148148149, 0, 0, 0, 0, 0, 0, 0; ...
                   0.10333333333333333, 0, 0, 0, 0.27111111111111114, -0.22222222222222221, 0.014444444444444444, 0, 0, 0, 0, 0, 0; ...
                   2, 0, 0, -8.8333333333333339, 15.644444444444444, -11.888888888888889, 0.74444444444444446, 3, 0, 0, 0, 0, 0; ...
                   -0.84259259259259256, 0, 0, 0.21296296296296297, -7.2296296296296294, 5.7592592592592595, -0.31666666666666665, 2.8333333333333335, -0.083333333333333329, 0, 0, 0, 0; ...
                   0.58121951219512191, 0, 0, -2.0792682926829267, 4.3863414634146345, -3.6707317073170733, 0.52024390243902441, 0.54878048780487809, 0.27439024390243905, 0.43902439024390244, 0, 0, 0; ...
                   0.014634146341463415, 0, 0, 0, 0, -0.14634146341463414, -0.014634146341463415, -0.073170731707317069, 0.073170731707317069, 0.14634146341463414, 0, 0, 0; ...
                   -0.43341463414634146, 0, 0, -2.0792682926829267, 4.3863414634146345, -3.524390243902439, 0.53487804878048784, 0.62195121951219512, 0.20121951219512196, 0.29268292682926828, 0, 1, 0];
            B.b = [0; 0; 0; 0; 0; 0.32380952380952382; 0.25714285714285712; 0.25714285714285712; 0.03214285714285714; 0.03214285714285714; 0; 0.04880952380952381; 0.04880952380952381];
            B.c = [0; 0.07407407407407407; 0.1111111111111111; 0.16666666666666666; 0.41666666666666669; 0.5; 0.83333333333333337; 0.16666666666666666; 0.66666666666666663; 0.33333333333333331; 1; 0; 1];
            B.d = [0.04880952380952381; 0; 0; 0; 0; 0.32380952380952382; 0.25714285714285712; 0.25714285714285712; 0.03214285714285714; 0.03214285714285714; 0.04880952380952381; 0; 0];
            B.p = 8;
            B.q = 7;
        end

        function B = Verner98ERK()
            % Usage: B = Verner98ERK()
            %
            % Utility routine to return the embedded ERK table corresponding to the
            % Verner-9-8-ERK method.
            %
            % Reference: Verner, Numer. Algorithms 53 (2010),
            %            doi:10.1007/s11075-009-9290-3.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.034619999999999998, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   -0.038933543885728734, 0.13595789452450918, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.03638413148954267, 0, 0.109152394468628, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   2.0257639143939699, 0, -7.6380238364962922, 6.1732599221023223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.051122755894060609, 0, 0, 0.17708237945550215, 0.00080277624092225019, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.13160063579752163, 0, 0, -0.29572762526696367, 0.087813780356429519, 0.6213052975225275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.07166666666666667, 0, 0, 0, 0, 0.33055335789153195, 0.24277997544180138, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.071806640625000001, 0, 0, 0, 0, 0.32943802832281771, 0.11651900292718229, -0.034013671874999998, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.048367576463406468, 0, 0, 0, 0, 0.039289899256761643, 0.10547409458903446, -0.021438652846483126, -0.10412291746271944, 0, 0, 0, 0, 0, 0, 0; ...
                   -0.026645614872014785, 0, 0, 0, 0, 0.033333333333333333, -0.16310722448724671, 0.033960816841277615, 0.1572319413814626, 0.21522674780318796, 0, 0, 0, 0, 0, 0; ...
                   0.036890092487086225, 0, 0, 0, 0, -0.1465181576725543, 0.22425777681720244, 0.022944057170660725, -0.0035850052905728761, 0.086692233164443855, 0.43838406519683376, 0, 0, 0, 0, 0; ...
                   -0.48660122151133406, 0, 0, 0, 0, -6.3046026502828534, -0.28124561828947259, -2.6790192362198493, 0.51881566392415757, 1.3653531876033418, 5.8850910885039465, 2.8028087862720628, 0, 0, 0, 0; ...
                   0.41853674577534716, 0, 0, 0, 0, 6.7245475819064593, -0.42544428016461178, 3.3432791530012658, 0.61708166311753776, -0.92996612393993283, -6.0999488047510111, -3.0022061878893989, 0.25532025294434457, 0, 0, 0; ...
                   -0.77937408612288461, 0, 0, 0, 0, -13.937342538107776, 1.2520488533793572, -14.69150040801687, -0.49470505853314167, 2.2429749091462368, 13.367893803828643, 14.396650486650687, -0.79758133317767999, 0.44093537095342777, 0, 0; ...
                   2.0580513374668863, 0, 0, 0, 0, 22.357937727968032, 0.90949810997556335, 35.891100982402641, -3.4425150276244536, -4.8654813580363685, -18.909803813543427, -34.263544480304517, 1.2647565216956427, 0, 0, 0];
            B.b = [0.014611976858423152; 0; 0; 0; 0; 0; 0; -0.39152118623313392; 0.23109325002895065; 0.12747667699928525; 0.22464341762041579; 0.5684352689748513; 0.058258715572158275; 0.13643174034822156; 0.030570139830827976; 0];
            B.c = [0; 0.034619999999999998; 0.097024350638780441; 0.14553652595817068; 0.56100000000000005; 0.229007911590485; 0.54499208840951496; 0.64500000000000002; 0.48375000000000001; 0.067570000000000005; 0.25; 0.65906506187309988; 0.8206; 0.9012; 1; 1];
            B.d = [0.01996996514886773; 0; 0; 0; 0; 0; 0; 2.1914993049493301; 0.088570718482084379; 0.11405602348659656; 0.25331638053451072; -2.0565643862409408; 0.340809679901312; 0; 0; 0.048342313738239585];
            B.p = 9;
            B.q = 8;
        end

    end
end
