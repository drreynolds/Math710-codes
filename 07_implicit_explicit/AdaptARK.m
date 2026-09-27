classdef AdaptARK < handle
    % AdaptARK.m
    %
    % Adaptive-stepsize implicit-explicit additive Runge--Kutta solver class
    % implementation file.
    %
    % Also contains functions to return specific embedded ARK Butcher table pairs.
    % The individual explicit and implicit embedded tables are stored in
    % AdaptERK.m and AdaptDIRK.m; the routines at the end of this file assemble
    % them into compatible pairs.
    %
    % Class to perform adaptive stepsize time evolution of the IVP
    %      y' = fE(t,y) + fI(t,y),  t in [t0, Tf],  y(t0) = y0
    % using an embedded implicit-explicit additive Runge--Kutta (ARK) time
    % stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        fE, fI, sol, AE, bE, cE, dE, AI, bI, cI, dI, minpq, s
        rtol = 1e-3, atol = 1e-14, maxit = 1e6
        bias = 1.0, growth = 50.0, safety = 0.85, hmin = 10*eps, hE = inf
        ONEMSM = 1.0 - sqrt(eps), ONEPSM = 1.0 + sqrt(eps)
        fails = 0, steps = 0, nsol = 0, error_norm = 0.0, h = 0.0
        save_step_hist = false
        w, yerr, step_hist, z, yt, data, kE, kI
    end

    methods
        function self = AdaptARK(fE, fI, yTemplate, sol, BE, BI, rtol, atol, maxit, bias, growth, safety, hmin, hE, save_step_hist)
            % Adaptive implicit-explicit additive Runge--Kutta class
            %
            % The six required arguments when constructing an AdaptARK object are
            % functions for the explicit and implicit portions of the IVP right-hand
            % side, an initial solution vector, an implicit solver to use, and explicit
            % and implicit embedded Butcher tables:
            %     fE = explicit ODE RHS function with calling syntax fE(t,y).
            %     fI = implicit ODE RHS function with calling syntax fI(t,y).
            %     y = MATLAB array with m entries.
            %     sol = algebraic solver object to use [ImplicitSolver]
            %     BE = explicit embedded Runge--Kutta Butcher table.
            %     BI = diagonally-implicit embedded Runge--Kutta Butcher table.
            %     hE = (optional) known stability step-size limit for fE.
            if nargin < 6
                error('AdaptARK requires fE, fI, a solution template, an implicit solver, and explicit and implicit embedded Butcher tables.');
            end

            y = yTemplate(:);

            % required inputs
            self.fE = fE;
            self.fI = fI;
            self.sol = sol;
            self.AE = BE.A;
            self.bE = BE.b(:);
            self.cE = BE.c(:);
            self.dE = BE.d(:);
            self.AI = BI.A;
            self.bI = BI.b(:);
            self.cI = BI.c(:);
            self.dI = BI.d(:);
            self.minpq = min([BE.p, BE.q, BI.p, BI.q]);

            % optional inputs
            if nargin >= 7 && ~isempty(rtol), self.rtol = rtol; end
            if nargin >= 8 && ~isempty(atol), self.atol = atol; end
            if nargin >= 9 && ~isempty(maxit), self.maxit = maxit; end
            if nargin >= 10 && ~isempty(bias), self.bias = bias; end
            if nargin >= 11 && ~isempty(growth), self.growth = growth; end
            if nargin >= 12 && ~isempty(safety), self.safety = safety; end
            if nargin >= 13 && ~isempty(hmin), self.hmin = hmin; end
            if nargin >= 14 && ~isempty(hE), self.hE = hE; end
            if nargin >= 15 && ~isempty(save_step_hist), self.save_step_hist = save_step_hist; end
            if isscalar(self.atol)
                self.atol = ones(numel(y),1)*self.atol;
            else
                self.atol = self.atol(:);
            end

            % internal data
            self.w = ones(numel(y),1);
            self.yerr = zeros(numel(y),1);
            self.step_hist = struct('t', [], 'h', [], 'err', []);
            self.z = zeros(numel(y),1);
            self.yt = zeros(numel(y),1);
            self.data = zeros(numel(y),1);
            self.s = numel(self.bE);
            self.kE = zeros(self.s, numel(y));
            self.kI = zeros(self.s, numel(y));

            % check for legal tables (including matching numbers of stages)
            if numel(self.cE) ~= self.s || numel(self.cI) ~= self.s || ...
                    numel(self.bI) ~= self.s || numel(self.dE) ~= self.s || ...
                    numel(self.dI) ~= self.s || size(self.AE,1) ~= self.s || ...
                    size(self.AE,2) ~= self.s || size(self.AI,1) ~= self.s || ...
                    size(self.AI,2) ~= self.s || ...
                    norm(self.bE-self.dE) < 1e-14 || ...
                    norm(self.bI-self.dI) < 1e-14 || ...
                    norm(self.AE - tril(self.AE,-1), inf) > 1e-14 || ...
                    norm(self.AI - tril(self.AI,0), inf) > 1e-14
                error('AdaptARK: incompatible Butcher tables supplied');
            end
            if BE.p ~= BI.p || BE.q ~= BI.q
                error('AdaptARK: Butcher table orders are incompatible');
            end
            if self.hE <= 0.0 || self.hE < self.hmin
                error('AdaptARK: illegal explicit stability step-size limit');
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
            % Utility routine to take a single implicit-explicit ARK time step,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS functions.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 4
                args = {};
            end
            if ~iscell(args)
                error('AdaptARK: args must be a cell array.');
            end

            y = y(:);
            % loop over stages, computing RHS vectors
            for i = 1:self.s

                % construct "data" for this stage solve
                self.data = y;
                for j = 1:(i-1)
                    self.data = self.data + self.h * (self.AE(i,j) * self.kE(j,:).' ...
                                                      + self.AI(i,j) * self.kI(j,:).');
                end

                % solve the implicit stage (or copy the data for an explicit stage)
                tstageI = t + self.h*self.cI(i);
                if abs(self.AI(i,i)) > 1e-14

                    % construct implicit residual and Jacobian solver for this stage
                    F = @(zcur) zcur(:) - self.data(:) - self.h * self.AI(i,i) * self.fI(tstageI, zcur(:), args{:});
                    self.sol.setup_linear_solver(tstageI, -self.h * self.AI(i,i), args);

                    % perform implicit solve, and return on solver failure
                    [self.z, ~, success] = self.sol.solve(F, y);
                    self.nsol = self.nsol + 1;
                    if ~success
                        return;
                    end
                else
                    self.z = self.data;
                end

                % store both RHS vectors at this stage
                self.kE(i,:) = self.fE(t + self.h*self.cE(i), self.z, args{:}).';
                self.kI(i,:) = self.fI(tstageI, self.z, args{:}).';
            end

            % update time step solution
            for i = 1:self.s
                y = y + self.h * (self.bE(i) * self.kE(i,:).' + self.bI(i) * self.kI(i,:).');
            end

            % compute error estimate (and norm), and return
            self.yerr(:) = 0.0;
            for i = 1:self.s
                self.yerr = self.yerr + self.h * ((self.bE(i) - self.dE(i)) * self.kE(i,:).' ...
                                                  + (self.bI(i) - self.dI(i)) * self.kI(i,:).');
            end
            self.error_norm = max(norm(self.yerr.*self.w, inf), 1e-8);
            success = true;
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The adaptive ARK time step evolution routine
            %
            % Inputs:  tspan holds the current time interval, [t0, tf], including any
            %              intermediate times when the solution is desired, i.e.
            %              [t0, t1, ..., tf]
            %          y holds the initial condition, y(t0)
            %          h optionally holds the requested initial step size (if it is
            %              not provided then an initial step size will be estimated);
            %              this is limited by the explicit stability step-size limit hE
            %          args holds optional equation parameters used when evaluating
            %              the RHS functions.
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
                error('AdaptARK:Evolve args must be a cell array.');
            end

            % store input step size, limited by the explicit stability limit
            self.h = h;
            if self.h ~= 0.0
                self.h = min(self.h, self.hE);
            end

            % store sizes
            y = y0(:);
            tspan = tspan(:);
            m = numel(y);
            N = numel(tspan)-1;

            % initialize output
            Y = zeros(N+1, m);
            Y(1,:) = y.';

            % set current time value
            t = tspan(1);

            % check for legal time span
            for n = 1:N
                if tspan(n+1) < tspan(n)
                    error('AdaptARK:Evolve illegal tspan');
                end
            end

            % initialize error weight vector, and check for legal tolerances
            self.w = self.error_weight(y);

            % estimate initial step size if not provided by user
            if self.h == 0.0

                % get ||y'(t0)||
                fn = self.fE(t, y, args{:}) + self.fI(t, y, args{:});

                % estimate initial h value via linearization, safety factor, and explicit stability limit
                self.error_norm = max(norm(fn(:).*self.w, inf), 1e-8);
                self.h = min(max(self.hmin, self.safety / self.error_norm), self.hE);
            end

            % iterate over output times
            for iout = 2:(N+1)

                % loop over internal steps to reach desired output time
                while (tspan(iout)-t) > sqrt(eps*tspan(iout))

                    % enforce maxit -- if we've exceeded attempts, return with failure
                    if (self.steps + self.fails) > self.maxit
                        fprintf('AdaptARK: reached maximum iterations, returning with failure\n');
                        success = false;
                        return;
                    end

                    % bound internal time step to not exceed the explicit stability limit or next output time
                    self.h = min([self.h, self.hE, tspan(iout)-t]);

                    % reset temporary solution to current solution, and take ARK step
                    self.yt = y;
                    [~, self.yt, success] = self.step(t, self.yt, args);
                    if ~success
                        fprintf('AdaptARK::Evolve error in time step at t = %g\n', t);
                        return;
                    end

                    % estimate step size growth/reduction factor based on error estimate
                    eta = self.safety * self.error_norm^(-1.0/(self.minpq+1));  % step size growth factor
                    eta = min(eta, self.growth);                                % limit maximum growth

                    % store step size in history if requested
                    if self.save_step_hist
                        self.step_hist.t(end+1,1) = t;
                        self.step_hist.h(end+1,1) = self.h;
                        self.step_hist.err(end+1,1) = self.error_norm;
                    end

                    % check error
                    if self.error_norm < self.ONEPSM  % successful step

                        % update current time, solution, error weights, work counter, and upcoming stepsize
                        t = t + self.h;
                        y = self.yt;
                        self.w = self.error_weight(y);
                        self.steps = self.steps + 1;
                        self.h = min(self.h * eta, self.hE);

                    else                              % failed step
                        self.fails = self.fails + 1;

                        % adjust step size, enforcing minimum and returning with failure if needed
                        if self.h > self.hmin                             % failure, but reduction possible
                            self.h = max(self.h * eta, self.hmin);
                        else                                              % failed with no reduction possible
                            fprintf('AdaptARK: error test failed at h=hmin, returning with failure\n');
                            success = false;
                            return;
                        end
                    end
                end

                % store current results in output arrays
                Y(iout,:) = y.';
            end

            % return with successful solution
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

        function set_hE(self, hE)
            % Resets the explicit stability step-size limit
            if nargin < 2
                hE = inf;
            end
            self.hE = hE;
        end

        function update_rhs(self, fE, fI)
            % Updates the RHS functions (cannot change vector dimensions)
            self.fE = fE;
            self.fI = fI;
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
            % Returns the accumulated number of steps
            out = self.steps;
        end

        function out = get_num_solves(self)
            % Returns the accumulated number of implicit solves
            out = self.nsol;
        end

        function out = get_current_step(self)
            % Returns the current internal step size
            out = self.h;
        end

        function out = get_step_history(self)
            % Returns the current step size history
            out = self.step_hist;
        end

        function reset(self)
            % Resets the accumulated number of steps
            self.fails = 0;
            self.error_norm = 0.0;
            self.nsol = 0;
            self.steps = 0;
            self.step_hist = struct('t', [], 'h', [], 'err', []);
        end
    end

    % embedded ARK Butcher table pair routines
    methods (Static)
        function [BE, BI] = ARS222()
            % Usage: [BE, BI] = AdaptARK.ARS222()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to the ARS(2,2,2) method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.Ascher222ERK();
            BI = AdaptDIRK.Ascher222SDIRK();
        end

        function [BE, BI] = SSP32()
            % Usage: [BE, BI] = AdaptARK.SSP32()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to the SSP(3,2) method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.SSP32ERK();
            BI = AdaptDIRK.SSP32DIRK();
        end

        function [BE, BI] = ARK232()
            % Usage: [BE, BI] = AdaptARK.ARK232()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to the ARK(2,3,2) method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.ARK232ERK();
            BI = AdaptDIRK.ARK232SDIRK();
        end

        function [BE, BI] = SSP2332Lspum()
            % Usage: [BE, BI] = AdaptARK.SSP2332Lspum()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to the SSP2(3,3,2)-lspum method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.SSP2332LspumERK();
            BI = AdaptDIRK.SSP2332LspumSDIRK();
        end

        function [BE, BI] = GiraldoARK2()
            % Usage: [BE, BI] = AdaptARK.GiraldoARK2()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to Giraldo's ARK2 method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.GiraldoARK2ERK();
            BI = AdaptDIRK.GiraldoARK2ESDIRK();
        end

        function [BE, BI] = ARK324L2SA()
            % Usage: [BE, BI] = AdaptARK.ARK324L2SA()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to Kennedy & Carpenter's ARK3(2)4L[2]SA method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.ARK324L2SAERK();
            BI = AdaptDIRK.ARK324L2SAESDIRK();
        end

        function [BE, BI] = SSP43()
            % Usage: [BE, BI] = AdaptARK.SSP43()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to the SSP(4,3) method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.SSP43ERK();
            BI = AdaptDIRK.SSP43ESDIRK();
        end

        function [BE, BI] = ARK436L2SA()
            % Usage: [BE, BI] = AdaptARK.ARK436L2SA()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to Kennedy & Carpenter's ARK4(3)6L[2]SA method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.ARK436L2SAERK();
            BI = AdaptDIRK.ARK436L2SAESDIRK();
        end

        function [BE, BI] = ARK437L2SA()
            % Usage: [BE, BI] = AdaptARK.ARK437L2SA()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to Kennedy & Carpenter's ARK4(3)7L[2]SA method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.ARK437L2SAERK();
            BI = AdaptDIRK.ARK437L2SAESDIRK();
        end

        function [BE, BI] = ARK548L2SA()
            % Usage: [BE, BI] = AdaptARK.ARK548L2SA()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to Kennedy & Carpenter's ARK5(4)8L[2]SA method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.ARK548L2SAERK();
            BI = AdaptDIRK.ARK548L2SAESDIRK();
        end

        function [BE, BI] = ARK548L2SAb()
            % Usage: [BE, BI] = AdaptARK.ARK548L2SAb()
            %
            % Utility routine to return the embedded ARK Butcher table pair corresponding
            % to Kennedy & Carpenter's ARK5(4)8L[2]SAb method.
            %
            % Outputs: BE holds the explicit embedded Butcher table
            %          BI holds the implicit embedded Butcher table
            AdaptARK.add_table_paths();
            BE = AdaptERK.ARK548L2SAbERK();
            BI = AdaptDIRK.ARK548L2SAbESDIRK();
        end
    end

    % path utility routine for the Butcher table pair routines
    methods (Static, Access = private)
        function add_table_paths()
            % Usage: AdaptARK.add_table_paths()
            %
            % Utility routine to add the folders holding ERK.m, DIRK.m, AdaptERK.m
            % and AdaptDIRK.m to the MATLAB path, so that the Butcher table pair
            % routines above can be called from any folder.
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root, '04_explicit_one_step'));
            addpath(fullfile(root, '05_implicit_one_step'));
        end
    end
end
