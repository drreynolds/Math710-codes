classdef MRI < handle
    % MRI.m
    %
    % Fixed-stepsize multirate infinitesimal (MRI) time stepper class implementation file.
    %
    % Class to perform fixed-stepsize MRI time evolution of the IVP
    %      y' = fs(t,y) + ff(t,y),  t in [t0, Tf],  y(t0) = y0
    % using a fixed step size explicit MRI method for the sub-IVP
    %      y' = fs(t,y),  y(tk) = yk,  t in [t_k, t_{k+1}],
    % and any object that supports the "Evolve" routine
    % for the sub-IVPs
    %      y' = ff(t,y) + r(t),  y(tk) = y_k,  t in [t_k, t_{k+1}].
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize MRI time stepper class
    %
    % The four required arguments when constructing a MRI object
    % are a function for the "slow" IVP right-hand side, an explicit Runge--Kutta
    % Butcher table to use for the slow IVP, and a solver for the "fast" sub-IVP:
    %     y = template for the solution vector.
    %     fs = slow ODE RHS function with calling syntax fs(t,y).
    %     ff = fast ODE RHS function with calling syntax fs(t,y).
    %     C = explicit MRI coupling table, with the following keys:
    %         'G' = 3D MATLAB array of coupling coefficients, with G[0,:,:] corresponding
    %               to the matrix of constant tendencies, G[1,:,:] the matrix of linear
    %               tendencies, etc.  Note: these must be strictly lower triangular.
    %         'c' = vector of slow stage abscissae.  Note: these must be sorted such that
    %               c[0] < c[1] < ... < c[end-1]
    %     FastSolver = object that implements the "Evolve" method for the fast IVP,
    %         and that support the "update_rhs" method to update the fast RHS function.
    %         This should support fast evolution intervals with different width, i.e., if
    %         using fixed substeps then there should be no restriction that the fast substep
    %         divide the interval evenly.
    %     H = (optional) input with requested stepsize to use for MRI time stepping.
    %         Note that this MUST be set either here or in the Evolve call.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        fs, ff, G, c, FastSolver, s, m
        H = 0.0
        steps = 0
        nrhs = 0
        Fs, deltac
    end

    methods
        function self = MRI(yTemplate, fs, ff, C, FastSolver, H)
            if nargin < 5
                error('MRI requires yTemplate, fs, ff, coupling table, and a fast solver.');
            end

            % Store the slow/fast split, coupling table, and fast subsolver.
            self.fs = fs;
            self.ff = ff;
            self.G = C.G;
            self.c = C.c(:);
            self.FastSolver = FastSolver;
            if nargin >= 6 && ~isempty(H), self.H = H; end
            self.s = numel(self.c);
            self.m = numel(yTemplate);
            self.Fs = zeros(self.m, self.s);

            % Coupling matrices must not depend on unavailable future stages.
            Gabssum = squeeze(sum(abs(self.G), 1));
            if norm(Gabssum - tril(Gabssum, -1), inf) > 1e-14
                error('MRI: incompatible coupling table supplied');
            end
            % Consecutive slow abscissae define the fast subinterval widths.
            self.deltac = diff(self.c);
            if any(self.deltac <= 0.0)
                error('MRI: coupling abscissae must be strictly increasing');
            end
        end

        function set_fast_rhs(self, tn, stage, H)
            % Usage: set_fast_rhs(tn, stage, H)
            %
            % Utility routine to update the RHS function in the fast solver based on the
            % time at the start of the current MRI step, and the current MRI stage index.

            ngammas = size(self.G, 1);
            tprev = tn + self.c(stage-1)*H;
            dc = self.deltac(stage-1);

            % Scale the coupling coefficients for the current fast subinterval.
            Gamma = zeros(self.s, ngammas);
            for col = 1:ngammas
                for row = 1:(stage-1)
                    Gamma(row,col) = self.G(col, stage, row) / dc;
                end
            end

            Fs = self.Fs;
            ff_orig = self.ff;
            % Replace the fast RHS with ff plus the MRI forcing from previous slow stages.
            self.FastSolver.update_rhs(@(tau, v, varargin) mriFastRhs( ...
                tau, v, Fs, Gamma, tprev, dc, H, ff_orig, varargin{:}));
        end

        function [t, y, success] = step(self, t, y, H, args)
            % Usage: t, y, success = step(t, y, H, args)
            %
            % Utility routine to take a single MRI time step of size H,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 5
                args = {};
            end
            if ~iscell(args)
                error('MRI: args must be a cell array.');
            end

            y = y(:);
            % Stage 1 is explicit, so store its slow RHS before any fast evolution.
            self.Fs(:,:) = 0.0;
            self.Fs(:,1) = self.fs(t, y, args{:});
            self.nrhs = self.nrhs + 1;

            for stage = 2:self.s
                % Evolve the fast subproblem with forcing built from previous slow stages.
                self.set_fast_rhs(t, stage, H);
                tstage = [t + self.c(stage-1)*H; t + self.c(stage)*H];
                [ytmp, success] = self.FastSolver.Evolve(tstage, y, [], args);
                if ~success
                    self.steps = self.steps + 1;
                    return;
                end

                % Store the new slow stage state and RHS for later coupling terms.
                y = ytmp(end,:).';
                self.Fs(:,stage) = self.fs(tstage(end), y, args{:});
                self.nrhs = self.nrhs + 1;
            end

            t = t + H;
            self.steps = self.steps + 1;
        end

        function update_rhs(self, fs)
            % Updates the slow RHS function (cannot change vector dimensions)
            self.fs = fs;
        end

        function reset(self)
            % Resets the accumulated number of steps

            self.steps = 0;
            self.nrhs = 0;
            if ismethod(self.FastSolver, 'reset')
                self.FastSolver.reset();
            end
        end

        function n = get_num_steps(self)
            % Returns the accumulated number of slow steps
            n = self.steps;
        end

        function n = get_num_rhs(self)
            % Returns the accumulated number of RHS evaluations
            n = self.nrhs;
        end

        function [Y, success] = Evolve(self, tspan, y0, H, args)
            % Usage: Y, success = Evolve(tspan, y0, H, args)
            %
            % The fixed-step MRI evolution routine
            %
            % Inputs:  tspan holds the current time interval, [t0, tf], including any
            %              intermediate times when the solution is desired, i.e.
            %              [t0, t1, ..., tf]
            %          y holds the initial condition, y(t0)
            %          H optionally holds the requested MRI step size (if it is not
            %              provided then the stored value will be used)
            %          args holds optional equation parameters used when evaluating
            %              the RHS.
            % Outputs: Y holds the computed solution at all tspan values,
            %              [y(t0), y(t1), ..., y(tf)]
            %          success = true if the solver traversed the interval,
            %              false if an integration step failed

            if nargin < 4 || isempty(H)
                H = 0.0;
            end
            if nargin < 5
                args = {};
            end
            if ~iscell(args)
                error('MRI:Evolve args must be a cell array.');
            end

            if H ~= 0.0
                self.H = H;
            end
            if self.H == 0.0
                error('MRI:Evolve called without specifying a nonzero step size');
            end

            % Initialize output storage, with the first row holding the initial condition.
            tspan = tspan(:);
            y = y0(:);
            nout = numel(tspan);
            m = numel(y);
            Y = zeros(nout, m);
            Y(1,:) = y.';

            % iterate over output times, filling the solution history
            for iout = 2:nout
                dt = tspan(iout) - tspan(iout-1);
                % Choose enough macro steps that no step exceeds the requested size.
                N = ceil(dt / self.H);
                if N < 1
                    N = 1;
                end
                Hcur = dt / N;
                t = tspan(iout-1);

                % March internally until the next requested output time is reached.
                for n = 1:N %#ok<NASGU>
                    [t, y, success] = self.step(t, y, Hcur, args);
                    if ~success
                        fprintf('MRI error in time step at t = %g\n', t);
                        return;
                    end
                end
                % Store the current solution as one row in the output history.
                Y(iout,:) = y.';
            end

            success = true;
        end
    end

    methods (Static)
        function C = MRIGARKERK22a()
            % Usage: C = MRIGARKERK22a()
            %
            % Returns a struct with the coupling coefficients and abscissae
            % for the explicit MRI-GARK-ERK22a method.

            c2 = 0.5;
            C.G = zeros(1, 3, 3);
            C.G(1,:,:) = [0, 0, 0; ...
                          c2, 0, 0; ...
                          -(2*c2*c2-2*c2+1)/(2*c2), 1/(2*c2), 0];
            C.c = [0; c2; 1];
        end

        function C = MRIGARKERK33a()
            % Usage: C = MRIGARKERK33a()
            %
            % Returns a struct with the coupling coefficients and abscissae
            % for the explicit MRI-GARK-ERK33a method.

            C.G = zeros(2, 4, 4);
            C.G(1,:,:) = [0, 0, 0, 0; ...
                          1/3, 0, 0, 0; ...
                          -1/3, 2/3, 0, 0; ...
                          0, -2/3, 1, 0];
            C.G(2,:,:) = [0, 0, 0, 0; ...
                          0, 0, 0, 0; ...
                          0, 0, 0, 0; ...
                          1/2, 0, -1/2, 0];
            C.c = [0; 1/3; 2/3; 1];
        end

        function C = MRIGARKERK45a()
            % Usage: C = MRIGARKERK45a()
            %
            % Returns a struct with the coupling coefficients and abscissae
            % for the explicit MRI-GARK-ERK45a method.

            C.G = zeros(2, 6, 6);
            C.G(1,:,:) = [0, 0, 0, 0, 0, 0; ...
                          1/5, 0, 0, 0, 0, 0; ...
                          -53/16, 281/80, 0, 0, 0, 0; ...
                          -36562993/71394880, 34903117/17848720, -88770499/71394880, 0, 0, 0; ...
                          -7631593/71394880, -166232021/35697440, 6068517/1519040, 8644289/8924360, 0, 0; ...
                          277061/303808, -209323/1139280, -1360217/1139280, -148789/56964, 147889/45120, 0];
            C.G(2,:,:) = [0, 0, 0, 0, 0, 0; ...
                          0, 0, 0, 0, 0, 0; ...
                          503/80, -503/80, 0, 0, 0, 0; ...
                          -1365537/35697440, 4963773/7139488, -1465833/2231090, 0, 0, 0; ...
                          66974357/35697440, 21445367/7139488, -3, -8388609/4462180, 0, 0; ...
                          -18227/7520, 2, 1, 5, -41933/7520, 0];
            C.c = [0; 1/5; 2/5; 3/5; 4/5; 1];
        end
    end
end

function val = mriFastRhs(tau, v, Fs, Gamma, tprev, dc, H, ff_orig, varargin)
    theta = (tau - tprev) / dc;
    thetapow = (theta/H).^(0:(size(Gamma,2)-1)).';
    val = Fs * (Gamma * thetapow) + ff_orig(tau, v, varargin{:});
end
