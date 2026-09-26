classdef ImEx_LMM < handle
    % ImEx_LMM.m
    %
    % Fixed-stepsize implicit-explicit linear multistep class implementation file.
    %
    % Also contains functions to return ImEx-LMM coefficients
    % of order 2.
    %
    % Class to perform fixed-stepsize time evolution of the IVP
    %      y' = fe(t,y) + fi(t,y),  t in [t0, Tf],  y(t0) = y0
    % using an implicit-explicit linear multistep (LMM) time stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize implicit-explicit linear multistep class
    %
    % The six required arguments when constructing an ImEx linear
    % multistep object are functions for the explicit and implicit
    % portions of the IVP right-hand side, an implicit solver to use,
    % and the LMM coefficients:
    %     fe = ODE RHS function for the explicit portion, with calling syntax fe(t,y).
    %     fi = ODE RHS function for the implicit portion, with calling syntax fi(t,y).
    %     sol = algebraic solver object to use [ImplicitSolver]; this should
    %           be constructed using the Jacobian of fi
    %     alpha = LMM coefficients on previous solution values
    %     beta = LMM coefficients on previous implicit RHS values
    %     gamma = LMM coefficients on previous explicit RHS values (gamma(1) must be 0)
    %     h = (optional) input with stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.
    % Note that the ImEx LMM has the form:
    %    \sum_{j=0}^{k-1} \alpha_j y_{n+1-j} = h\sum_{j=0}^{k-1} \beta_j fi_{n+1-j}
    %                                        + h\sum_{j=1}^{k-1} \gamma_j fe_{n+1-j},
    % for computing each internal step, for an ODE IVP of the form
    %    y' = fe(t,y) + fi(t,y), t in tspan,
    %    y(t0) = y0.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        fe, fi, sol, alpha, beta, gamma, k
        h = 0.0
        steps = 0
        yprev, feprev, fiprev, data
    end

    methods
        function self = ImEx_LMM(fe, fi, sol, alpha, beta, gamma, h)
            if nargin < 6
                error('ImEx_LMM requires fe, fi, sol, alpha, beta, and gamma.');
            end

            % Store the RHS functions, nonlinear solver, and LMM coefficient vectors.
            self.fe = fe;
            self.fi = fi;
            self.sol = sol;
            self.alpha = alpha(:);
            self.beta = beta(:);
            self.gamma = gamma(:);
            % optional inputs
            if nargin >= 7 && ~isempty(h), self.h = h; end
            self.k = numel(self.alpha);

            % The leading solution coefficient must be invertible for implicit update.
            if abs(self.alpha(1)) == 0
                error('ImEx_LMM: alpha(1) must be nonzero');
            end
            if abs(self.beta(1)) == 0
                error('ImEx_LMM: beta(1) must be nonzero');
            end
            if self.gamma(1) ~= 0
                error('ImEx_LMM: gamma(1) must be zero');
            end
            if numel(self.beta) ~= self.k
                error('ImEx_LMM: alpha and beta must have the same length');
            end
            if numel(self.gamma) ~= self.k
                error('ImEx_LMM: alpha and gamma must have the same length');
            end
        end

        function [t, success] = imex_lmm_step(self, t, args)
            % Usage: t, success = imex_lmm_step(t, args)
            %
            % Utility routine to take a single ImEx LMM time step,
            % where the input `t` is overwritten by the updated value.
            % args is used for optional parameters of the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 3
                args = {};
            end
            if ~iscell(args)
                error('ImEx_LMM: args must be a cell array.');
            end

            t = t + self.h;
            % Collect all known history terms on the right-hand side of the LMM equation.
            self.data = (self.h * self.beta(2) / self.alpha(1)) * self.fiprev{end} ...
                + (self.h * self.gamma(2) / self.alpha(1)) * self.feprev{end} ...
                - (self.alpha(2) / self.alpha(1)) * self.yprev{end};
            for i = 3:self.k
                self.data = self.data + (self.h * self.beta(i) / self.alpha(1)) * self.fiprev{end-i+2} ...
                    + (self.h * self.gamma(i) / self.alpha(1)) * self.feprev{end-i+2} ...
                    - (self.alpha(i) / self.alpha(1)) * self.yprev{end-i+2};
            end

            % Define the residual for the new unknown value y_{n+1}.
            F = @(ynew) ynew(:) - self.data(:) - (self.h * self.beta(1) / self.alpha(1)) ...
                * self.fi(t, ynew(:), args{:});
            % Tell the Newton solver to use I - (h*beta_0/alpha_0)*J for this ImEx LMM step.
            self.sol.setup_linear_solver(t, -self.h * self.beta(1) / self.alpha(1), args);

            % Solve the nonlinear residual equation for the new solution.
            [y, ~, success] = self.sol.solve(F, self.yprev{end});
            if ~success
                return;
            end

            % Advance the history queues by dropping the oldest entry and appending the new one.
            self.yprev = [self.yprev(2:end), {y}];
            self.feprev = [self.feprev(2:end), {self.fe(t, y, args{:})}];
            self.fiprev = [self.fiprev(2:end), {self.fi(t, y, args{:})}];
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

        function n = get_num_solves(self)
            % Returns the accumulated number of implicit solves
            n = self.steps;
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The fixed-step implicit-explicit linear multistep evolution routine.
            %
            % Note: this requires that y0 has separate rows containing
            % sufficiently accurate "initial" values for all previous LMM steps.
            %
            % Inputs:  tspan holds the current time interval, [t0, tf], including any
            %              intermediate times when the solution is desired, i.e.
            %              [t0, t1, ..., tf]
            %          y0 holds the initial conditions [nd-array, shape(k-1,n)],
            %              sorted as [y0(t0-(k-2)*h), ... y0(t0-h), y0(t0)]
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
                error('ImEx_LMM:Evolve args must be a cell array.');
            end

            % update stored stepsize when requested by the caller
            if h ~= 0.0
                self.h = h;
            end
            % require a nonzero stepsize before evolving
            if self.h == 0.0
                error('ImEx_LMM:Evolve called without specifying a nonzero step size');
            end

            % Verify that each output interval can be reached by an integer number of steps.
            tspan = tspan(:);
            for n = 1:(numel(tspan)-1)
                hn = tspan(n+1)-tspan(n);
                if abs(round(hn/self.h) - (hn/self.h)) > 100*sqrt(eps)*abs(self.h)
                    error('input values in tspan (%e,%e) are not separated by a multiple of h = %e', tspan(n), tspan(n+1), self.h);
                end
            end

            % A k-step method needs k-1 previous solution values to start.
            if size(y0,1) < (self.k-1)
                error('ImEx_LMM:Evolve received insufficient initial conditions');
            end

            nout = numel(tspan);
            m = size(y0,2);
            Y = zeros(nout, m);
            Y(1,:) = y0(end,:);

            % Initialize the solution and RHS history queues from the supplied startup values.
            self.data = y0(end,:).';
            self.yprev = cell(1, self.k-1);
            self.feprev = cell(1, self.k-1);
            self.fiprev = cell(1, self.k-1);
            for i = 1:(self.k-1)
                yi = y0(i,:).';
                ti = tspan(1) - (self.k-1-i)*self.h;
                self.yprev{i} = yi;
                self.feprev{i} = self.fe(ti, yi, args{:});
                self.fiprev{i} = self.fi(ti, yi, args{:});
            end

            % iterate over output times, filling the solution history
            for iout = 2:nout
                % The divisibility check above allows this rounded internal step count.
                N = round((tspan(iout)-tspan(iout-1))/self.h);
                t = tspan(iout-1);
                % March internally until the next requested output time is reached.
                for n = 1:N %#ok<NASGU>
                    [t, success] = self.imex_lmm_step(t, args);
                    if ~success
                        fprintf('imex_lmm error in time step at t = %g\n', t);
                        return;
                    end
                end
                % Store the newest history value as the output solution.
                Y(iout,:) = self.yprev{end}.';
            end

            success = true;
        end
    end

    methods (Static)
        function [alpha, beta, gamma, p] = SBDF2()
            % Usage: alphas, betas, gammas, p = SBDF2()
            %
            % Utility routine to return the 2nd order semi-implicit BDF (SBDF-2)
            % ImEx LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous implicit RHS values
            %          gammas holds the LMM coefficients on previous explicit RHS values
            %          p holds the LMM method order

            alpha = [1; -4/3; 1/3];
            beta = [2/3; 0; 0];
            gamma = [0; 4/3; -2/3];
            p = 2;
        end

        function [alpha, beta, gamma, p] = CNAB()
            % Usage: alphas, betas, gammas, p = CNAB()
            %
            % Utility routine to return the 2nd order Crank-Nicolson, Adams-Bashforth
            % (CNAB) ImEx LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous implicit RHS values
            %          gammas holds the LMM coefficients on previous explicit RHS values
            %          p holds the LMM method order

            alpha = [1; -1; 0];
            beta = [0.5; 0.5; 0];
            gamma = [0; 1.5; -0.5];
            p = 2;
        end

        function [alpha, beta, gamma, p] = MCNAB()
            % Usage: alphas, betas, gammas, p = MCNAB()
            %
            % Utility routine to return the 2nd order modified Crank-Nicolson,
            % Adams-Bashforth (MCNAB) ImEx LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous implicit RHS values
            %          gammas holds the LMM coefficients on previous explicit RHS values
            %          p holds the LMM method order

            alpha = [1; -1; 0];
            beta = [9/16; 6/16; 1/16];
            gamma = [0; 1.5; -0.5];
            p = 2;
        end

        function [alpha, beta, gamma, p] = CNLF()
            % Usage: alphas, betas, gammas, p = CNLF()
            %
            % Utility routine to return the 2nd order Crank-Nicolson, leapfrog
            % (CNLF) ImEx LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous implicit RHS values
            %          gammas holds the LMM coefficients on previous explicit RHS values
            %          p holds the LMM method order

            alpha = [1; 0; -1];
            beta = [1; 0; 1];
            gamma = [0; 2; 0];
            p = 2;
        end
    end
end
