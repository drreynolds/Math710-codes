classdef Implicit_LMM < handle
    % Implicit_LMM.m
    %
    % Fixed-stepsize implicit linear multistep class implementation file.
    %
    % Also contains functions to return Adams-Moulton and BDF LMM coefficients
    % of orders 1-4.
    %
    % Class to perform fixed-stepsize time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using an implicit linear multistep (LMM) time stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize implicit linear multistep class
    %
    % The four required arguments when constructing an implicit linear
    % multistep object are a function for the IVP right-hand side, an
    % implicit solver to use, and the LMM coefficients:
    %     f = ODE RHS function with calling syntax f(t,y).
    %     sol = algebraic solver object to use [ImplicitSolver]
    %     alpha = LMM coefficients on previous solution values
    %     beta = LMM coefficients on previous RHS values
    %     h = (optional) input with stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.
    % Note that the LMM has the form:
    %    \sum_{j=0}^{k-1} \alpha_j y_{n+1-j} = h\sum_{j=0}^{k-1} \beta_j f_{n+1-j},
    % for computing each internal step, for an ODE IVP of the form
    %    y' = f(t,y), t in tspan,
    %    y(t0) = y0.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        f, sol, alpha, beta, k
        h = 0.0
        steps = 0
        yprev, fprev, data
    end

    methods
        function self = Implicit_LMM(f, sol, alpha, beta, h)
            if nargin < 4
                error('Implicit_LMM requires f, sol, alpha, and beta.');
            end

            % Store the RHS, nonlinear solver, and LMM coefficient vectors.
            self.f = f;
            self.sol = sol;
            self.alpha = alpha(:);
            self.beta = beta(:);
            % optional inputs
            if nargin >= 5 && ~isempty(h), self.h = h; end
            self.k = numel(self.alpha);

            % The leading solution coefficient must be invertible for implicit update.
            if abs(self.alpha(1)) == 0
                error('Implicit_LMM: alpha(1) must be nonzero');
            end
            if numel(self.beta) ~= self.k
                error('Implicit_LMM: alpha and beta must have the same length');
            end
        end

        function [t, success] = implicit_lmm_step(self, t, args)
            % Usage: t, success = implicit_lmm_step(t, args)
            %
            % Utility routine to take a single implicit LMM time step,
            % where the inputs `t` is overwritten by the updated value.
            % args is used for optional parameters of the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 3
                args = {};
            end
            if ~iscell(args)
                error('Implicit_LMM: args must be a cell array.');
            end

            t = t + self.h;
            % Collect all known history terms on the right-hand side of the LMM equation.
            self.data = (self.h * self.beta(2) / self.alpha(1)) * self.fprev{end} ...
                - (self.alpha(2) / self.alpha(1)) * self.yprev{end};
            for i = 3:self.k
                self.data = self.data + (self.h * self.beta(i) / self.alpha(1)) * self.fprev{end-i+2} ...
                    - (self.alpha(i) / self.alpha(1)) * self.yprev{end-i+2};
            end

            gamma = self.h * self.beta(1) / self.alpha(1);
            % Define the residual for the new unknown value y_{n+1}.
            F = @(ynew) ynew(:) - self.data(:) - gamma * self.f(t, ynew(:), args{:});
            % Tell the Newton solver to use I - gamma*J for this implicit LMM step.
            self.sol.setup_linear_solver(t, -gamma, args);

            % Solve the nonlinear residual equation for the new solution.
            [y, ~, success] = self.sol.solve(F, self.yprev{end});
            if ~success
                return;
            end

            % Advance the history queues by dropping the oldest entry and appending the new one.
            self.yprev = [self.yprev(2:end), {y}];
            self.fprev = [self.fprev(2:end), {self.f(t, y, args{:})}];
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
            n = self.steps;
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The fixed-step implicit linear multistep evolution routine.
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
                error('Implicit_LMM:Evolve args must be a cell array.');
            end

            % update stored stepsize when requested by the caller
            if h ~= 0.0
                self.h = h;
            end
            % require a nonzero stepsize before evolving
            if self.h == 0.0
                error('Implicit_LMM:Evolve called without specifying a nonzero step size');
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
                error('Implicit_LMM:Evolve received insufficient initial conditions');
            end

            nout = numel(tspan);
            m = size(y0,2);
            Y = zeros(nout, m);
            Y(1,:) = y0(end,:);

            % Initialize the solution and RHS history queues from the supplied startup values.
            self.data = y0(end,:).';
            self.yprev = cell(1, self.k-1);
            self.fprev = cell(1, self.k-1);
            for i = 1:(self.k-1)
                yi = y0(i,:).';
                ti = tspan(1) - (self.k-1-i)*self.h;
                self.yprev{i} = yi;
                self.fprev{i} = self.f(ti, yi, args{:});
            end

            % iterate over output times, filling the solution history
            for iout = 2:nout
                % The divisibility check above allows this rounded internal step count.
                N = round((tspan(iout)-tspan(iout-1))/self.h);
                t = tspan(iout-1);
                % March internally until the next requested output time is reached.
                for n = 1:N %#ok<NASGU>
                    [t, success] = self.implicit_lmm_step(t, args);
                    if ~success
                        fprintf('implicit_lmm error in time step at t = %g\n', t);
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
        function [alpha, beta, p] = AdamsMoulton1()
            % Usage: alphas, betas, p = AdamsMoulton1()
            %
            % Utility routine to return the 1st order Adams Moulton LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous RHS values
            %          p holds the LMM method order

            alpha = [1; -1];
            beta = [1; 0];
            p = 1;
        end

        function [alpha, beta, p] = AdamsMoulton2()
            % Usage: alphas, betas, p = AdamsMoulton2()
            %
            % Utility routine to return the 2nd order Adams Moulton LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous RHS values
            %          p holds the LMM method order

            alpha = [1; -1];
            beta = [0.5; 0.5];
            p = 2;
        end

        function [alpha, beta, p] = AdamsMoulton3()
            % Usage: alphas, betas, p = AdamsMoulton3()
            %
            % Utility routine to return the 3rd order Adams Moulton LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous RHS values
            %          p holds the LMM method order

            alpha = [1; -1; 0];
            beta = [5/12; 8/12; -1/12];
            p = 3;
        end

        function [alpha, beta, p] = AdamsMoulton4()
            % Usage: alphas, betas, p = AdamsMoulton4()
            %
            % Utility routine to return the 4th order Adams Moulton LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous RHS values
            %          p holds the LMM method order

            alpha = [1; -1; 0; 0];
            beta = [9/24; 19/24; -5/24; 1/24];
            p = 4;
        end

        function [alpha, beta, p] = BDF1()
            % Usage: alphas, betas, p = BDF1()
            %
            % Utility routine to return the 1st order BDF LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous RHS values
            %          p holds the LMM method order

            alpha = [1; -1];
            beta = [1; 0];
            p = 1;
        end

        function [alpha, beta, p] = BDF2()
            % Usage: alphas, betas, p = BDF2()
            %
            % Utility routine to return the 2nd order BDF LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous RHS values
            %          p holds the LMM method order

            alpha = [1; -4/3; 1/3];
            beta = [2/3; 0; 0];
            p = 2;
        end

        function [alpha, beta, p] = BDF3()
            % Usage: alphas, betas, p = BDF3()
            %
            % Utility routine to return the 3rd order BDF LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous RHS values
            %          p holds the LMM method order

            alpha = [1; -18/11; 9/11; -2/11];
            beta = [6/11; 0; 0; 0];
            p = 3;
        end

        function [alpha, beta, p] = BDF4()
            % Usage: alphas, betas, p = BDF4()
            %
            % Utility routine to return the 4th order BDF LMM coefficients.
            %
            % Outputs: alphas holds the LMM coefficients on previous solution values
            %          betas holds the LMM coefficients on previous RHS values
            %          p holds the LMM method order

            alpha = [1; -48/25; 36/25; -16/25; 3/25];
            beta = [12/25; 0; 0; 0; 0];
            p = 4;
        end
    end
end
