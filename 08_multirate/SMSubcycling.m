classdef SMSubcycling < handle
    % SMSubcycling.m
    %
    % Fixed-stepsize Strang-Marchuk based subcycling time stepper class implementation file.
    %
    % Class to perform fixed-stepsize, but subcycled, time evolution of the IVP
    %      y' = fs(t,y) + ff(t,y),  t in [t0, Tf],  y(t0) = y0
    % using a fixed step size ERK method for the sub-IVP
    %      y' = fs(t,y),  y(tk) = yk,  t in [t_k, t_k + h]
    % to compute ytmp at t_k+h, and any object that supports the "Evolve" routine
    % for the sub-IVP
    %      y' = ff(t,y),  y(tk) = ytmp,  t in [t_k, t_k + h].
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize Strang-Marchuk subcycling time stepper class
    %
    % The three required arguments when constructing a SMSubcycling object
    % are a function for the "slow" IVP right-hand side, an explicit Runge--Kutta
    % Butcher table to use for the slow IVP, and a solver for the "fast" sub-IVP:
    %     fs = ODE RHS function with calling syntax fs(t,y).
    %     B = Explicit Runge--Kutta Butcher table.
    %     FastSolver = object that implements the "Evolve" method for the fast IVP.
    %     h = (optional) input with requested stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        fs, A, b, c, FastSolver, s
        H = 0.0
        steps = 0
        nrhs = 0
        k, z
    end

    methods
        function self = SMSubcycling(fs, B, FastSolver, H)
            if nargin < 3
                error('SMSubcycling requires fs, a Butcher table, and a fast solver.');
            end

            % Store the slow RHS, slow RK table, and fast subsolver.
            self.fs = fs;
            self.A = B.A;
            self.b = B.b(:);
            self.c = B.c(:);
            self.FastSolver = FastSolver;
            if nargin >= 4 && ~isempty(H), self.H = H; end
            self.s = numel(self.c);

            % The slow method is explicit, so A may only have entries below the diagonal.
            if numel(self.b) ~= self.s || size(self.A,1) ~= self.s || size(self.A,2) ~= self.s || ...
                    norm(self.A - tril(self.A,-1), inf) > 1e-14
                error('SMSubcycling: incompatible Butcher table supplied');
            end
        end

        function [t, y, success] = step(self, t, y, H, args)
            % Usage: t, y, success = step(t, y, H, args)
            %
            % Utility routine to take a single Strang-Marchuk subcycled time step of size H,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 5
                args = {};
            end
            if ~iscell(args)
                error('SMSubcycling: args must be a cell array.');
            end

            y = y(:);
            % Apply the fast subsolver over the first half step.
            [ytmp, success] = self.FastSolver.Evolve([t; t+H/2], y, [], args);
            if ~success
                return;
            end
            y = ytmp(end,:).';

            % Apply the slow explicit RK method over the full macro step.
            self.k(1,:) = self.fs(t, y, args{:}).';
            self.nrhs = self.nrhs + 1;
            for i = 2:self.s
                self.z = y;
                for j = 1:(i-1)
                    self.z = self.z + H * self.A(i,j) * self.k(j,:).';
                end
                self.k(i,:) = self.fs(t + self.c(i)*H, self.z, args{:}).';
                self.nrhs = self.nrhs + 1;
            end

            % Combine slow stage data before the second fast half step.
            for i = 1:self.s
                y = y + H * self.b(i) * self.k(i,:).';
            end

            % Finish with the fast subsolver over the second half step.
            [ytmp, success] = self.FastSolver.Evolve([t+H/2; t+H], y, [], args);
            if ~success
                return;
            end

            y = ytmp(end,:).';
            t = t + H;
            self.steps = self.steps + 1;
        end

        function update_rhs(self, fs)
            % Updates the slow RHS function (cannot change vector dimensions)
            self.fs = fs;
        end

        function reset(self)
            % Resets the accumulated number of steps

            % reset accumulated statistics
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
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The fixed-step Strang-Marchuk subcycled evolution routine
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
                error('SMSubcycling:Evolve args must be a cell array.');
            end

            if H ~= 0.0
                self.H = H;
            end
            if self.H == 0.0
                error('SMSubcycling:Evolve called without specifying a nonzero step size');
            end

            % Initialize output storage, with the first row holding the initial condition.
            tspan = tspan(:);
            y = y0(:);
            nout = numel(tspan);
            m = numel(y);
            Y = zeros(nout, m);
            Y(1,:) = y.';

            % Allocate slow-stage work arrays now that the solution dimension is known.
            self.k = zeros(self.s, m);
            self.z = y;

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
                        fprintf('SMSubcycling error in time step at t = %g\n', t);
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
