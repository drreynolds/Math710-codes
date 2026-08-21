classdef ImplicitSolver < handle
    % ImplicitSolver.m
    %
    % Module containing the ImplicitSolver class, to be used within implicit time stepping methods.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % The required argument when constructing an ImplicitSolver object is a
    % function for the Jacobian of the ODE right-hand side function, f_y.
    % This routine may return either a dense or sparse matrix; the matrix
    % format determines which MATLAB identity and LU factorization strategy
    % are used internally.
    %
    % Other optional inputs focus on specific Newton-Raphson solver options:
    %     maxiter = maximum allowed number of nonlinear iterations
    %     rtol    = relative solution tolerance
    %     atol    = absolute solution tolerance, either scalar or vector
    %     Jfreq   = frequency for reconstructing the Jacobian solver
    %
    properties
        f_y
        maxiter = 10, rtol = 1e-3, atol = 0.0, Jfreq = 1
        linear_solver = []
        total_iters = 0
        total_setups = 0
    end

    methods
        function self = ImplicitSolver(f_y, maxiter, rtol, atol, Jfreq)
            % Construct an implicit nonlinear solver for use by time steppers.
            if nargin < 1
                error('ImplicitSolver requires an f_y input.');
            end

            % required inputs
            self.f_y = f_y;

            % optional inputs
            if nargin >= 2 && ~isempty(maxiter), self.maxiter = maxiter; end
            if nargin >= 3 && ~isempty(rtol), self.rtol = rtol; end
            if nargin >= 4 && ~isempty(atol), self.atol = atol; end
            if nargin >= 5 && ~isempty(Jfreq) && Jfreq > 0, self.Jfreq = Jfreq; end
        end

        function setup_linear_solver(self, t, gamma, args)
            % Create a function that solve() can call to construct linear
            % solvers as needed for the Newton-Raphson method.  This is
            % designed to be called by the implicit time integrator at each
            % implicit step and/or stage.
            %
            % args holds optional user-provided parameters for the Jacobian.
            if nargin < 4
                args = {};
            end
            if ~iscell(args)
                error('ImplicitSolver:setup_linear_solver args must be a cell array.');
            end

            self.linear_solver = @(y) self.build_linear_solver(t, gamma, y, args);
        end

        function [y, iters, success] = solve(self, Ffcn, y0)
            % Implements a modified Newton-Raphson method for approximating
            % a root of the nonlinear system F(y) = 0.  The iteration stops
            % when
            %
            %    || (ynew - yold) / (atol + rtol*abs(ynew)) ||_RMS < 1.
            %
            % Required inputs:
            %     Ffcn = nonlinear residual function
            %     y0   = initial guess at the solution
            %
            % Outputs:
            %     y       = approximate solution
            %     iters   = number of nonlinear iterations performed
            %     success = true if iteration converged, false otherwise

            % ensure that linear_solver has been created
            if isempty(self.linear_solver)
                error('linear_solver has not been created');
            end

            % initialize outputs
            y = y0(:);
            iters = 0;
            success = false;

            % store nonlinear system size
            n = numel(y);

            % evaluate initial residual
            F = Ffcn(y);
            F = F(:);    % ensure that F is a column vector

            % set up initial Jacobian solver
            Jsolver = self.linear_solver(y);
            self.total_setups = self.total_setups + 1;

            % perform iteration
            for its = 1:self.maxiter
                % increment iteration counters
                iters = iters + 1;
                self.total_iters = self.total_iters + 1;

                % solve Newton linear system
                h = Jsolver(F);

                % compute Newton update
                y = y - h;

                % check for convergence
                scale = self.local_scale(y);
                if norm(h ./ scale) / sqrt(n) < 1
                    success = true;
                    return;
                end

                F = Ffcn(y);
                F = F(:);    % ensure that F is a column vector

                % update Jacobian every Jfreq iterations
                if mod(its, self.Jfreq) == 0
                    Jsolver = self.linear_solver(y);
                    self.total_setups = self.total_setups + 1;
                end
            end

            % if we made it here, then iteration did not converge; return
            % with the current solution and success still set to false
        end

        function n = get_total_iters(self)
            % Return the total number of nonlinear solver iterations over
            % the life of the solver.
            n = self.total_iters;
        end

        function n = get_total_setups(self)
            % Return the total number of linear solver setup calls over the
            % life of the solver.
            n = self.total_setups;
        end

        function reset(self)
            % Reset the solver statistics.
            self.total_iters = 0;
            self.total_setups = 0;
        end
    end

    methods (Access = private)
        function Jsolve = build_linear_solver(self, t, gamma, y, args)
            % Construct and factor the Newton matrix I + gamma*f_y(t,y).
            Jac = self.f_y(t, y, args{:});
            if issparse(Jac)
                Jac = speye(numel(y)) + gamma * Jac;
            else
                Jac = eye(numel(y)) + gamma * Jac;
            end

            try
                if exist('decomposition', 'file')
                    Jfact = decomposition(Jac, 'lu');
                    Jsolve = @(b) Jfact \ b;
                elseif issparse(Jac)
                    [L, U, P, Q] = lu(Jac);
                    Jsolve = @(b) Q * (U \ (L \ (P*b)));
                else
                    [L, U, P] = lu(Jac);
                    Jsolve = @(b) U \ (L \ (P*b));
                end
            catch
                error('Jacobian factorization failure');
            end
        end

        function scale = local_scale(self, y)
            % Construct the weighted RMS scaling vector used by the Newton
            % convergence test.
            scale = self.atol + self.rtol * abs(y);
            scale = max(scale, 1e-15);
        end
    end
end
