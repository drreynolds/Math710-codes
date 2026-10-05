classdef FractionalStep < handle
    % FractionalStep.m
    %
    % Fixed-stepsize fractional-step (operator-splitting) time stepper class
    % implementation file.
    %
    % Class to perform fixed-stepsize time evolution of the M-way split IVP
    %      y' = f^{1}(t,y) + f^{2}(t,y) + ... + f^{M}(t,y),  t in [t0, Tf],  y(t0) = y0
    % using a fractional-step method
    %      Psi_H = Phi^{s}_{a_s H} o ... o Phi^{1}_{a_1 H},
    %      Phi^{k}_{a_k H} = phi^{M}_{alpha_k^{M} H} o ... o phi^{1}_{alpha_k^{1} H},
    % where each sub-flow phi^{l} is approximated by any object that supports the
    % "Evolve" routine for the sub-IVP
    %      y' = f^{l}(t,y),  t in [t_k, t_k + alpha_k^{l} H].
    %
    % Also contains functions to return specific fractional-step coefficients.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize fractional-step time stepper class
    %
    % The two required arguments when constructing a FractionalStep object
    % are a table of fractional-step coefficients, and a cell array of solvers
    % for the sub-IVPs, one per partition:
    %     S = fractional-step coefficient table (e.g., S = FractionalStep.StrangMarchuk()),
    %         with the field
    %         S.alpha = M x s array, where alpha(l,k) holds the fraction
    %             alpha_k^{l} of the step taken by partition l in stage k
    %             (the same layout as the table of fractional-step methods
    %             in the lecture notes, with one row per partition).
    %     Solvers = cell array of M objects that implement the "Evolve" method,
    %             where Solvers{l} evolves the sub-IVP y' = f^{l}(t,y).  Each
    %             solver stores its own right-hand side function and, for
    %             fixed-step solvers, its own step size.
    %     H = (optional) input with requested stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.
    %
    % Within each stage, partition 1 is advanced first.  Sub-steps with
    % alpha_k^{l} = 0 are skipped.  A sub-step with alpha_k^{l} < 0 asks its
    % solver to evolve backward in time; all of the course solvers (fixed-step
    % and adaptive) support this, although a backward sub-step of a dissipative
    % piece (e.g., diffusion) is considered inherently unstable.

    % Stored method coefficients, sub-solvers, and run statistics.
    properties
        alpha, Solvers
        H = 0.0
        steps = 0
        M, s
    end

    methods
        function self = FractionalStep(S, Solvers, H)
            if nargin < 2
                error('FractionalStep requires a coefficient table and a cell array of solvers.');
            end

            % required inputs
            self.alpha = double(S.alpha);
            self.Solvers = Solvers;

            % optional inputs
            if nargin >= 3 && ~isempty(H), self.H = H; end

            % internal data
            self.steps = 0;
            self.M = size(self.alpha, 1);
            self.s = size(self.alpha, 2);

            % check for legal inputs
            if numel(self.Solvers) ~= self.M
                error('FractionalStep: need one solver per partition');
            end
            for l = 1:self.M
                if ~ismethod(self.Solvers{l}, 'Evolve')
                    error('FractionalStep: each solver must implement the Evolve method');
                end
            end
            if norm(sum(self.alpha, 2) - 1.0, inf) > 1e-14
                error('FractionalStep: each partition must advance a total time of H');
            end
        end

        function [t, y, success] = step(self, t, y, H, args)
            % Usage: t, y, success = step(t, y, H, args)
            %
            % Utility routine to take a single fractional-step time step of size H,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 5
                args = {};
            end
            if ~iscell(args)
                error('FractionalStep: args must be a cell array.');
            end

            y = y(:);

            % tau(l) holds how far partition l has been advanced within this
            % step, so that each sub-IVP starts at that partition's own time
            tau = zeros(self.M, 1);

            % loop over stages, and over partitions within each stage
            for k = 1:self.s
                for l = 1:self.M

                    % skip sub-steps of zero length
                    if self.alpha(l,k) == 0.0
                        continue;
                    end

                    % call the partition's solver to evolve its sub-IVP
                    tspan = [t + tau(l)*H; t + (tau(l) + self.alpha(l,k))*H];
                    [ytmp, success] = self.Solvers{l}.Evolve(tspan, y, [], args);
                    if ~success
                        self.steps = self.steps + 1;
                        return;
                    end
                    y = ytmp(2,:).';  % extract the solution at the end of the sub-step
                    tau(l) = tau(l) + self.alpha(l,k);
                end
            end

            % update current time and step counter, and return
            t = t + H;
            self.steps = self.steps + 1;
            success = true;
        end

        function reset(self)
            % Resets the accumulated number of steps
            self.steps = 0;
            for l = 1:self.M
                if ismethod(self.Solvers{l}, 'reset')
                    self.Solvers{l}.reset();
                end
            end
        end

        function n = get_num_steps(self)
            % Returns the accumulated number of fractional steps
            n = self.steps;
        end

        function n = get_num_rhs(self)
            % Returns the accumulated number of RHS evaluations, over all partitions
            n = 0;
            for l = 1:self.M
                if ismethod(self.Solvers{l}, 'get_num_rhs')
                    n = n + self.Solvers{l}.get_num_rhs();
                end
            end
        end

        function [Y, success] = Evolve(self, tspan, y0, H, args)
            % Usage: Y, success = Evolve(tspan, y0, H, args)
            %
            % The fixed-step fractional-step evolution routine
            %
            % Inputs:  tspan holds the current time interval, [t0, tf], including any
            %              intermediate times when the solution is desired, i.e.
            %              [t0, t1, ..., tf]
            %          y holds the initial condition, y(t0)
            %          H optionally holds the requested step size (if it is not
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
                error('FractionalStep:Evolve args must be a cell array.');
            end

            % set time step for evoluation based on input-vs-stored value
            if H ~= 0.0
                self.H = H;
            end

            % raise error if step size was never set
            if self.H == 0.0
                error('FractionalStep:Evolve called without specifying a nonzero step size');
            end

            % initialize output, and set first entry corresponding to initial condition
            tspan = tspan(:);
            y = y0(:);
            Y = zeros(numel(tspan), numel(y));
            Y(1,:) = y.';

            % loop over desired output times
            for iout = 2:numel(tspan)

                % determine how many internal steps are required, and the actual step size to use
                [N, H] = substeps(tspan(iout)-tspan(iout-1), self.H);

                % reset "current" t that will be evolved internally
                t = tspan(iout-1);

                % iterate over internal time steps to reach next output
                for n = 1:N

                    % perform fractional step
                    [t, y, success] = self.step(t, y, H, args);
                    if ~success
                        fprintf('FractionalStep error in time step at t = %g\n', t);
                        success = false;
                        return;
                    end
                end

                % store current results in output arrays
                Y(iout,:) = y.';
            end

            % return with "success" flag
            success = true;
        end
    end

    methods (Static)
        function S = LieTrotter(M)
            % Usage: S = FractionalStep.LieTrotter(M)
            %
            % Utility routine to return the coefficients of the first-order Lie--Trotter
            % splitting for M partitions (default 2), which advances each partition
            % once, over the full step, in the order 1, 2, ..., M.
            %
            % Reference: lecture notes, table of fractional-step methods.
            %
            % Outputs: S.alpha holds the fractional-step coefficients
            %          S.p holds the splitting order
            if nargin < 1
                M = 2;
            end
            alpha = ones(M, 1);
            S = struct('alpha', alpha, 'p', 1);
        end

        function S = LieTrotterAdjoint(M)
            % Usage: S = FractionalStep.LieTrotterAdjoint(M)
            %
            % Utility routine to return the coefficients of the adjoint of the
            % Lie--Trotter splitting for M partitions (default 2), which advances each
            % partition once, over the full step, in the order M, M-1, ..., 1.  Since
            % partition 1 is advanced first within each stage, this uses M stages,
            % where stage k advances only partition M+1-k.
            %
            % Reference: lecture notes, discussion of Lie--Trotter splitting.
            %
            % Outputs: S.alpha holds the fractional-step coefficients
            %          S.p holds the splitting order
            if nargin < 1
                M = 2;
            end
            alpha = zeros(M, M);
            for l = 1:M
                alpha(l, M+1-l) = 1.0;
            end
            S = struct('alpha', alpha, 'p', 1);
        end

        function S = StrangMarchuk(M)
            % Usage: S = FractionalStep.StrangMarchuk(M)
            %
            % Utility routine to return the coefficients of the second-order symmetric
            % Strang--Marchuk splitting for M partitions (default 2): a half step of
            % partitions 1, ..., M-1, a full step of partition M, and then half steps of
            % partitions M-1, ..., 1.  Since partition 1 is advanced first within each
            % stage, this uses M stages, where stage 1 advances every partition and
            % stage k > 1 advances only partition M+1-k.  For M = 2 these are the
            % coefficients in the lecture notes.
            %
            % Reference: lecture notes, table of fractional-step methods.
            %
            % Outputs: S.alpha holds the fractional-step coefficients
            %          S.p holds the splitting order
            if nargin < 1
                M = 2;
            end
            alpha = zeros(M, M);
            alpha(:, 1) = 0.5;
            alpha(M, 1) = 1.0;
            for l = 1:(M-1)
                alpha(l, M+1-l) = 0.5;
            end
            S = struct('alpha', alpha, 'p', 2);
        end

        function S = OS2(mu)
            % Usage: S = FractionalStep.OS2(mu)
            %
            % Utility routine to return the coefficients of the two-stage,
            % second-order, 2-way splitting OS2(2,2)-mu, for a parameter mu != 1.
            % All coefficients are positive for 0 < mu < 1/2; mu = 1/2 gives the
            % Strang--Marchuk splitting with the roles of the partitions exchanged.
            %
            % Reference: Spiteri & Wei, J. Comput. Phys. 476:111900 (2023), Table 1,
            %            doi:10.1016/j.jcp.2022.111900.
            %
            % Outputs: S.alpha holds the fractional-step coefficients
            %          S.p holds the splitting order
            if mu == 1.0
                error('FractionalStep.OS2: requires mu != 1');
            end
            alpha = [(2*mu-1)/(2*mu-2), -1/(2*mu-2);
                     1-mu, mu];
            S = struct('alpha', alpha, 'p', 2);
        end

        function S = Ruth()
            % Usage: S = FractionalStep.Ruth()
            %
            % Utility routine to return the coefficients of the three-stage,
            % third-order, 2-way Ruth splitting.  Note that each partition takes one
            % backward (negative) sub-step.
            %
            % Reference: Spiteri & Wei, J. Comput. Phys. 476:111900 (2023), Table 3,
            %            doi:10.1016/j.jcp.2022.111900.
            %
            % Outputs: S.alpha holds the fractional-step coefficients
            %          S.p holds the splitting order
            alpha = [7/24, 3/4, -1/24;
                     2/3, -2/3, 1];
            S = struct('alpha', alpha, 'p', 3);
        end

        function S = OS3_32()
            % Usage: S = FractionalStep.OS3_32()
            %
            % Utility routine to return the coefficients of the three-stage,
            % second-order, 3-way splitting OS3(3,2).  Note that partitions 2 and 3
            % each take one backward (negative) sub-step.
            %
            % Reference: Spiteri & Wei, J. Comput. Phys. 476:111900 (2023), Table 2,
            %            doi:10.1016/j.jcp.2022.111900.
            %
            % Outputs: S.alpha holds the fractional-step coefficients
            %          S.p holds the splitting order
            alpha = [1/3, 1/3, 1/3;
                     1, -1/2, 1/2;
                     1/4, 1, -1/4];
            S = struct('alpha', alpha, 'p', 2);
        end
    end
end
