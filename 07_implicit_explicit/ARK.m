classdef ARK < handle
    % ARK.m
    %
    % Fixed-stepsize implicit-explicit additive Runge--Kutta stepper class
    % implementation file.
    %
    % Also contains functions to return specific ARK Butcher table pairs.  The
    % individual explicit and implicit tables are stored in ERK.m, DIRK.m,
    % AdaptERK.m and AdaptDIRK.m; the routines at the end of this file assemble
    % them into compatible pairs.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        fE, fI, sol, AE, bE, cE, AI, bI, cI, s
        h = 0.0
        steps = 0
        nsol = 0
        kE, kI, z, data
    end

    methods
        function self = ARK(fE, fI, sol, BE, BI, h)
            % Fixed stepsize implicit-explicit additive Runge--Kutta class
            % to perform fixed-stepsize time evolution of the IVP
            %      y' = fE(t,y) + fI(t,y),  t in [t0, Tf],  y(t0) = y0
            % using an implicit-explicit additive Runge--Kutta (ARK) time stepping
            % method.
            %
            % The five required arguments when constructing an ARK object are
            % functions for the explicit and implicit portions of the IVP right-hand
            % side, an implicit solver to use, and explicit and implicit Butcher tables:
            %     fE = explicit ODE RHS function with calling syntax fE(t,y).
            %     fI = implicit ODE RHS function with calling syntax fI(t,y).
            %     sol = algebraic solver object to use [ImplicitSolver]
            %     BE = explicit Runge--Kutta Butcher table.
            %     BI = diagonally-implicit Runge--Kutta Butcher table.
            %     h = (optional) input with requested stepsize to use for time stepping.
            %         Note that this MUST be set either here or in the Evolve call.
            if nargin < 5
                error('ARK requires fE, fI, implicit solver object, and explicit and implicit Butcher tables.');
            end

            % required inputs
            self.fE = fE;
            self.fI = fI;
            self.sol = sol;
            self.AE = BE.A;
            self.bE = BE.b(:);
            self.cE = BE.c(:);
            self.AI = BI.A;
            self.bI = BI.b(:);
            self.cI = BI.c(:);

            % optional inputs
            if nargin >= 6 && ~isempty(h), self.h = h; end

            % internal data
            self.s = numel(self.cE);

            % check for legal tables (including matching numbers of stages)
            if numel(self.cI) ~= self.s || numel(self.bE) ~= self.s || ...
                    numel(self.bI) ~= self.s || size(self.AE,1) ~= self.s || ...
                    size(self.AE,2) ~= self.s || size(self.AI,1) ~= self.s || ...
                    size(self.AI,2) ~= self.s || ...
                    norm(self.AE - tril(self.AE,-1), inf) > 1e-14 || ...
                    norm(self.AI - tril(self.AI,0), inf) > 1e-14
                error('ARK: incompatible Butcher tables supplied');
            end
        end

        function [t, y, success] = ark_step(self, t, y, h, args)
            % Usage: t, y, success = ark_step(t, y, h, args)
            %
            % Utility routine to take a single implicit-explicit ARK time step of size h,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS functions.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 5
                args = {};
            end
            if ~iscell(args)
                error('ARK: args must be a cell array.');
            end

            y = y(:);
            % loop over stages, computing RHS vectors
            for i = 1:self.s

                % construct "data" for this stage solve
                self.data = y;
                for j = 1:(i-1)
                    self.data = self.data + h * (self.AE(i,j) * self.kE(j,:).' ...
                                                 + self.AI(i,j) * self.kI(j,:).');
                end

                % solve the implicit stage (or copy the data for an explicit stage)
                tstageI = t + h*self.cI(i);
                if abs(self.AI(i,i)) > 1e-14
                    F = @(zcur) zcur(:) - self.data(:) - h * self.AI(i,i) * self.fI(tstageI, zcur(:), args{:});
                    self.sol.setup_linear_solver(tstageI, -h * self.AI(i,i), args);
                    [self.z, ~, success] = self.sol.solve(F, y);
                    self.nsol = self.nsol + 1;
                    if ~success
                        return;
                    end
                else
                    self.z = self.data;
                end

                % store both RHS vectors at this stage
                self.kE(i,:) = self.fE(t + h*self.cE(i), self.z, args{:}).';
                self.kI(i,:) = self.fI(tstageI, self.z, args{:}).';
            end

            % update time step solution
            for i = 1:self.s
                y = y + h * (self.bE(i) * self.kE(i,:).' + self.bI(i) * self.kI(i,:).');
            end
            t = t + h;
            self.steps = self.steps + 1;
            success = true;
        end

        function update_rhs(self, fE, fI)
            % Updates the RHS functions (cannot change vector dimensions)
            self.fE = fE;
            self.fI = fI;
        end

        function reset(self)
            % Resets the accumulated number of steps
            self.steps = 0;
            self.nsol = 0;
        end

        function n = get_num_steps(self)
            % Returns the accumulated number of steps
            n = self.steps;
        end

        function n = get_num_solves(self)
            % Returns the accumulated number of implicit solves
            n = self.nsol;
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The fixed-step ARK evolution routine
            %
            % Inputs:  tspan holds the current time interval, [t0, tf], including any
            %              intermediate times when the solution is desired, i.e.
            %              [t0, t1, ..., tf]
            %          y holds the initial condition, y(t0)
            %          h optionally holds the requested step size (if it is not
            %              provided then the stored value will be used)
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
                error('ARK:Evolve args must be a cell array.');
            end

            % set time step for evolution based on input-vs-stored value
            if h ~= 0.0
                self.h = h;
            end

            % raise error if step size was never set
            if self.h == 0.0
                error('ARK:Evolve called without specifying a nonzero step size');
            end

            % initialize output, and set first entry corresponding to initial condition
            tspan = tspan(:);
            y = y0(:);
            nout = numel(tspan);
            m = numel(y);
            Y = zeros(nout, m);
            Y(1,:) = y.';

            % initialize internal solution-vector-sized data
            self.kE = zeros(self.s, m);
            self.kI = zeros(self.s, m);
            self.z = y;
            self.data = y;

            % loop over desired output times
            for iout = 2:nout

                % determine how many internal steps are required, and the actual step size to use
                [N, hcur] = substeps(tspan(iout)-tspan(iout-1), self.h);

                % reset "current" t that will be evolved internally
                t = tspan(iout-1);

                % iterate over internal time steps to reach next output
                for n = 1:N

                    % perform implicit-explicit additive Runge--Kutta update
                    [t, y, success] = self.ark_step(t, y, hcur, args);
                    if ~success
                        fprintf('ARK::Evolve error in time step at t = %g\n', t);
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

    % ARK Butcher table pair routines
    methods (Static)
        function [BE, BI] = ARS111()
            % Usage: [BE, BI] = ARK.ARS111()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the ARS(1,1,1) forward-backward Euler method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.Ascher111ERK();
            BI = DIRK.Ascher111SDIRK();
        end

        function [BE, BI] = ARS122()
            % Usage: [BE, BI] = ARK.ARS122()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the ARS(1,2,2) implicit-explicit midpoint method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.Ascher122ERK();
            BI = DIRK.Ascher122SDIRK();
        end

        function [BE, BI] = ARS222()
            % Usage: [BE, BI] = ARK.ARS222()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the ARS(2,2,2) method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = AdaptERK.Ascher222ERK();
            BI = AdaptDIRK.Ascher222SDIRK();
        end

        function [BE, BI] = ARS232()
            % Usage: [BE, BI] = ARK.ARS232()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the ARS(2,3,2) method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.Ascher232ERK();
            BI = DIRK.Ascher232SDIRK();
        end

        function [BE, BI] = ARS233()
            % Usage: [BE, BI] = ARK.ARS233()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the ARS(2,3,3) method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.Ascher233ERK();
            BI = DIRK.Ascher233SDIRK();
        end

        function [BE, BI] = ARS343()
            % Usage: [BE, BI] = ARK.ARS343()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the ARS(3,4,3) method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.Ascher343ERK();
            BI = DIRK.Ascher343SDIRK();
        end

        function [BE, BI] = ARS443()
            % Usage: [BE, BI] = ARK.ARS443()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the ARS(4,4,3) method, with padded SDIRK table.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.Ascher443ERK();
            BI = DIRK.Ascher443PaddedSDIRK();
        end

        function [BE, BI] = SSP222()
            % Usage: [BE, BI] = ARK.SSP222()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the SSP2(2,2,2) method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.SSP222ERK();
            BI = DIRK.SSP222SDIRK();
        end

        function [BE, BI] = SSP2332Lpm1()
            % Usage: [BE, BI] = ARK.SSP2332Lpm1()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the SSP2(3,3,2)-lpm1 method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.SSP2332Lpm1ERK();
            BI = DIRK.SSP2332Lpm1SDIRK();
        end

        function [BE, BI] = SSP2332Lpm2()
            % Usage: [BE, BI] = ARK.SSP2332Lpm2()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the SSP2(3,3,2)-lpm2 method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.SSP2332Lpm2ERK();
            BI = DIRK.SSP2332Lpm2SDIRK();
        end

        function [BE, BI] = SSP2332Lpum()
            % Usage: [BE, BI] = ARK.SSP2332Lpum()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the SSP2(3,3,2)-lpum method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.SSP2332LpumERK();
            BI = DIRK.SSP2332LpumSDIRK();
        end

        function [BE, BI] = SSP2332a()
            % Usage: [BE, BI] = ARK.SSP2332a()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the SSP2(3,3,2)-a method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.SSP2332aERK();
            BI = DIRK.SSP2332aDIRK();
        end

        function [BE, BI] = DBM53()
            % Usage: [BE, BI] = ARK.DBM53()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the DBM-5-3 method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.DBM53ERK();
            BI = DIRK.DBM53ESDIRK();
        end

        function [BE, BI] = Cooper4()
            % Usage: [BE, BI] = ARK.Cooper4()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the Cooper4 method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.Cooper4ERK();
            BI = DIRK.Cooper4ESDIRK();
        end

        function [BE, BI] = SSP3433()
            % Usage: [BE, BI] = ARK.SSP3433()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the SSP3(4,3,3) method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.SSP3433ERK();
            BI = DIRK.SSP3433SDIRK();
        end

        function [BE, BI] = Cooper6()
            % Usage: [BE, BI] = ARK.Cooper6()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the Cooper6 method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.Cooper6ERK();
            BI = DIRK.Cooper6ESDIRK();
        end

        function [BE, BI] = ARK324L2SA()
            % Usage: [BE, BI] = ARK.ARK324L2SA()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to Kennedy & Carpenter's ARK3(2)4L[2]SA method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = AdaptERK.ARK324L2SAERK();
            BI = AdaptDIRK.ARK324L2SAESDIRK();
        end

        function [BE, BI] = ARK436L2SA()
            % Usage: [BE, BI] = ARK.ARK436L2SA()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to Kennedy & Carpenter's ARK4(3)6L[2]SA method.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = AdaptERK.ARK436L2SAERK();
            BI = AdaptDIRK.ARK436L2SAESDIRK();
        end

        function [BE, BI] = HeunImplicitMidpoint()
            % Usage: [BE, BI] = ARK.HeunImplicitMidpoint()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the first-order pairing of Heun's method with the padded
            % implicit midpoint method, from the ARK coupling example.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.Heun();
            BI = DIRK.Ascher122SDIRK();
        end

        function [BE, BI] = RK4ESDIRK3()
            % Usage: [BE, BI] = ARK.RK4ESDIRK3()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the second-order pairing of the classical RK4 method with
            % a third-order ESDIRK method, from the ARK coupling example.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.ERK4();
            BI = DIRK.ARKCouplingESDIRK3();
        end

        function [BE, BI] = ERK3ESDIRK3()
            % Usage: [BE, BI] = ARK.ERK3ESDIRK3()
            %
            % Utility routine to return the ARK Butcher table pair corresponding
            % to the third-order pairing of a third-order ERK method with
            % a third-order ESDIRK method, from the ARK coupling example.
            %
            % Outputs: BE holds the explicit Butcher table
            %          BI holds the implicit Butcher table
            ARK.add_table_paths();
            BE = ERK.ARKCouplingERK3();
            BI = DIRK.ARKCouplingESDIRK3();
        end
    end

    % path utility routine for the Butcher table pair routines
    methods (Static, Access = private)
        function add_table_paths()
            % Usage: ARK.add_table_paths()
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
