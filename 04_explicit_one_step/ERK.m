classdef ERK < handle
    % ERK.m
    %
    % Fixed-stepsize explicit Runge--Kutta stepper class implementation file.
    %
    % Also contains functions to return specific explicit Butcher tables.
    %
    % Class to perform fixed-stepsize time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using an explicit Runge--Kutta (ERK) time stepping method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize explicit Runge--Kutta class
    %
    % The two required arguments when constructing an ERK object are a
    % function for the IVP right-hand side, and a Butcher table:
    %     f = ODE RHS function with calling syntax f(t,y).
    %     B = Explicit Runge--Kutta Butcher table.
    %     h = (optional) input with requested stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        f, A, b, c, s
        h = 0.0
        steps = 0
        nrhs = 0
        k, z
    end

    methods
        function self = ERK(f, B, h)
            if nargin < 2
                error('ERK requires an RHS function handle and a Butcher table.');
            end

            % Store the RHS and unpack the Butcher table.
            self.f = f;
            self.A = B.A;
            self.b = B.b(:);
            self.c = B.c(:);
            % optional inputs
            if nargin >= 3 && ~isempty(h), self.h = h; end
            self.s = numel(self.c);

            % Explicit RK tables must have matching sizes and strictly lower triangular A.
            if numel(self.b) ~= self.s || size(self.A,1) ~= self.s || size(self.A,2) ~= self.s || ...
                    norm(self.A - tril(self.A,-1), inf) > 1e-14
                error('ERK: incompatible Butcher table supplied');
            end
        end

        function [t, y, success] = erk_step(self, t, y, h, args)
            % Usage: t, y, success = erk_step(t, y, h, args)
            %
            % Utility routine to take a single explicit RK time step of size h,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 5
                args = {};
            end
            if ~iscell(args)
                error('ERK: args must be a cell array.');
            end

            % Compute each stage using only previously available explicit stages.
            self.k(1,:) = self.f(t, y, args{:}).';
            self.nrhs = self.nrhs + 1;
            for i = 2:self.s
                self.z = y(:);
                for j = 1:(i-1)
                    self.z = self.z + h * self.A(i,j) * self.k(j,:).';
                end
                self.k(i,:) = self.f(t + self.c(i)*h, self.z, args{:}).';
                self.nrhs = self.nrhs + 1;
            end

            y = y(:);
            % combine stage data to update the time-step solution
            for i = 1:self.s
                y = y + h * self.b(i) * self.k(i,:).';
            end
            t = t + h;
            self.steps = self.steps + 1;
            success = true;
        end

        function update_rhs(self, f)
            % Updates the RHS function (cannot change vector dimensions)
            self.f = f;
        end

        function reset(self)
            % Resets the accumulated number of steps

            % reset accumulated statistics
            self.steps = 0;
            self.nrhs = 0;
        end

        function n = get_num_steps(self)
            % Returns the accumulated number of steps
            n = self.steps;
        end

        function n = get_num_rhs(self)
            % Returns the accumulated number of RHS evaluations
            n = self.nrhs;
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The fixed-step explicit Runge--Kutta evolution routine.
            %
            % Inputs:  tspan holds the current time interval, [t0, tf], including any
            %              intermediate times when the solution is desired, i.e.
            %              [t0, t1, ..., tf]
            %          y holds the initial condition, y(t0)
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
                error('ERK:Evolve args must be a cell array.');
            end

            % update stored stepsize when requested by the caller
            if h ~= 0.0
                self.h = h;
            end
            % require a nonzero stepsize before evolving
            if self.h == 0.0
                error('ERK:Evolve called without specifying a nonzero step size');
            end

            % Initialize output storage, with the first row holding the initial condition.
            tspan = tspan(:);
            y = y0(:);
            nout = numel(tspan);
            m = numel(y);
            Y = zeros(nout, m);
            Y(1,:) = y.';

            % Allocate stage and work vectors now that the solution dimension is known.
            self.k = zeros(self.s, m);
            self.z = y;

            % iterate over output times, filling the solution history
            for iout = 2:nout
                dt = tspan(iout) - tspan(iout-1);

                % determine how many internal steps are required, and the actual step size to use
                [N, hcur] = substeps(dt, self.h);
                t = tspan(iout-1);

                % March internally until the next requested output time is reached.
                for n = 1:N
                    [t, y, success] = self.erk_step(t, y, hcur, args);
                    if ~success
                        fprintf('ERK::Evolve error in time step at t = %g\n', t);
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
        function B = ERK1()
            % Usage: B = ERK1()
            %
            % Utility routine to return the ERK table corresponding to forward Euler, posed as an ERK method.
            %
            % Reference: Euler, Institutiones calculi integralis, Vol. 1 (1768).
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = 0.0;
            B.b = 1.0;
            B.c = 0.0;
            B.p = 1;
        end

        function B = Heun()
            % Usage: B = Heun()
            %
            % Utility routine to return the ERK table corresponding to Heun's method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0.0, 0.0; 1.0, 0.0];
            B.b = [0.5; 0.5];
            B.c = [0.0; 1.0];
            B.p = 2;
        end

        function B = ERK2()
            % Usage: B = ERK2()
            %
            % Utility routine to return the ERK table corresponding
            % to the standard 2nd-order ERK method.
            %
            % Reference: Runge, Math. Ann. 46 (1895), doi:10.1007/BF01446807.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0.0, 0.0; 0.5, 0.0];
            B.b = [0.0; 1.0];
            B.c = [0.0; 0.5];
            B.p = 2;
        end

        function B = ERK3()
            % Usage: B = ERK3()
            %
            % Utility routine to return the ERK table corresponding
            % to the standard 3rd-order ERK method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0.0, 0.0, 0.0; 2.0/3.0, 0.0, 0.0; 0.0, 2.0/3.0, 0.0];
            B.b = [0.25; 3.0/8.0; 3.0/8.0];
            B.c = [0.0; 2.0/3.0; 2.0/3.0];
            B.p = 3;
        end

        function B = ERK4()
            % Usage: B = ERK4()
            %
            % Utility routine to return the ERK table corresponding
            % to the standard 4th-order ERK method.
            %
            % Reference: Kutta, Z. Math. Phys. 46:435--453 (1901).
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0.0, 0.0, 0.0, 0.0; ...
                   0.5, 0.0, 0.0, 0.0; ...
                   0.0, 0.5, 0.0, 0.0; ...
                   0.0, 0.0, 1.0, 0.0];
            B.b = [1.0/6.0; 1.0/3.0; 1.0/3.0; 1.0/6.0];
            B.c = [0.0; 0.5; 0.5; 1.0];
            B.p = 4;
        end

        % Additional non-embedded explicit Runge--Kutta tables.

        function B = ERK11()
            % Usage: B = ERK11()
            %
            % Utility routine to return the ERK table corresponding to the ERK-1-1
            % method.
            %
            % Reference: Euler, Institutiones calculi integralis, Vol. 1 (1768).
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = 0;
            B.b = 1;
            B.c = 0;
            B.p = 1;
        end

        function B = Ascher232ERK()
            % Usage: B = Ascher232ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % Ascher(2,3,2)-ERK method.
            %
            % Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
            %            doi:10.1016/S0168-9274(97)00056-1.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0; ...
                   0.29289321881345243, 0, 0; ...
                   -0.94280904158206347, 1.9428090415820636, 0];
            B.b = [0; 0.70710678118654757; 0.29289321881345243];
            B.c = [0; 0.29289321881345243; 1];
            B.p = 2;
        end

        function B = ERK22()
            % Usage: B = ERK22()
            %
            % Utility routine to return the ERK table corresponding to the ERK-2-2
            % method.
            %
            % Reference: Ralston, Math. Comp. 16 (1962),
            %            doi:10.1090/S0025-5718-1962-0150954-0.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0; ...
                   0.66666666666666663, 0];
            B.b = [0.25; 0.75];
            B.c = [0; 0.66666666666666663];
            B.p = 2;
        end

        function B = LobattoIII22IRK()
            % Usage: B = LobattoIII22IRK()
            %
            % Utility routine to return the ERK table corresponding to the
            % LobattoIII-2-2-IRK method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0; ...
                   1, 0];
            B.b = [0.5; 0.5];
            B.c = [0; 1];
            B.p = 2;
        end

        function B = SSP222ERK()
            % Usage: B = SSP222ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % SSP2(2,2,2)-ERK method.
            %
            % Reference: Pareschi & Russo, J. Sci. Comput. 25 (2005),
            %            doi:10.1007/BF02728986.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0; ...
                   1, 0];
            B.b = [0.5; 0.5];
            B.c = [0; 1];
            B.p = 2;
        end

        function B = SSP2332Lpm1ERK()
            % Usage: B = SSP2332Lpm1ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % SSP2(3,3,2)-lpm1-ERK method.
            %
            % Reference: Higueras, Happenhofer, Koch & Kupka, J. Comput. Appl. Math. 272
            %            (2014), doi:10.1016/j.cam.2014.05.011.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0; ...
                   0.5, 0, 0; ...
                   0.5, 0.5, 0];
            B.b = [0.33333333333333331; 0.33333333333333331; 0.33333333333333331];
            B.c = [0; 0.5; 1];
            B.p = 2;
        end

        function B = SSP2332Lpm2ERK()
            % Usage: B = SSP2332Lpm2ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % SSP2(3,3,2)-lpm2-ERK method.
            %
            % Reference: Higueras, Happenhofer, Koch & Kupka, J. Comput. Appl. Math. 272
            %            (2014), doi:10.1016/j.cam.2014.05.011.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0; ...
                   0.5, 0, 0; ...
                   0.5, 0.5, 0];
            B.b = [0.33333333333333331; 0.33333333333333331; 0.33333333333333331];
            B.c = [0; 0.5; 1];
            B.p = 2;
        end

        function B = SSP2332LpumERK()
            % Usage: B = SSP2332LpumERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % SSP2(3,3,2)-lpum-ERK method.
            %
            % Reference: Higueras, Happenhofer, Koch & Kupka, J. Comput. Appl. Math. 272
            %            (2014), doi:10.1016/j.cam.2014.05.011.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0; ...
                   0.5, 0, 0; ...
                   0.5, 0.5, 0];
            B.b = [0.33333333333333331; 0.33333333333333331; 0.33333333333333331];
            B.c = [0; 0.5; 1];
            B.p = 2;
        end

        function B = SSP2332aERK()
            % Usage: B = SSP2332aERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % SSP2(3,3,2)-a-ERK method.
            %
            % Reference: Pareschi & Russo, J. Sci. Comput. 25 (2005),
            %            doi:10.1007/BF02728986.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0; ...
                   0.5, 0, 0; ...
                   0.5, 0.5, 0];
            B.b = [0.33333333333333331; 0.33333333333333331; 0.33333333333333331];
            B.c = [0; 0.5; 1];
            B.p = 2;
        end

        function B = DBM53ERK()
            % Usage: B = DBM53ERK()
            %
            % Utility routine to return the ERK table corresponding to the DBM-5-3-ERK
            % method.
            %
            % Reference: the DBM453 method of Vogl, Steyer, Reynolds, Ullrich &
            %            Woodward, J. Adv. Model. Earth Syst. 11 (2019),
            %            doi:10.1029/2019MS001700.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0, 0; ...
                   0.10306208811591838, 0, 0, 0, 0; ...
                   -0.94124866143519892, 1.6626399742527356, 0, 0, 0; ...
                   -1.3670975201437765, 1.3815852911016873, 1.2673234025619065, 0, 0; ...
                   -0.81287582068772446, 0.8122373906050574, 0.90644429603699306, 0.094194134045674116, 0];
            B.b = [0.87795339639076675; -0.72692641526151547; 0.75204137157372719; -0.2289802940041509; 0.32591194130117246];
            B.c = [0; 0.1030620881159184; 0.72139131281753666; 1.2818111735198174; 1];
            B.p = 3;
        end

        function B = Ascher233ERK()
            % Usage: B = Ascher233ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % Ascher(2,3,3)-ERK method.
            %
            % Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
            %            doi:10.1016/S0168-9274(97)00056-1.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0; ...
                   0.78867513459481275, 0, 0; ...
                   -0.21132486540518725, 0.42264973081037449, 0];
            B.b = [0; 0.5; 0.5];
            B.c = [0; 0.78867513459481275; 0.21132486540518725];
            B.p = 3;
        end

        function B = Ascher343ERK()
            % Usage: B = Ascher343ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % Ascher(3,4,3)-ERK method.
            %
            % Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
            %            doi:10.1016/S0168-9274(97)00056-1.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0; ...
                   0.435866521508459, 0, 0, 0; ...
                   0.3212788860286272, 0.39665437472560205, 0, 0; ...
                   -0.10585829607187969, 0.55292914803593984, 0.55292914803593984, 0];
            B.b = [0; 1.2084966491760101; -0.64436317068446924; 0.435866521508459];
            B.c = [0; 0.435866521508459; 0.71793326075422947; 1];
            B.p = 3;
        end

        function B = Ascher443ERK()
            % Usage: B = Ascher443ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % Ascher(4,4,3)-ERK method.
            %
            % Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
            %            doi:10.1016/S0168-9274(97)00056-1.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0, 0; ...
                   0.5, 0, 0, 0, 0; ...
                   0.61111111111111116, 0.055555555555555552, 0, 0, 0; ...
                   0.83333333333333337, -0.83333333333333337, 0.5, 0, 0; ...
                   0.25, 1.75, 0.75, -1.75, 0];
            B.b = [0.25; 1.75; 0.75; -1.75; 0];
            B.c = [0; 0.5; 0.66666666666666663; 0.5; 1];
            B.p = 3;
        end

        function B = KnothWolkeERK()
            % Usage: B = KnothWolkeERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % Knoth-Wolke-ERK method.
            %
            % Reference: Knoth & Wolke, Appl. Numer. Math. 28 (1998),
            %            doi:10.1016/S0168-9274(98)00051-8.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0; ...
                   0.33333333333333331, 0, 0; ...
                   -0.1875, 0.9375, 0];
            B.b = [0.16666666666666666; 0.29999999999999999; 0.53333333333333333];
            B.c = [0; 0.33333333333333331; 0.75];
            B.p = 3;
        end

        function B = Cooper4ERK()
            % Usage: B = Cooper4ERK()
            %
            % Utility routine to return the ERK table corresponding to the Cooper4-ERK
            % method.
            %
            % Reference: the order-3 methods with mu = 1/2 in Cooper & Sayfy, Math.
            %            Comp. 40 (1983), doi:10.1090/S0025-5718-1983-0679441-1.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0; ...
                   0.66666666666666663, 0, 0, 0; ...
                   0.16666666666666666, 0.5, 0, 0; ...
                   0.25, 0.25, 0.5, 0];
            B.b = [0.25; 0.25; 0.5; 0];
            B.c = [0; 0.66666666666666663; 0.66666666666666663; 1];
            B.p = 3;
        end

        function B = SSP3332ERK()
            % Usage: B = SSP3332ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % SSP3(3,3,2)-ERK method.
            %
            % Reference: Pareschi & Russo, J. Sci. Comput. 25 (2005),
            %            doi:10.1007/BF02728986.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0; ...
                   1, 0, 0; ...
                   0.25, 0.25, 0];
            B.b = [0.16666666666666666; 0.16666666666666666; 0.66666666666666663];
            B.c = [0; 1; 0.5];
            B.p = 3;
        end

        function B = SSP3433ERK()
            % Usage: B = SSP3433ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % SSP3(4,3,3)-ERK method.
            %
            % Reference: Pareschi & Russo, J. Sci. Comput. 25 (2005),
            %            doi:10.1007/BF02728986.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0; ...
                   0, 0, 0, 0; ...
                   0, 1, 0, 0; ...
                   0, 0.25, 0.25, 0];
            B.b = [0; 0.16666666666666666; 0.16666666666666666; 0.66666666666666663];
            B.c = [0; 0; 1; 0.5];
            B.p = 3;
        end

        function B = ThreeEighthRuleERK()
            % Usage: B = ThreeEighthRuleERK()
            %
            % Utility routine to return the ERK table corresponding to the 3/8-Rule-ERK
            % method.
            %
            % Reference: Kutta, Z. Math. Phys. 46:435--453 (1901).
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0; ...
                   0.33333333333333331, 0, 0, 0; ...
                   -0.33333333333333331, 1, 0, 0; ...
                   1, -1, 1, 0];
            B.b = [0.125; 0.375; 0.375; 0.125];
            B.c = [0; 0.33333333333333331; 0.66666666666666663; 1];
            B.p = 4;
        end

        function B = ERK44()
            % Usage: B = ERK44()
            %
            % Utility routine to return the ERK table corresponding to the ERK-4-4
            % method.
            %
            % Reference: Kutta, Z. Math. Phys. 46:435--453 (1901).
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0; ...
                   0.5, 0, 0, 0; ...
                   0, 0.5, 0, 0; ...
                   0, 0, 1, 0];
            B.b = [0.16666666666666666; 0.33333333333333331; 0.33333333333333331; 0.16666666666666666];
            B.c = [0; 0.5; 0.5; 1];
            B.p = 4;
        end

        function B = Cooper6ERK()
            % Usage: B = Cooper6ERK()
            %
            % Utility routine to return the ERK table corresponding to the Cooper6-ERK
            % method.
            %
            % Reference: the first of the order-4 methods in Cooper & Sayfy, Math. Comp.
            %            40 (1983), doi:10.1090/S0025-5718-1983-0679441-1.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.5, 0, 0, 0, 0, 0; ...
                   0.25, 0.25, 0, 0, 0, 0; ...
                   0.25, 0.25, 0, 0, 0, 0; ...
                   0, -1, 0, 2, 0, 0; ...
                   0.16666666666666666, 0, 0, 0.66666666666666663, 0.16666666666666666, 0];
            B.b = [0.16666666666666666; 0; 0; 0.66666666666666663; 0.16666666666666666; 0];
            B.c = [0; 0.5; 0.5; 0.5; 1; 1];
            B.p = 4;
        end

        function B = Butcher76ERK()
            % Usage: B = Butcher76ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % Butcher-7-6-ERK method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0, 0, 0, 0; ...
                   0.33333333333333331, 0, 0, 0, 0, 0, 0; ...
                   0, 0.66666666666666663, 0, 0, 0, 0, 0; ...
                   0.083333333333333329, 0.33333333333333331, -0.083333333333333329, 0, 0, 0, 0; ...
                   0.52083333333333337, -2.2916666666666665, 0.72916666666666663, 1.875, 0, 0, 0; ...
                   0.14999999999999999, -0.45833333333333331, -0.125, 0.5, 0.10000000000000001, 0, 0; ...
                   -1.0038461538461538, 2.5384615384615383, 0.27564102564102566, -3.0256410256410255, 0.1641025641025641, 2.0512820512820511, 0];
            B.b = [0.065000000000000002; 0; 0.27500000000000002; 0.27500000000000002; 0.16; 0.16; 0.065000000000000002];
            B.c = [0; 0.33333333333333331; 0.66666666666666663; 0.33333333333333331; 0.83333333333333337; 0.16666666666666666; 1];
            B.p = 6;
        end

        function B = Butcher76bERK()
            % Usage: B = Butcher76bERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % Butcher-7-6b-ERK method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0, 0, 0, 0; ...
                   0.40000000000000002, 0, 0, 0, 0, 0, 0; ...
                   0, 0.80000000000000004, 0, 0, 0, 0, 0; ...
                   0.11591220850480109, 0.15089163237311384, -0.044581618655692733, 0, 0, 0, 0; ...
                   -0.065185185185185179, -0.6518518518518519, 0.21652421652421652, 1.0338461538461539, 0, 0, 0; ...
                   0.19811320754716982, 0, -0.15239477503628446, -0.47024673439767778, 0.42452830188679247, 0, 0; ...
                   -0.51747532894736847, -1.4473684210526316, 0.33574772267206476, 0.071735829959514164, 1.489514802631579, 1.067845394736842, 0];
            B.b = [0; 0; 0.27544070512820512; 0.32187009419152274; 0.26905293367346939; 0.069010416666666671; 0.064625850340136057];
            B.c = [0; 0.40000000000000002; 0.80000000000000004; 0.22222222222222221; 0.53333333333333333; 0; 1];
            B.p = 6;
        end

        function B = Butcher97ERK()
            % Usage: B = Butcher97ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % Butcher-9-7-ERK method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.16666666666666666, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0, 0.33333333333333331, 0, 0, 0, 0, 0, 0, 0; ...
                   0.125, 0, 0.375, 0, 0, 0, 0, 0, 0; ...
                   0.11119459053343352, 0, 0.11269722013523667, -0.042073628850488355, 0, 0, 0, 0, 0; ...
                   -1.6625514403292181, 0, -6.2962962962962967, 2.3656672545561435, 6.2598471487360374, 0, 0, 0, 0; ...
                   1.0270720533111204, 0, 3.620991253644315, -1.1409531742726244, -3.0885345391801033, 0.43856726364014992, 0, 0, 0; ...
                   0.032467532467532464, 0, 0, 0.17810760667903525, -0.08904042386185243, -0.16436688311688311, 0.042832167832167832, 0, 0; ...
                   -3.53125, 0, -8.8636363636363633, 4.5714285714285712, 8.2039620535714288, -1.423828125, 0.73082386363636365, 1.3125, 0];
            B.b = [0; 0; 0; 0.30476190476190479; 0.28165080001017501; 0.094921875000000003; 0.22445245726495727; 0.05347222222222222; 0.040740740740740744];
            B.c = [0; 0.16666666666666666; 0.33333333333333331; 0.5; 0.18181818181818182; 0.66666666666666663; 0.8571428571428571; 0; 1];
            B.p = 7;
        end

        function B = CooperVerner118ERK()
            % Usage: B = CooperVerner118ERK()
            %
            % Utility routine to return the ERK table corresponding to the
            % CooperVerner-11-8-ERK method.
            %
            % Reference: Cooper & Verner, SIAM J. Numer. Anal. 9 (1972),
            %            doi:10.1137/0709037.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.14285714285714285, -0.21171150086599511, 0.89618119336284074, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.18550685351137902, 0, 0.57667147269560881, 0.065148509147000641, 0, 0, 0, 0, 0, 0, 0; ...
                   0.1996369936449133, 0, 0.37729376930432884, -0.46345538964060623, 0.38652462669136406, 0, 0, 0, 0, 0, 0; ...
                   0.1289862929772419, 0, -0.033025511314484911, -0.34970528631774239, 0.32851721314173604, 0.097900456159259519, 0, 0, 0, 0, 0; ...
                   0.071428571428571425, 0, 0, 0, 0.0020021659931149178, -0.011868683886786038, 0.1111111111111111, 0, 0, 0, 0; ...
                   0.03125, 0, 0, 0, -0.0090869611008205595, 0.15277777777777779, -0.63254616069590974, 0.95760534401895248, 0, 0, 0; ...
                   0.071428571428571425, 0, 0, 0, 0.1111111111111111, -0.63793135018526459, 2.0310831391668618, -1.8108630829377543, 1.0624984467704632, 0, 0; ...
                   0, 0, 0, 0, -0.55122056307272915, 2.4513804324169666, -7.1649515532313819, 7.5538404421202712, -2.2291582101947447, 0.94010945196161799, 0];
            B.b = [0.050000000000000003; 0; 0; 0; 0; 0; 0; 0.2722222222222222; 0.35555555555555557; 0.2722222222222222; 0.050000000000000003];
            B.c = [0; 0.5; 0.5; 0.82732683535398854; 0.82732683535398854; 0.5; 0.17267316464601143; 0.17267316464601143; 0.5; 0.82732683535398854; 1];
            B.p = 8;
        end

        function B = Ascher111ERK()
            % Usage: B = Ascher111ERK()
            %
            % Utility routine to return the ERK table corresponding to
            % the explicit component of the ARS(1,1,1) method.
            %
            % Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
            %            doi:10.1016/S0168-9274(97)00056-1.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0; ...
                   1, 0];
            B.b = [1; 0];
            B.c = [0; 1];
            B.p = 1;
        end

        function B = Ascher122ERK()
            % Usage: B = Ascher122ERK()
            %
            % Utility routine to return the ERK table corresponding to
            % the explicit component of the ARS(1,2,2) method.
            %
            % Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
            %            doi:10.1016/S0168-9274(97)00056-1.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0; ...
                   0.5, 0];
            B.b = [0; 1];
            B.c = [0; 0.5];
            B.p = 2;
        end

        function B = ARKCouplingERK3()
            % Usage: B = ARKCouplingERK3()
            %
            % Utility routine to return the ERK table corresponding to
            % the third-order explicit method from the ARK coupling example.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0, 0, 0, 0; ...
                   0.5, 0, 0, 0; ...
                   0, 0.5, 0, 0; ...
                   1, 0, 0, 0];
            B.b = [1/6; 0; 2/3; 1/6];
            B.c = [0; 0.5; 0.5; 1];
            B.p = 3;
        end

    end
end
