classdef IRK < handle
    % IRK.m
    %
    % Fixed-stepsize fully-implicit Runge--Kutta stepper class
    % implementation file.
    %
    % Also contains functions to return specific IRK Butcher tables.
    %
    % Class to perform fixed-stepsize time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using a fully-implicit Runge--Kutta (IRK) time stepping
    % method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Fixed stepsize fully-implicit Runge--Kutta class
    %
    % The five required arguments when constructing an IRK object are a
    % function for the IVP right-hand side, an implicit solver to use,
    % and a Butcher table:
    %     f = ODE RHS function with calling syntax f(t,y).
    %     sol = algebraic solver object to use [ImplicitSolver]
    %     A = Runge--Kutta stage coefficients (s*s matrix)
    %     b = Runge--Kutta solution weights (s array)
    %     c = Runge--Kutta abscissae (s array).
    %     h = (optional) input with stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        f, sol, A, b, c, s
        h = 0.0
        steps = 0
        nsol = 0
        k, z
    end

    methods
        function self = IRK(f, sol, B, h)
            if nargin < 3
                error('IRK requires f, implicit solver object, and a Butcher table.');
            end

            % Store the RHS, nonlinear solver, and Butcher table.
            self.f = f;
            self.sol = sol;
            self.A = B.A;
            self.b = B.b(:);
            self.c = B.c(:);
            % optional inputs
            if nargin >= 4 && ~isempty(h), self.h = h; end
            self.s = numel(self.c);

            % Fully implicit RK tables may be dense, but the dimensions must agree.
            if numel(self.b) ~= self.s || size(self.A,1) ~= self.s || size(self.A,2) ~= self.s
                error('IRK: incompatible Butcher table supplied');
            end
        end

        function [t, y, success] = irk_step(self, t, y, args)
            % Usage: t, y, success = irk_step(t, y, args)
            %
            % Utility routine to take a single fully-implicit RK time step,
            % where the inputs (t,y) are overwritten by the updated versions.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 4
                args = {};
            end
            if ~iscell(args)
                error('IRK: args must be a cell array.');
            end

            y = y(:);
            m = numel(y);
            s = self.s;
            h = self.h;

            % Define the coupled residual for all stage states at once.
            F = @(z) self.irkResidual(z, t, y, h, args);
            % Build the corresponding coupled Newton linear solver on demand.
            self.sol.linear_solver = @(z) self.irkLinearSolver(z, t, h, args);

            % Use the old solution repeated at each stage as the nonlinear initial guess.
            for i = 1:s
                self.z((m*(i-1)+1):(m*i)) = y;
            end

            % Solve all coupled implicit stages simultaneously.
            [self.z, ~, success] = self.sol.solve(F, self.z);
            self.nsol = self.nsol + 1;
            if ~success
                return;
            end

            % Evaluate and store RHS values at the converged stage states.
            self.k(:,:) = 0.0;
            for j = 1:s
                idx = (m*(j-1)+1):(m*j);
                tj = t + self.c(j)*h;
                zj = self.z(idx);
                self.k(j,:) = self.f(tj, zj, args{:}).';
            end

            % Combine stage RHS values with the RK weights to update the solution.
            for i = 1:s
                y = y + h * self.b(i) * self.k(i,:).';
            end
            t = t + h;
            self.steps = self.steps + 1;
            success = true;
        end

        function reset(self)
            % Resets the accumulated number of steps

            % reset accumulated statistics
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
            % The fixed-step IRK evolution routine.
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
                error('IRK:Evolve args must be a cell array.');
            end

            % update stored stepsize when requested by the caller
            if h ~= 0.0
                self.h = h;
            end
            % require a nonzero stepsize before evolving
            if self.h == 0.0
                error('IRK:Evolve called without specifying a nonzero step size');
            end

            % Verify that each output interval can be reached by an integer number of steps.
            tspan = tspan(:);
            for n = 1:(numel(tspan)-1)
                hn = tspan(n+1)-tspan(n);
                if abs(round(hn/self.h) - (hn/self.h)) > 100*sqrt(eps)*abs(self.h)
                    error('input values in tspan (%e,%e) are not separated by a multiple of h = %e', tspan(n), tspan(n+1), self.h);
                end
            end

            % Initialize output storage, with the first row holding the initial condition.
            y = y0(:);
            nout = numel(tspan);
            m = numel(y);
            Y = zeros(nout, m);
            Y(1,:) = y.';

            % Allocate stage and work vectors now that the solution dimension is known.
            self.k = zeros(self.s, m);
            self.z = zeros(self.s*m, 1);

            % iterate over output times, filling the solution history
            for iout = 2:nout
                % The divisibility check above allows this rounded internal step count.
                N = round((tspan(iout)-tspan(iout-1))/self.h);
                t = tspan(iout-1);

                % March internally until the next requested output time is reached.
                for n = 1:N %#ok<NASGU>
                    [t, y, success] = self.irk_step(t, y, args);
                    if ~success
                        fprintf('IRK::Evolve error in time step at t = %g\n', t);
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
        function B = RadauIIA2()
            % Usage: B = RadauIIA2()
            %
            % Utility routine to return the O(h^3) RadauIIA 2-stage IRK table.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [5.0/12.0, -1.0/12.0; 9.0/12.0, 3.0/12.0];
            B.b = [0.75; 0.25];
            B.c = [1.0/3.0; 1.0];
            B.p = 3;
        end

        function B = GaussLegendre2()
            % Usage: B = GaussLegendre2()
            %
            % Utility routine to return the O(h^4) Gauss-Legendre 2-stage IRK table.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0.25, (3.0-2.0*sqrt(3.0))/12.0; ...
                   (3.0+2.0*sqrt(3.0))/12.0, 0.25];
            B.b = [0.5; 0.5];
            B.c = [(3.0-sqrt(3.0))/6.0; (3.0+sqrt(3.0))/6.0];
            B.p = 4;
        end

        function B = RadauIIA3()
            % Usage: B = RadauIIA3()
            %
            % Utility routine to return the O(h^5) RadauIIA 3-stage IRK table.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [(88.0 - 7.0*sqrt(6.0))/360.0, (296.0 - 169.0*sqrt(6.0))/1800.0, (-2.0 + 3.0*sqrt(6.0))/225.0; ...
                   (296.0 + 169.0*sqrt(6.0))/1800.0, (88.0 + 7.0*sqrt(6.0))/360.0, (-2.0 - 3.0*sqrt(6.0))/225.0; ...
                   (16.0 - sqrt(6.0))/36.0, (16.0 + sqrt(6.0))/36.0, 1.0/9.0];
            B.b = [(16.0 - sqrt(6.0))/36.0; (16.0 + sqrt(6.0))/36.0; 1.0/9.0];
            B.c = [(4.0 - sqrt(6.0))/10.0; (4.0 + sqrt(6.0))/10.0; 1.0];
            B.p = 5;
        end

        function B = GaussLegendre3()
            % Usage: B = GaussLegendre3()
            %
            % Utility routine to return the O(h^6) Gauss-Legendre 3-stage IRK table.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [5.0/36.0, 2.0/9.0 - sqrt(15.0)/15.0, 5.0/36.0 - sqrt(15.0)/30.0; ...
                   5.0/36.0 + sqrt(15.0)/24.0, 2.0/9.0, 5.0/36.0 - sqrt(15.0)/24.0; ...
                   5.0/36.0 + sqrt(15.0)/30.0, 2.0/9.0 + sqrt(15.0)/15.0, 5.0/36.0];
            B.b = [5.0/18.0; 4.0/9.0; 5.0/18.0];
            B.c = [(5.0 - sqrt(15.0))/10.0; 0.5; (5.0 + sqrt(15.0))/10.0];
            B.p = 6;
        end

        function B = GaussLegendre6()
            % Usage: B = GaussLegendre6()
            %
            % Utility routine to return the O(h^12) Gauss-Legendre 6-stage IRK table.
            %
            % Outputs: B.A holds the Runge--Kutta stage coefficients
            %          B.b holds the Runge--Kutta solution weights
            %          B.c holds the Runge--Kutta abscissae
            %          B.p holds the Runge--Kutta method order

            B.A = [0.042831123094792580851996218950605, -0.014763725997197424643891429014278, 0.0093250507064777618411400734121424, -0.0056688580494835162182488917046817, 0.0028544333150993149102007359161104, -0.00081278017126476782600392067714199; ...
                   0.092673491430378856970823740288243, 0.090190393262034655662118827897123, -0.020300102293239581308124404430781, 0.010363156240246421640614877198502, -0.0048871929280376802268550750181669, 0.001355561055485051944941864725486; ...
                   0.082247922612843859526233540856659, 0.19603216233324501065540377853111, 0.11697848364317276194496135254516, -0.020482527745656096032756375665715, 0.007989991899662334513029865501749, -0.0020756257848663355105554732114538; ...
                   0.087737871974451497214547911112663, 0.1723907946244069768112077902925, 0.25443949503200161992267908075603, 0.11697848364317276194496135254516, -0.015651375809175699331166122736864, 0.00341432357674130217775889704455; ...
                   0.084306685134100109759050573175723, 0.18526797945210699155109273081241, 0.22359381104609910224930782789182, 0.2542570695795851051980471095211, 0.090190393262034655662118827897123, -0.007011245240793695266831302387034; ...
                   0.086475026360849929529996358578351, 0.17752635320896999641403691987814, 0.239625825335829040108171596795, 0.22463191657986776204878263167818, 0.19514451252126673596812908480852, 0.042831123094792580851996218950605];
            B.b = [0.085662246189585161703992437901209; 0.18038078652406931132423765579425; 0.23395696728634552388992270509032; 0.23395696728634552388992270509032; 0.18038078652406931132423765579425; 0.085662246189585161703992437901209];
            B.c = [0.0337652428984239749709672651079; 0.16939530676686775922945571437594; 0.38069040695840154764351126459587; 0.61930959304159845235648873540413; 0.83060469323313224077054428562406; 0.9662347571015760250290327348921];
            B.p = 12;
        end
        % Additional fully-implicit Runge--Kutta tables.

        function B = IRK11()
            % Utility routine to return the IRK table IRK-1-1.

            B.A = 1;
            B.b = 1;
            B.c = 1;
            B.p = 1;
        end

        function B = LobattoIIIC22IRK()
            % Utility routine to return the IRK table LobattoIIIC-2-2-IRK.

            B.A = [0.5, -0.5; ...
                   0.5, 0.5];
            B.b = [0.5; 0.5];
            B.c = [0; 1];
            B.p = 2;
        end

        function B = CrankNicolson22IRK()
            % Utility routine to return the IRK table Crank-Nicolson-2-2-IRK.

            B.A = [0.5, 0.5; ...
                   0, 0];
            B.b = [0.5; 0.5];
            B.c = [1; 0];
            B.p = 2;
        end

        function B = SIRK22()
            % Utility routine to return the IRK table SIRK-2-2.

            B.A = [0.18933982822017859, -0.017766952966368876; ...
                   0.60355339059327373, 0.39644660940672621];
            B.b = [0.60355339059327373; 0.39644660940672621];
            B.c = [0.17157287525380971; 1];
            B.p = 2;
        end

        function B = LobattoIIIA22IRK()
            % Utility routine to return the IRK table LobattoIIIA-2-2-IRK.

            B.A = [0, 0; ...
                   0.5, 0.5];
            B.b = [0.5; 0.5];
            B.c = [0; 1];
            B.p = 2;
        end

        function B = LobattoIII22IRK()
            % Utility routine to return the IRK table LobattoIII-2-2-IRK.

            B.A = [0, 0; ...
                   1, 0];
            B.b = [0.5; 0.5];
            B.c = [0; 1];
            B.p = 2;
        end

        function B = LobattoIIIC34IRK()
            % Utility routine to return the IRK table LobattoIIIC-3-4-IRK.

            B.A = [0.16666666666666666, -0.33333333333333331, 0.16666666666666666; ...
                   0.16666666666666666, 0.41666666666666669, -0.083333333333333329; ...
                   0.16666666666666666, 0.66666666666666663, 0.16666666666666666];
            B.b = [0.16666666666666666; 0.66666666666666663; 0.16666666666666666];
            B.c = [0; 0.5; 1];
            B.p = 4;
        end

        function B = LobattoIIIA34IRK()
            % Utility routine to return the IRK table LobattoIIIA-3-4-IRK.

            B.A = [0, 0, 0; ...
                   0.20833333333333334, 0.33333333333333331, -0.041666666666666664; ...
                   0.16666666666666666, 0.66666666666666663, 0.16666666666666666];
            B.b = [0.16666666666666666; 0.66666666666666663; 0.16666666666666666];
            B.c = [0; 0.5; 1];
            B.p = 4;
        end

        function B = LobattoIIIB34IRK()
            % Utility routine to return the IRK table LobattoIIIB-3-4-IRK.

            B.A = [0.16666666666666666, -0.16666666666666666, 0; ...
                   0.16666666666666666, 0.33333333333333331, 0; ...
                   0.16666666666666666, 0.83333333333333337, 0];
            B.b = [0.16666666666666666; 0.66666666666666663; 0.16666666666666666];
            B.c = [0; 0.5; 1];
            B.p = 4;
        end

        function B = LobattoIII34IRK()
            % Utility routine to return the IRK table LobattoIII-3-4-IRK.

            B.A = [0, 0, 0; ...
                   0.25, 0.25, 0; ...
                   0, 1, 0];
            B.b = [0.16666666666666666; 0.66666666666666663; 0.16666666666666666];
            B.c = [0; 0.5; 1];
            B.p = 4;
        end

        function B = RadauIA35IRK()
            % Utility routine to return the IRK table RadauIA-3-5-IRK.

            B.A = [0.1111111111111111, -0.19163831904350989, 0.080527207932398773; ...
                   0.1111111111111111, 0.29207341166522843, -0.048133497054657366; ...
                   0.1111111111111111, 0.53702238594354623, 0.19681547722366044];
            B.b = [0.1111111111111111; 0.51248582618842164; 0.37640306270046725];
            B.c = [0; 0.35505102572168223; 0.84494897427831783];
            B.p = 5;
        end

        function B = RadauI35IRK()
            % Utility routine to return the IRK table RadauI-3-5-IRK.

            B.A = [0, 0, 0; ...
                   0.15265986323710906, 0.22041241452319316, -0.018021252038619952; ...
                   0.087340136762890958, 0.57802125203861998, 0.17958758547680684];
            B.b = [0.1111111111111111; 0.51248582618842164; 0.37640306270046725];
            B.c = [0; 0.35505102572168223; 0.84494897427831783];
            B.p = 5;
        end

        function B = RadauII35IRK()
            % Utility routine to return the IRK table RadauII-3-5-IRK.

            B.A = [0.17958758547680684, -0.024536559755124632, 0; ...
                   0.42453655975512467, 0.22041241452319316, 0; ...
                   0.29587585476806849, 0.70412414523193156, 0];
            B.b = [0.37640306270046725; 0.51248582618842164; 0.1111111111111111];
            B.c = [0.15505102572168222; 0.64494897427831777; 1];
            B.p = 5;
        end

        function B = LobattoIIIC46IRK()
            % Utility routine to return the IRK table LobattoIIIC-4-6-IRK.

            B.A = [0.083333333333333329, -0.18633899812498247, 0.18633899812498247, -0.083333333333333329; ...
                   0.083333333333333329, 0.25, -0.09420793070830881, 0.037267799624996496; ...
                   0.083333333333333329, 0.42754126404164217, 0.25, -0.037267799624996496; ...
                   0.083333333333333329, 0.41666666666666669, 0.41666666666666669, 0.083333333333333329];
            B.b = [0.083333333333333329; 0.41666666666666669; 0.41666666666666669; 0.083333333333333329];
            B.c = [0; 0.27639320225002101; 0.72360679774997894; 1];
            B.p = 6;
        end

        function B = LobattoIIIA46IRK()
            % Utility routine to return the IRK table LobattoIIIA-4-6-IRK.

            B.A = [0, 0, 0, 0; ...
                   0.11030056647916492, 0.1896994335208351, -0.033907364229143894, 0.010300566479164915; ...
                   0.073032766854168416, 0.45057403089581055, 0.22696723314583159, -0.026967233145831583; ...
                   0.083333333333333329, 0.41666666666666669, 0.41666666666666669, 0.083333333333333329];
            B.b = [0.083333333333333329; 0.41666666666666669; 0.41666666666666669; 0.083333333333333329];
            B.c = [0; 0.27639320225002101; 0.72360679774997894; 1];
            B.p = 6;
        end

        function B = LobattoIIIB46IRK()
            % Utility routine to return the IRK table LobattoIIIB-4-6-IRK.

            B.A = [0.083333333333333329, -0.13483616572915791, 0.051502832395824573, 0; ...
                   0.083333333333333329, 0.22696723314583159, -0.033907364229143894, 0; ...
                   0.083333333333333329, 0.45057403089581055, 0.1896994335208351, 0; ...
                   0.083333333333333329, 0.36516383427084209, 0.55150283239582454, 0];
            B.b = [0.083333333333333329; 0.41666666666666669; 0.41666666666666669; 0.083333333333333329];
            B.c = [0; 0.27639320225002101; 0.72360679774997894; 1];
            B.p = 6;
        end

        function B = LobattoIII46IRK()
            % Utility routine to return the IRK table LobattoIII-4-6-IRK.

            B.A = [0, 0, 0, 0; ...
                   0.12060113295832983, 0.16666666666666666, -0.010874597374975477, 0; ...
                   0.046065533708336839, 0.51087459737497543, 0.16666666666666666, 0; ...
                   0.16666666666666666, 0.23032766854168418, 0.60300566479164919, 0];
            B.b = [0.083333333333333329; 0.41666666666666669; 0.41666666666666669; 0.083333333333333329];
            B.c = [0; 0.27639320225002101; 0.72360679774997894; 1];
            B.p = 6;
        end

        function B = RadauIIA47IRK()
            % Utility routine to return the IRK table RadauIIA-4-7-IRK.

            B.A = [0.11299947932315618, -0.040309220723522207, 0.025802377420336392, -0.0099046765072664245; ...
                   0.23438399574740026, 0.2068925739353589, -0.047857128048540719, 0.016047422806516273; ...
                   0.21668178462325033, 0.4061232638673733, 0.18903651817005634, -0.02418210489983294; ...
                   0.22046221117676837, 0.38819346884317191, 0.32884431998005975, 0.0625];
            B.b = [0.22046221117676837; 0.38819346884317191; 0.32884431998005975; 0.0625];
            B.c = [0.088587959512703943; 0.40946686444073471; 0.787659461760847; 1];
            B.p = 7;
        end

        function B = LobattoIIIC58IRK()
            % Utility routine to return the IRK table LobattoIIIC-5-8-IRK.

            B.A = [0.050000000000000003, -0.11666666666666667, 0.13333333333333333, -0.11666666666666667, 0.050000000000000003; ...
                   0.050000000000000003, 0.16111111111111112, -0.069011541029643186, 0.05200216599311492, -0.021428571428571429; ...
                   0.050000000000000003, 0.28130918332304278, 0.20277777777777778, -0.052836961100820548, 0.018749999999999999; ...
                   0.050000000000000003, 0.27022005622910727, 0.36742423944234159, 0.16111111111111112, -0.021428571428571429; ...
                   0.050000000000000003, 0.2722222222222222, 0.35555555555555557, 0.2722222222222222, 0.050000000000000003];
            B.b = [0.050000000000000003; 0.2722222222222222; 0.35555555555555557; 0.2722222222222222; 0.050000000000000003];
            B.c = [0; 0.17267316464601143; 0.5; 0.82732683535398854; 1];
            B.p = 8;
        end

        function B = LobattoIIIA58IRK()
            % Utility routine to return the IRK table LobattoIIIA-5-8-IRK.

            B.A = [0, 0, 0, 0, 0; ...
                   0.067728432186156887, 0.11974476934341167, -0.02173572186655812, 0.010635824225415506, -0.0037001392424145306; ...
                   0.040625000000000001, 0.30318418332304276, 0.17777777777777778, -0.030961961100820546, 0.0093749999999999997; ...
                   0.053700139242414534, 0.26158639799680672, 0.37729127742211366, 0.15247745287881054, -0.017728432186156898; ...
                   0.050000000000000003, 0.2722222222222222, 0.35555555555555557, 0.2722222222222222, 0.050000000000000003];
            B.b = [0.050000000000000003; 0.2722222222222222; 0.35555555555555557; 0.2722222222222222; 0.050000000000000003];
            B.c = [0; 0.17267316464601143; 0.5; 0.82732683535398854; 1];
            B.p = 8;
        end

        function B = LobattoIIIB58IRK()
            % Utility routine to return the IRK table LobattoIIIB-5-8-IRK.

            B.A = [0.050000000000000003, -0.096521464124631987, 0.066666666666666666, -0.020145202542034668, 0; ...
                   0.050000000000000003, 0.15247745287881054, -0.040440112458214612, 0.010635824225415506, 0; ...
                   0.050000000000000003, 0.28886363427630579, 0.17777777777777778, -0.016641412054083558, 0; ...
                   0.050000000000000003, 0.26158639799680672, 0.39599566801377017, 0.11974476934341167, 0; ...
                   0.050000000000000003, 0.29236742476425687, 0.28888888888888886, 0.36874368634685417, 0];
            B.b = [0.050000000000000003; 0.2722222222222222; 0.35555555555555557; 0.2722222222222222; 0.050000000000000003];
            B.c = [0; 0.17267316464601143; 0.5; 0.82732683535398854; 1];
            B.p = 8;
        end

        function B = LobattoIII58IRK()
            % Utility routine to return the IRK table LobattoIII-5-8-IRK.

            B.A = [0, 0, 0, 0, 0; ...
                   0.071428571428571425, 0.1111111111111111, -0.011868683886786038, 0.0020021659931149178, 0; ...
                   0.03125, 0.32505918332304279, 0.15277777777777779, -0.0090869611008205595, 0; ...
                   0.071428571428571425, 0.22022005622910731, 0.42456709658519876, 0.1111111111111111, 0; ...
                   0, 0.3888888888888889, 0.22222222222222221, 0.3888888888888889, 0];
            B.b = [0.050000000000000003; 0.2722222222222222; 0.35555555555555557; 0.2722222222222222; 0.050000000000000003];
            B.c = [0; 0.17267316464601143; 0.5; 0.82732683535398854; 1];
            B.p = 8;
        end

        function B = RadauIIA59IRK()
            % Utility routine to return the IRK table RadauIIA-5-9-IRK.

            B.A = [0.072998864317903367, -0.02673533110794565, 0.01867692976398445, -0.01287910609330652, 0.0050428392338820521; ...
                   0.1537752314791824, 0.14621486784749349, -0.036444568905128157, 0.021233063119304799, -0.0079355799027288135; ...
                   0.1400630456848099, 0.29896712949128329, 0.1675850701352492, -0.033969101686617938, 0.010944288744192329; ...
                   0.14489430810953419, 0.27650006876016081, 0.32579792291041909, 0.12875675325491151, -0.015708917378806069; ...
                   0.14371356079122591, 0.28135601514946212, 0.31182652297574132, 0.2231039010835707, 0.040000000000000001];
            B.b = [0.14371356079122591; 0.28135601514946212; 0.31182652297574132; 0.2231039010835707; 0.040000000000000001];
            B.c = [0.057104196114517683; 0.2768430136381238; 0.58359043236891683; 0.86024013565621948; 1];
            B.p = 9;
        end

    end

    methods (Access = private)
        function resid = irkResidual(self, z, t, y, h, args)
            s = self.s;
            m = numel(y);
            resid = z(:);
            for i = 1:s
                idx = (m*(i-1)+1):(m*i);
                resid(idx) = resid(idx) - y;
            end

            for j = 1:s
                jidx = (m*(j-1)+1):(m*j);
                tj = t + self.c(j)*h;
                zj = z(jidx);
                kj = self.f(tj, zj, args{:});
                self.k(j,:) = kj(:).';
                for i = 1:s
                    iidx = (m*(i-1)+1):(m*i);
                    resid(iidx) = resid(iidx) - h * self.A(i,j) * kj(:);
                end
            end
        end

        function Jsolve = irkLinearSolver(self, z, t, h, args)
            s = self.s;
            m = numel(z) / s;

            Jtest = self.sol.f_y(t + self.c(1)*h, z(1:m), args{:});
            if issparse(Jtest)
                Jac = speye(numel(z));
            else
                Jac = eye(numel(z));
            end

            for j = 1:s
                jidx = (m*(j-1)+1):(m*j);
                tj = t + self.c(j)*h;
                zj = z(jidx);
                if j == 1
                    Jj = Jtest;
                else
                    Jj = self.sol.f_y(tj, zj, args{:});
                end
                for i = 1:s
                    iidx = (m*(i-1)+1):(m*i);
                    Jac(iidx,jidx) = Jac(iidx,jidx) - h * self.A(i,j) * Jj;
                end
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
                error('IRK Jacobian factorization failure');
            end
        end
    end
end
