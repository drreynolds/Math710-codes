classdef AdaptDIRK < handle
    % AdaptDIRK.m
    %
    % Adaptive-stepsive diagonally-implicit Runge--Kutta solver class
    % implementation file.
    %
    % Also contains functions to return specific embedded DIRK Butcher tables.
    %
    % Class to perform adaptive stepsize time evolution of the IVP
    %      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
    % using an embedded diagonally-implicit Runge--Kutta (DIRK) time stepping
    % method.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Adaptive diagonally-implicit Runge--Kutta class
    %
    % The four required arguments when constructing a DIRK object are a
    % function for the IVP right-hand side, an implicit solver to use,
    % and a Butcher table:
    %     f = ODE RHS function with calling syntax f(t,y).
    %     y = MATLAB array with m entries.
    %     sol = algebraic solver object to use [ImplicitSolver]
    %     B = diagonally-implicit embedded Butcher table.
    %     h = (optional) input with stepsize to use for time stepping.
    %         Note that this MUST be set either here or in the Evolve call.

    % Stored problem data, method coefficients, temporary vectors, and run statistics.
    properties
        f, sol, A, b, c, d, minpq, s
        rtol = 1e-3, atol = 1e-14, maxit = 1e6
        bias = 1.0, growth = 50.0, safety = 0.85, hmin = 10*eps
        ONEMSM = 1.0 - sqrt(eps), ONEPSM = 1.0 + sqrt(eps)
        fails = 0, steps = 0, nsol = 0, error_norm = 0.0, h = 0.0
        save_step_hist = false
        w, yerr, step_hist, z, yt, data, k
    end

    methods
        function self = AdaptDIRK(f, yTemplate, sol, B, rtol, atol, maxit, bias, growth, safety, hmin, save_step_hist)
            if nargin < 4
                error('AdaptDIRK requires f, a solution template, an implicit solver, and an embedded Butcher table.');
            end

            y = yTemplate(:);

            % Store the RHS, nonlinear solver, and embedded Butcher table.
            self.f = f;
            self.sol = sol;
            self.A = B.A;
            self.b = B.b(:);
            self.c = B.c(:);
            self.d = B.d(:);
            self.minpq = min(B.p, B.q);

            % Overwrite default adaptivity controls when supplied.
            if nargin >= 5 && ~isempty(rtol), self.rtol = rtol; end
            if nargin >= 6 && ~isempty(atol), self.atol = atol; end
            if nargin >= 7 && ~isempty(maxit), self.maxit = maxit; end
            if nargin >= 8 && ~isempty(bias), self.bias = bias; end
            if nargin >= 9 && ~isempty(growth), self.growth = growth; end
            if nargin >= 10 && ~isempty(safety), self.safety = safety; end
            if nargin >= 11 && ~isempty(hmin), self.hmin = hmin; end
            if nargin >= 12 && ~isempty(save_step_hist), self.save_step_hist = save_step_hist; end
            if isscalar(self.atol)
                self.atol = ones(numel(y),1)*self.atol;
            else
                self.atol = self.atol(:);
            end

            % Allocate work vectors and initialize run statistics.
            self.w = ones(numel(y),1);
            self.yerr = zeros(numel(y),1);
            self.step_hist = struct('t', [], 'h', [], 'err', []);
            self.z = zeros(numel(y),1);
            self.yt = zeros(numel(y),1);
            self.data = zeros(numel(y),1);
            self.s = numel(self.b);
            self.k = zeros(self.s, numel(y));

            % Embedded DIRK tables need matching sizes and no entries above the diagonal.
            if numel(self.c) ~= self.s || numel(self.d) ~= self.s || ...
                    size(self.A,1) ~= self.s || size(self.A,2) ~= self.s || ...
                    norm(self.b-self.d, inf) < 1e-14 || ...
                    norm(self.A - tril(self.A,0), inf) > 1e-14
                error('AdaptDIRK: incompatible embedded Butcher table supplied');
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
            % Utility routine to take a single diagonally-implicit RK time step,
            % where the inputs (t,y) are overwritten by the updated versions.
            % args is used for optional parameters of the RHS.
            % If success==true then the step succeeded; otherwise it failed.

            if nargin < 4
                args = {};
            end
            if ~iscell(args)
                error('AdaptDIRK: args must be a cell array.');
            end

            y = y(:);
            % Solve one implicit stage at a time, using previous stages explicitly.
            for i = 1:self.s
                self.data = y;
                for j = 1:(i-1)
                    self.data = self.data + self.h * self.A(i,j) * self.k(j,:).';
                end

                tstage = t + self.h*self.c(i);
                gamma = self.h * self.A(i,i);
                % Define the stage residual for the unknown stage state z_i.
                F = @(zcur) zcur(:) - self.data(:) - gamma * self.f(tstage, zcur(:), args{:});
                % Tell the Newton solver to use I - gamma*J for this stage.
                self.sol.setup_linear_solver(tstage, -gamma, args);

                % Solve this implicit stage, then store its RHS value.
                [self.z, ~, success] = self.sol.solve(F, y);
                self.nsol = self.nsol + 1;
                if ~success
                    return;
                end

                self.k(i,:) = self.f(tstage, self.z, args{:}).';
            end

            % Combine stage RHS values with the higher-order output weights.
            for i = 1:self.s
                y = y + self.h * self.b(i) * self.k(i,:).';
            end

            % The embedded formulas differ only in their output weights.
            self.yerr(:) = 0.0;
            for i = 1:self.s
                self.yerr = self.yerr + self.h * (self.b(i)-self.d(i)) * self.k(i,:).';
            end
            % Measure the local error in the user-specified weighted norm.
            self.error_norm = max(norm(self.yerr.*self.w, inf), 1e-8);
            success = true;
        end

        function [Y, success] = Evolve(self, tspan, y0, h, args)
            % Usage: Y, success = Evolve(tspan, y0, h, args)
            %
            % The adaptive DIRK time step evolution routine
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
                error('AdaptDIRK:Evolve args must be a cell array.');
            end

            % Store the requested initial step size; zero means estimate it below.
            self.h = h;

            y = y0(:);
            % Initialize output storage, with the first row holding the initial condition.
            tspan = tspan(:);
            m = numel(y);
            N = numel(tspan)-1;

            Y = zeros(N+1, m);
            Y(1,:) = y.';
            t = tspan(1);

            % Reject decreasing output times before any steps are attempted.
            for n = 1:N
                if tspan(n+1) < tspan(n)
                    error('AdaptDIRK:Evolve illegal tspan');
                end
            end

            % set error weights for the initial solution
            self.w = self.error_weight(y);

            % Estimate an initial step size from the weighted RHS norm if none was supplied.
            if self.h == 0.0
                fn = self.f(t, y, args{:});
                self.error_norm = max(norm(fn(:).*self.w, inf), 1e-8);
                self.h = max(self.hmin, self.safety/self.error_norm);
            end

            % iterate over output times, filling the solution history
            for iout = 2:(N+1)
                % Take as many adaptive internal steps as needed to hit this output time.
                while (tspan(iout)-t) > sqrt(eps*tspan(iout))
                    if (self.steps + self.fails) > self.maxit
                        fprintf('AdaptDIRK: reached maximum iterations, returning with failure\n');
                        success = false;
                        return;
                    end

                    % Do not step beyond the next requested output time.
                    self.h = min(self.h, tspan(iout)-t);

                    % Take one trial DIRK step from the current accepted solution.
                    self.yt = y;
                    [~, self.yt, success] = self.step(t, self.yt, args);
                    if ~success
                        fprintf('AdaptDIRK::Evolve error in time step at t = %g\n', t);
                        return;
                    end

                    % Convert the weighted error estimate into the next step-size factor.
                    eta = self.safety * self.error_norm^(-1.0/(self.minpq+1));
                    eta = min(eta, self.growth);

                    if self.save_step_hist
                        % Record every trial step so accepted and rejected steps can be plotted.
                        self.step_hist.t(end+1,1) = t;
                        self.step_hist.h(end+1,1) = self.h;
                        self.step_hist.err(end+1,1) = self.error_norm;
                    end

                    % successful step: update solution and prepare the next trial
                    if self.error_norm < self.ONEPSM
                        t = t + self.h;
                        y = self.yt;
                        self.w = self.error_weight(y);
                        self.steps = self.steps + 1;
                        self.h = self.h * eta;
                    % failed step: reduce the stepsize and retry
                    else
                        self.fails = self.fails + 1;
                        if self.h > self.hmin
                            self.h = max(self.h * eta, self.hmin);
                        else
                            fprintf('AdaptDIRK: error test failed at h=hmin, returning with failure\n');
                            success = false;
                            return;
                        end
                    end
                end

                % Store the accepted solution at this requested output time.
                Y(iout,:) = y.';
            end

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

        function update_rhs(self, f)
            % Updates the RHS function (cannot change vector dimensions)
            self.f = f;
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

    methods (Static)
        function B = SDIRK21()
            % Usage: B = SDIRK21()
            %
            % Utility routine to return the SDIRK table corresponding to
            % an embedded method with order 2 and embedding order 1.
            %
            % Reference: base method (implicit portion of IMEX-SSP2(2,2,2)): Pareschi &
            %            Russo, J. Sci. Comput. 25 (2005), doi:10.1007/BF02728986.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            gamma = 1 - 1/sqrt(2);
            B.A = [gamma, 0; 1-2*gamma, gamma];
            B.b = [0.5; 0.5];
            B.c = [gamma; 1-gamma];
            B.d = [5/12; 7/12];
            B.p = 2;
            B.q = 1;
        end

        function B = ESDIRK324L2SA()
            % Usage: B = ESDIRK324L2SA()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ESDIRK3(2)4L[2]SA method.
            %
            % Reference: eqs. (27)-(28) of Kennedy & Carpenter, Appl. Numer. Math. 146
            %            (2019), doi:10.1016/j.apnum.2019.07.008.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            gamma = 0.43586652150845899941601945;
            gamma2 = gamma*gamma;
            gamma3 = gamma2*gamma;
            gamma4 = gamma3*gamma;
            gamma5 = gamma4*gamma;
            c3 = 3/5;

            a32 = c3*(c3 - 2*gamma)/(4*gamma);
            a31 = c3 - gamma - a32;

            b2 = (-2 + 3*c3 + 6*gamma*(1-c3))/(12*gamma*(c3 - 2*gamma));
            b3 = (1 - 6*gamma + 6*gamma2)/(3*c3*(c3 - 2*gamma));
            b1 = 1 - gamma - b2 - b3;

            d2 = c3*(-1 + 6*gamma - 24*gamma3 + 12*gamma4 - 6*gamma5)/(4*gamma*(2*gamma-c3)*(1 - 6*gamma + 6*gamma2)) ...
               + (3 - 27*gamma + 68*gamma2 - 55*gamma3 + 21*gamma4 - 6*gamma5)/(2*(2*gamma-c3)*(1 - 6*gamma + 6*gamma2));
            d3 = -gamma*(-2 + 21*gamma - 68*gamma2 + 79*gamma3 - 33*gamma4 + 12*gamma5)/(c3*(c3 - 2*gamma)*(1 - 6*gamma + 6*gamma2));
            d4 = -3*gamma2*(-1 + 4*gamma - 2*gamma2 + gamma3)/(1 - 6*gamma + 6*gamma2);
            d1 = 1 - d2 - d3 - d4;

            B.A = [0, 0, 0, 0; ...
                   gamma, gamma, 0, 0; ...
                   a31, a32, gamma, 0; ...
                   b1, b2, b3, gamma];
            B.b = [b1; b2; b3; gamma];
            B.c = [0; 2*gamma; c3; 1];
            B.d = [d1; d2; d3; d4];
            B.p = 3;
            B.q = 2;
        end

        function B = ESDIRK32()
            % Usage: B = ESDIRK32()
            %
            % Utility routine to return the an ESDIRK table corresponding to
            % a 5-stage, 3rd-order method with 2nd-order embedding.
            %
            % Reference: Table 10 of Kennedy & Carpenter, Diagonally implicit
            %            Runge--Kutta methods for ordinary differential equations. A
            %            review, NASA/TM-2016-219173 (2016).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0; ...
                   9/40, 9/40, 0, 0, 0; ...
                   9*(1+sqrt(2))/80, 9*(1+sqrt(2))/80, 9/40, 0, 0; ...
                   (22+15*sqrt(2))/80/(1+sqrt(2)), (22+15*sqrt(2))/80/(1+sqrt(2)), -7/40/(1+sqrt(2)), 9/40, 0; ...
                   (2398+1205*sqrt(2))/2835/(4+3*sqrt(2)), (2398+1205*sqrt(2))/2835/(4+3*sqrt(2)), -2374*(1+2*sqrt(2))/2835/(5+3*sqrt(2)), 5827/7560, 9/40];
            B.b = B.A(end,:).';
            B.c = [0; 9/20; 9*(2+sqrt(2))/40; 3/5; 1];
            B.d = [4555948517383/24713416420891; 4555948517383/24713416420891; -7107561914881/25547637784726; 30698249/44052120; 49563/233080];
            B.p = 3;
            B.q = 2;
        end

        function B = ESDIRK43()
            % Usage: B = ESDIRK43()
            %
            % Utility routine to return the an ESDIRK table corresponding to
            % a 7-stage, 4th-order method with 3rd-order embedding.
            %
            % Reference: Table 10 of Kennedy & Carpenter, Appl. Numer. Math. 146 (2019),
            %            doi:10.1016/j.apnum.2019.07.008.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            b = [0; -5649241495537/14093099002237; 5718691255176/6089204655961; 2199600963556/4241893152925; 8860614275765/11425531467341; -3696041814078/6641566663007; 1/8];
            b(1) = 1 - sum(b);
            c = [0; 1/4; 1200237871921/16391473681546; 1/2; 395/567; 89/126; 1];
            d = [0; -1517409284625/6267517876163; 8291371032348/12587291883523; 5328310281212/10646448185159; 5405006853541/7104492075037; -4254786582061/7445269677723; 19/140];
            d(1) = 1 - sum(d);
            A = [0, 0, 0, 0, 0, 0, 0; ...
                 0, 1/8, 0, 0, 0, 0, 0; ...
                 0, -39188347878/1513744654945, 1/8, 0, 0, 0, 0; ...
                 0, 1748874742213/5168247530883, -1748874742213/5795261096931, 1/8, 0, 0, 0; ...
                 0, -6429340993097/17896796106705, 9711656375562/10370074603625, 1137589605079/3216875020685, 1/8, 0, 0; ...
                 0, 405169606099/1734380148729, -264468840649/6105657584947, 118647369377/6233854714037, 683008737625/4934655825458, 1/8, 0; ...
                 b.'];
            for i = 2:6
                A(i,1) = c(i) - sum(A(i,:));
            end
            B.A = A;
            B.b = b;
            B.c = c;
            B.d = d;
            B.p = 4;
            B.q = 3;
        end

        function B = ESDIRK54()
            % Usage: B = ESDIRK54()
            %
            % Utility routine to return the an ESDIRK table corresponding to
            % a 7-stage, 5th-order method with 4th-order embedding.
            %
            % Reference: Table 11 of Kennedy & Carpenter, Appl. Numer. Math. 146 (2019),
            %            doi:10.1016/j.apnum.2019.07.008.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            c = [0; 46/125; 7121331996143/11335814405378; 49/353; 3706679970760/5295570149437; 347/382; 1];
            b = [0; -188593204321/4778616380481; 2809310203510/10304234040467; 1021729336898/2364210264653; 870612361811/2470410392208; -1307970675534/8059683598661; 23/125];
            b(1) = 1 - sum(b);
            d = [0; -582099335757/7214068459310; 615023338567/3362626566945; 3192122436311/6174152374399; 6156034052041/14430468657929; -1011318518279/9693750372484; 1914490192573/13754262428401];
            d(1) = 1 - sum(d);
            A = [0, 0, 0, 0, 0, 0, 0; ...
                 0, 23/125, 0, 0, 0, 0, 0; ...
                 0, 791020047304/3561426431547, 23/125, 0, 0, 0, 0; ...
                 0, -158159076358/11257294102345, -85517644447/5003708988389, 23/125, 0, 0, 0; ...
                 0, -1653327111580/4048416487981, 1514767744496/9099671765375, 14283835447591/12247432691556, 23/125, 0, 0; ...
                 0, -4540011970825/8418487046959, -1790937573418/7393406387169, 10819093665085/7266595846747, 4109463131231/7386972500302, 23/125, 0; ...
                 b.'];
            for i = 2:6
                A(i,1) = c(i) - sum(A(i,:));
            end
            B.A = A;
            B.b = b;
            B.c = c;
            B.d = d;
            B.p = 5;
            B.q = 4;
        end

        function B = ESDIRK843()
            % Usage: B = ESDIRK843()
            %
            % Utility routine to return the ESDIRK table corresponding to
            % a fourth-order method a semilinear order 3.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0; ...
                   31/125, 31/125, 0, 0, 0, 0, 0, 0; ...
                   3781/15500, 63/124, 31/125, 0, 0, 0, 0, 0; ...
                   -3882222210210885/75584786387396543, -3882222210210885/75584786387396543, 0, 31/125, 0, 0, 0, 0; ...
                   34038088698073943/122803704925069405, 33514318812866834/119213963756997001, 224887786579749/60595115130919582, 4253927007933940/115874777755193681, 31/125, 0, 0, 0; ...
                   137658149652207956/209706981851726679, 126492513018975825/128664872604952432, 11417678526293581/37223122310090063, -103353126478507816/135174297737314759, -5/7, 31/125, 0, 0; ...
                   117786594983325079/151727549241844762, 18526475144695067/21268081176298953, 7206985701555927/80976168939093068, -94099054066115167/95522373062038575, -26/27, 190069087194766309/197235632620571833, 31/125, 0; ...
                   -29602757552094/1071399797354437, 548139805377293/112676962442277364, 424515922983497/13913811815644881, 67814305287223931/162780150685834614, -41854401642916128/116143966895455495, 74958030483037457/95092867575812394, -21188129/211373000, 31/125];
            B.b = B.A(end,:).';
            B.c = sum(B.A, 2);
            B.d = B.A(end-1,:).';
            B.p = 4;
            B.q = 3;
        end

        function B = EDIRK1054()
            % Usage: B = EDIRK1054()
            %
            % Utility routine to return the EDDIRK table corresponding to
            % a 5th-order accurate method with semilinear order 4.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   23704630662296147/85251944606883963, 23704630662296147/85251944606883963, 0, 0, 0, 0, 0, 0, 0, 0; ...
                   -3828053375722559/66474452137650668, -3828053375722559/66474452137650668, 23704630662296147/85251944606883963, 0, 0, 0, 0, 0, 0, 0; ...
                   -1768624057837184/97907440837578655, -1768624057837184/97907440837578655, 32342625154963567/63657230586675651, 23704630662296147/85251944606883963, 0, 0, 0, 0, 0, 0; ...
                   38689688244273643/145529658745107532, 38689688244273643/145529658745107532, 5165440406558499/48318409929685292, 0, 23704630662296147/85251944606883963, 0, 0, 0, 0, 0; ...
                   -2048929420167937/62953617727398324, -2048929420167937/62953617727398324, -5242126029351595/74777062100226619, 0, 0, 23704630662296147/85251944606883963, 0, 0, 0, 0; ...
                   -14097385432048041/83960854026807536, -14097385432048041/83960854026807536, 102631915790147645/94805799854610222, -5088592904909032/78095945223394903, 4194495314217601/126041470655949480, -25019099907264765/59359248890957054, 23704630662296147/85251944606883963, 0, 0, 0; ...
                   -6666023823723632/60782333039950069, -6666023823723632/60782333039950069, 34950019030688054/58700988111542175, 16157137791883982/129136549150431411, -4633364877709012/111280515475374397, -44830020893037844/115925512623522465, -10683392218257989/83044047426145149, 23704630662296147/85251944606883963, 0, 0; ...
                   7974359957524127/41777369871990865, 7974359957524127/41777369871990865, 31122288307425661/48758046711807126, 124912727797607611/63031353902083464, -68608793789563332/113404568873370149, -95359235367355842/59441684582698261, -162051494287025479/83831556722521602, 148921337658127561/79960515909413127, 23704630662296147/85251944606883963, 0; ...
                   -10206283873289495/82173081853978556, -10206283873289495/82173081853978556, 0, -149692166756442484/122226586801197919, 59118216399459218/50501318642781983, 239399454367668061/196562057586860935, 62112483136249447/41672907966740429, -117867017378953048/115349226975194417, -40768154109170907/61569993216212962, 23704630662296147/85251944606883963];
            B.b = B.A(end,:).';
            B.c = sum(B.A, 2);
            B.d = [-7377933185266438/48541560297323275; -7377933185266438/48541560297323275; 0; 39662348139301097/87268009515385808; 68262268363872686/135528001467088899; 58007573143164457/132805403176109435; -28930654619832593/286820031950471742; 15187635870586502/50762060951677207; -247981587118689503/472717409267306743; 4/17];
            B.p = 5;
            B.q = 4;
        end
        % Additional embedded diagonally-implicit Runge--Kutta tables.

        function B = Ascher222SDIRK()
            % Usage: B = Ascher222SDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % Ascher(2,2,2)-SDIRK method.
            %
            % Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
            %            doi:10.1016/S0168-9274(97)00056-1.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0, 0.29289321881345243, 0; ...
                   0, 0.70710678118654757, 0.29289321881345243];
            B.b = [0; 0.70710678118654757; 0.29289321881345243];
            B.c = [0; 0.29289321881345243; 1];
            B.d = [0; 0.59999999999999998; 0.40000000000000002];
            B.p = 2;
            B.q = 1;
        end

        function B = SSP32DIRK()
            % Usage: B = SSP32DIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % SSP(3,2)-DIRK method.
            %
            % References: method: Pareschi & Russo, J. Sci. Comput. 25 (2005),
            %             doi:10.1007/BF02728986.
            %             embedding: SUNDIALS ARKODE documentation, table
            %             ARKODE_SSP_DIRK_3_1_2,
            %             https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html#c.ARKODE_SSP_DIRK_3_1_2.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.25, 0, 0; ...
                   0, 0.25, 0; ...
                   0.33333333333333331, 0.33333333333333331, 0.33333333333333331];
            B.b = [0.33333333333333331; 0.33333333333333331; 0.33333333333333331];
            B.c = [0.25; 0.25; 1];
            B.d = [0.23333333333333331; 0.33333333333333331; 0.43333333333333335];
            B.p = 2;
            B.q = 1;
        end

        function B = TRBDF2ESDIRK()
            % Usage: B = TRBDF2ESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % TRBDF2-ESDIRK method.
            %
            % Reference: Bank, Coughran, Fichtner, Grosse, Rose & Smith, IEEE Trans.
            %            Comput.-Aided Des. Integr. Circuits Syst. 4 (1985),
            %            doi:10.1109/TCAD.1985.1270142.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0.29289321881345243, 0.29289321881345243, 0; ...
                   0.35355339059327379, 0.35355339059327379, 0.29289321881345243];
            B.b = [0.35355339059327379; 0.35355339059327379; 0.29289321881345243];
            B.c = [0; 0.58578643762690485; 1];
            B.d = [0.21548220313557542; 0.6868867239266071; 0.097631072937817476];
            B.p = 2;
            B.q = 3;
        end

        function B = TRX2ESDIRK()
            % Usage: B = TRX2ESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % TRX2-ESDIRK method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0.25, 0.25, 0; ...
                   0.25, 0.5, 0.25];
            B.b = [0.25; 0.5; 0.25];
            B.c = [0; 0.5; 1];
            B.d = [0.16666666666666666; 0.66666666666666663; 0.16666666666666666];
            B.p = 2;
            B.q = 3;
        end

        function B = BillingtonSDIRK()
            % Usage: B = BillingtonSDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % Billington-SDIRK method.
            %
            % Reference: Billington, Type-insensitive codes for the solution of stiff
            %            and nonstiff systems of ordinary differential equations,
            %            Master's thesis, University of Manchester (1983).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.29289321881300001, 0, 0; ...
                   0.79898987322299997, 0.29289321881300001, 0; ...
                   0.74078922884099996, 0.25921077115899999, 0.29289321881300001];
            B.b = [0.74078922883999998; 0.25921077115899999; 0];
            B.c = [0.29289321881300001; 1.091883092037; 1.292893218813];
            B.d = [0.69166511599199998; 0.50359702988300004; -0.19526214587599999];
            B.p = 2;
            B.q = 3;
        end

        function B = SDIRK22()
            % Usage: B = SDIRK22()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % SDIRK-2-2 method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.29289321881345254, 0; ...
                   0.70710678118654746, 0.29289321881345254];
            B.b = [0.70710678118654746; 0.29289321881345254];
            B.c = [0.29289321881345254; 1];
            B.d = [-0.70710678118654746; 1.7071067811865475];
            B.p = 2;
            B.q = 1;
        end

        function B = SDIRK21Alt()
            % Usage: B = SDIRK21Alt()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % SDIRK-2-1 method.
            %
            % TODO: add citation
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [1, 0; ...
                   -1, 1];
            B.b = [0.5; 0.5];
            B.c = [1; 0];
            B.d = [1; 0];
            B.p = 2;
            B.q = 1;
        end

        function B = ARK232SDIRK()
            % Usage: B = ARK232SDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ARK(2,3,2)-SDIRK method.
            %
            % Reference: Giraldo, Kelly & Constantinescu, SIAM J. Sci. Comput. 35
            %            (2013), doi:10.1137/120876034.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0.29289321881345254, 0.29289321881345254, 0; ...
                   0.35355339059327373, 0.35355339059327373, 0.29289321881345254];
            B.b = [0.35355339059327373; 0.35355339059327373; 0.29289321881345254];
            B.c = [0; 0.58578643762690508; 1];
            B.d = [0.32322330470336313; 0.32322330470336313; 0.35355339059327373];
            B.p = 2;
            B.q = 1;
        end

        function B = SSP2332LspumSDIRK()
            % Usage: B = SSP2332LspumSDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % SSP2(3,3,2)-lspum-SDIRK method.
            %
            % Reference: Higueras, Happenhofer, Koch & Kupka, J. Comput. Appl. Math. 272
            %            (2014), doi:10.1016/j.cam.2014.05.011.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.18181818181818182, 0, 0; ...
                   0.44372294372294374, 0.18181818181818182, 0; ...
                   0.44004329004329007, 0.19090909090909092, 0.18181818181818182];
            B.b = [0.43636363636363634; 0.20000000000000001; 0.36363636363636365];
            B.c = [0.18181818181818182; 0.62554112554112551; 0.81277056277056281];
            B.d = [0.43160569105691055; 0.19718218773096821; 0.37121212121212122];
            B.p = 2;
            B.q = 1;
        end

        function B = GiraldoARK2ESDIRK()
            % Usage: B = GiraldoARK2ESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % Giraldo-ARK2-ESDIRK method.
            %
            % Reference: Giraldo, Kelly & Constantinescu, SIAM J. Sci. Comput. 35
            %            (2013), doi:10.1137/120876034.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0; ...
                   0.29289321881345254, 0.29289321881345254, 0; ...
                   0.35355339059327373, 0.35355339059327373, 0.29289321881345254];
            B.b = [0.35355339059327373; 0.35355339059327373; 0.29289321881345254];
            B.c = [0; 0.58578643762690485; 1];
            B.d = [0.32322330470336313; 0.32322330470336313; 0.35355339059327373];
            B.p = 2;
            B.q = 1;
        end

        function B = ARK324L2SAESDIRK()
            % Usage: B = ARK324L2SAESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ARK3(2)4L[2]SA-ESDIRK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 44 (2003),
            %            doi:10.1016/S0168-9274(02)00138-1.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0; ...
                   0.435866521508459, 0.435866521508459, 0, 0; ...
                   0.25764824606642722, -0.093514767574886248, 0.435866521508459, 0; ...
                   0.18764102434672383, -0.59529747357695495, 0.97178992772177208, 0.435866521508459];
            B.b = [0.18764102434672383; -0.59529747357695495; 0.97178992772177208; 0.435866521508459];
            B.c = [0; 0.87173304301691801; 0.59999999999999998; 1];
            B.d = [0.21474028622338914; -0.4851622638849391; 0.86872500252038753; 0.40169697514116243];
            B.p = 3;
            B.q = 2;
        end

        function B = SSP43ESDIRK()
            % Usage: B = SSP43ESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % SSP(4,3)-ESDIRK method.
            %
            % Reference: SUNDIALS ARKODE documentation, table ARKODE_ESDIRK_4_2_3,
            %            https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html#c.ARKODE_ESDIRK_4_2_3.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0; ...
                   0.064133478491540996, 0.435866521508459, 0, 0; ...
                   0.3964116613974677, 0.1677218170940733, 0.435866521508459, 0; ...
                   -0.49838612606484062, 1.3860128578277058, -0.82349325327132405, 0.435866521508459];
            B.b = [0.16666666666666666; 0.16666666666666666; 0.16666666666666666; 0.5];
            B.c = [0; 0.5; 1; 0.50000000000000022];
            B.d = [0.25; 0.25; 0.25; 0.25];
            B.p = 3;
            B.q = 2;
        end

        function B = Kvaerno423ESDIRK()
            % Usage: B = Kvaerno423ESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % Kvaerno(4,2,3)-ESDIRK method.
            %
            % Reference: Kvaerno, BIT Numer. Math. 44 (2004),
            %            doi:10.1023/B:BITN.0000046811.70614.38.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0; ...
                   0.43586652149999999, 0.43586652149999999, 0, 0; ...
                   0.49056338841910802, 0.073570090080892006, 0.43586652149999999, 0; ...
                   0.308809969973036, 1.4905633882541061, -1.2352398797271451, 0.43586652149999999];
            B.b = [0.308809969973036; 1.4905633882541061; -1.2352398797271451; 0.43586652149999999];
            B.c = [0; 0.87173304299999999; 1; 1];
            B.d = [0.49056338841910802; 0.073570090080892006; 0.43586652149999999; 0];
            B.p = 3;
            B.q = 2;
        end

        function B = ESDIRK32I5L2SA()
            % Usage: B = ESDIRK32I5L2SA()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ESDIRK3(2I)5L[2]SA method.
            %
            % Reference: Table 11 of Kennedy & Carpenter, Diagonally implicit
            %            Runge--Kutta methods for ordinary differential equations. A
            %            review, NASA/TM-2016-219173 (2016).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0; ...
                   0.22500000000000001, 0.22500000000000001, 0, 0, 0; ...
                   0.2638888888888889, 0.31111111111111112, 0.22500000000000001, 0, 0; ...
                   0.28967013888888887, 0.32361111111111113, 0.16171874999999999, 0.22500000000000001, 0; ...
                   0.21322176213480562, 0.32600479805448751, 0.53737799467613134, -0.30160455486542442, 0.22500000000000001];
            B.b = [0.21322176213480562; 0.32600479805448751; 0.53737799467613134; -0.30160455486542442; 0.22500000000000001];
            B.c = [0; 0.45000000000000001; 0.80000000000000004; 1; 1];
            B.d = [0.2272160877233341; 0.32556661956247879; 0.46861135758651284; -0.20520648979080866; 0.18381242491848293];
            B.p = 3;
            B.q = 2;
        end

        function B = ARK436L2SAESDIRK()
            % Usage: B = ARK436L2SAESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ARK4(3)6L[2]SA-ESDIRK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 44 (2003),
            %            doi:10.1016/S0168-9274(02)00138-1.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.25, 0.25, 0, 0, 0, 0; ...
                   0.13777600000000001, -0.055775999999999999, 0.25, 0, 0, 0; ...
                   0.14463686602698217, -0.22393190761334475, 0.44929504158636258, 0.25, 0, 0; ...
                   0.098258783283564771, -0.59154424281967044, 0.81012105382829958, 0.28316440570780599, 0.25, 0; ...
                   0.15791629516167136, 0, 0.18675894052400077, 0.68056529530933463, -0.27524053099500667, 0.25];
            B.b = [0.15791629516167136; 0; 0.18675894052400077; 0.68056529530933463; -0.27524053099500667; 0.25];
            B.c = [0; 0.5; 0.33200000000000002; 0.62; 0.84999999999999998; 1];
            B.d = [0.15471180076321217; 0; 0.18920519166068023; 0.70204537122892186; -0.31918739906357912; 0.27322503541076487];
            B.p = 4;
            B.q = 3;
        end

        function B = ESDIRK436L2SA()
            % Usage: B = ESDIRK436L2SA()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ESDIRK4(3)6L[2]SA method.
            %
            % Reference: Table 16 of Kennedy & Carpenter, Diagonally implicit
            %            Runge--Kutta methods for ordinary differential equations. A
            %            review, NASA/TM-2016-219173 (2016).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.25, 0.25, 0, 0, 0, 0; ...
                   -0.051776695296636893, -0.051776695296636893, 0.25, 0, 0, 0; ...
                   -0.076554608384557271, -0.076554608384557271, 0.52810921676911449, 0.25, 0, 0; ...
                   -0.7274063478261299, -0.7274063478261299, 1.5849950617406794, 0.65981763391158055, 0.25, 0; ...
                   -0.01558763503571651, -0.01558763503571651, 0.3876576709132033, 0.50177261957216313, -0.10825502041393352, 0.25];
            B.b = [-0.01558763503571651; -0.01558763503571651; 0.3876576709132033; 0.50177261957216313; -0.10825502041393352; 0.25];
            B.c = [0; 0.5; 0.14644660940672621; 0.625; 1.04; 1];
            B.d = [-0.096513342168180333; -0.096513342168180333; 0.52281995099623424; 0.52056786462218851; -0.08255805440762122; 0.23219692312555915];
            B.p = 4;
            B.q = 3;
        end

        function B = ESDIRK43I6L2SA()
            % Usage: B = ESDIRK43I6L2SA()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ESDIRK4(3I)6L[2]SA method.
            %
            % Reference: Table 17 of Kennedy & Carpenter, Diagonally implicit
            %            Runge--Kutta methods for ordinary differential equations. A
            %            review, NASA/TM-2016-219173 (2016).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.25, 0.25, 0, 0, 0, 0; ...
                   -0.051776695296636879, -0.051776695296636879, 0.25, 0, 0, 0; ...
                   -0.12100710522809056, -0.12100710522809056, 0.57237336777017467, 0.25, 0, 0; ...
                   -0.61195040173360549, -0.61195040173360549, 1.3587904035004925, 0.61511039996671835, 0.25, 0; ...
                   0.066324612081910969, 0.066324612081910969, 0.22236269423069369, 0.50213093874834147, -0.10714285714285714, 0.25];
            B.b = [0.066324612081910969; 0.066324612081910969; 0.22236269423069369; 0.50213093874834147; -0.10714285714285714; 0.25];
            B.c = [0; 0.5; 0.14644660940672621; 0.58035915731399346; 1; 1];
            B.d = [-0.018437906120612999; -0.018437906120612999; 0.36437953464320771; 0.51624973045226041; -0.06251150951414057; 0.21875805665989839];
            B.p = 4;
            B.q = 3;
        end

        function B = QESDIRK436L2SA()
            % Usage: B = QESDIRK436L2SA()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % QESDIRK4(3)6L[2]SA method.
            %
            % Reference: Table 20 of Kennedy & Carpenter, Diagonally implicit
            %            Runge--Kutta methods for ordinary differential equations. A
            %            review, NASA/TM-2016-219173 (2016).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.10666666666666667, 0.10666666666666667, 0, 0, 0, 0; ...
                   0.066274169979695208, -0.19882250993908562, 0.32000000000000001, 0, 0, 0; ...
                   -0.48860873691193951, -6.1907515200673577, 6.8830952208408389, 0.32000000000000001, 0, 0; ...
                   -3.1283269307169475, -32.206585572984054, 35.100563192176956, 0.95811761036506327, 0.32000000000000001, 0; ...
                   0.11507824627133408, 0, 0.13891173144118288, 0.5589929940192544, -0.1329829717317714, 0.32000000000000001];
            B.b = [0.11507824627133408; 0; 0.13891173144118288; 0.5589929940192544; -0.1329829717317714; 0.32000000000000001];
            B.c = [0; 0.21333333333333335; 0.18745166004060956; 0.52373496386154128; 1.0437682988410166; 1];
            B.d = [-0.044357100747042959; -1.5839347397183428; 1.8584346081654304; 0.57804067934327363; -0.11463186751229457; 0.30644842046897641];
            B.p = 4;
            B.q = 3;
        end

        function B = ARK437L2SAESDIRK()
            % Usage: B = ARK437L2SAESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ARK4(3)7L[2]SA-ESDIRK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 136 (2019),
            %            doi:10.1016/j.apnum.2018.10.007.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0; ...
                   0.1235, 0.1235, 0, 0, 0, 0, 0; ...
                   0.14907768747653863, 0.14907768747653863, 0.1235, 0, 0, 0, 0; ...
                   0.12483442871739439, 0.12483442871739439, -0.038168857434788782, 0.1235, 0, 0, 0; ...
                   -0.073031940302180909, -0.073031940302180909, -0.24343568716014671, 0.34099956776450852, 0.1235, 0, 0; ...
                   -0.15296500088128806, -0.15296500088128806, 0.072205620474335874, 0.40430630248551713, 0.40591807880272318, 0.1235, 0; ...
                   0, 0, 0.51611072831742366, -0.14606356393857081, 0.23473048589019332, 0.27172234973095377, 0.1235];
            B.b = [0; 0; 0.51611072831742366; -0.14606356393857081; 0.23473048589019332; 0.27172234973095377; 0.1235];
            B.c = [0; 0.247; 0.42165537495307726; 0.33500000000000002; 0.074999999999999997; 0.69999999999999996; 1];
            B.d = [0; 0; 0.51752174615934821; -0.15173820706113939; 0.23672007870234135; 0.27544638219944978; 0.12205000000000001];
            B.p = 4;
            B.q = 3;
        end

        function B = Cash524SDIRK()
            % Usage: B = Cash524SDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % Cash(5,2,4)-SDIRK method.
            %
            % Reference: Cash, IMA J. Appl. Math. 24 (1979),
            %            doi:10.1093/imamat/24.3.293.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.43586652150799998, 0, 0, 0, 0; ...
                   -1.1358665214999999, 0.43586652150799998, 0, 0, 0; ...
                   1.0854333067899999, -0.72129982828700001, 0.43586652150799998, 0, 0; ...
                   0.41634950154700001, 0.19098400418399999, -0.118643265417, 0.43586652150799998, 0; ...
                   0.89686965294400001, 0.018272527273400001, -0.084590031070599994, -0.26641867064699998, 0.43586652150799998];
            B.b = [0.89686965294400001; 0.018272527273400001; -0.084590031070599994; -0.26641867064699998; 0.43586652150799998];
            B.c = [0.43586652150799998; -0.69999999999999996; 0.80000000000000004; 0.92455676181400004; 1];
            B.d = [1.0564621610705236; -0.05646216107052357; 0; 0; 0];
            B.p = 4;
            B.q = 2;
        end

        function B = Cash534SDIRK()
            % Usage: B = Cash534SDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % Cash(5,3,4)-SDIRK method.
            %
            % Reference: Cash, IMA J. Appl. Math. 24 (1979),
            %            doi:10.1093/imamat/24.3.293.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.43586652150799998, 0, 0, 0, 0; ...
                   -1.1358665214999999, 0.43586652150799998, 0, 0, 0; ...
                   1.0854333067899999, -0.72129982828700001, 0.43586652150799998, 0, 0; ...
                   0.41634950154700001, 0.19098400418399999, -0.118643265417, 0.43586652150799998, 0; ...
                   0.89686965294400001, 0.018272527273400001, -0.084590031070599994, -0.26641867064699998, 0.43586652150799998];
            B.b = [0.89686965294400001; 0.018272527273400001; -0.084590031070599994; -0.26641867064699998; 0.43586652150799998];
            B.c = [0.43586652150799998; -0.69999999999999996; 0.80000000000000004; 0.92455676181400004; 1];
            B.d = [0.77669193291000005; 0.029747279148399999; -0.026744023907400001; 0.220304811849; 0];
            B.p = 4;
            B.q = 3;
        end

        function B = Kvaerno534ESDIRK()
            % Usage: B = Kvaerno534ESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % Kvaerno(5,3,4)-ESDIRK method.
            %
            % Reference: Kvaerno, BIT Numer. Math. 44 (2004),
            %            doi:10.1023/B:BITN.0000046811.70614.38.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0; ...
                   0.43586652149999999, 0.43586652149999999, 0, 0, 0; ...
                   0.140737774731968, -0.108365551378832, 0.43586652149999999, 0, 0; ...
                   0.102399400616089, -0.37687845226732403, 0.83861253015123305, 0.43586652149999999, 0; ...
                   0.15702489786099499, 0.117330441357768, 0.61667803039168001, -0.32689989111044399, 0.43586652149999999];
            B.b = [0.15702489786099499; 0.117330441357768; 0.61667803039168001; -0.32689989111044399; 0.43586652149999999];
            B.c = [0; 0.87173304299999999; 0.46823874485313599; 1; 1];
            B.d = [0.102399400616089; -0.37687845226732403; 0.83861253015123305; 0.43586652149999999; 0];
            B.p = 4;
            B.q = 3;
        end

        function B = SDIRK54Table()
            % Usage: B = SDIRK54Table()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % SDIRK-5-4 method.
            %
            % Reference: Hairer & Wanner, Solving Ordinary Differential Equations II,
            %            2nd ed., Springer (1996), doi:10.1007/978-3-642-05221-7.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0.25, 0, 0, 0, 0; ...
                   0.5, 0.25, 0, 0, 0; ...
                   0.34000000000000002, -0.040000000000000001, 0.25, 0, 0; ...
                   0.2727941176470588, -0.050367647058823531, 0.027573529411764705, 0.25, 0; ...
                   1.0416666666666667, -1.0208333333333333, 7.8125, -7.083333333333333, 0.25];
            B.b = [1.0416666666666667; -1.0208333333333333; 7.8125; -7.083333333333333; 0.25];
            B.c = [0.25; 0.75; 0.55000000000000004; 0.5; 1];
            B.d = [1.2291666666666667; -0.17708333333333334; 7.03125; -7.083333333333333; 0];
            B.p = 4;
            B.q = 3;
        end

        function B = ESDIRK536L2SA()
            % Usage: B = ESDIRK536L2SA()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ESDIRK5(3)6L[2]SA method.
            %
            % Reference: Table 23 of Kennedy & Carpenter, Diagonally implicit
            %            Runge--Kutta methods for ordinary differential equations. A
            %            review, NASA/TM-2016-219173 (2016).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0; ...
                   0.27805384113645232, 0.27805384113645232, 0, 0, 0, 0; ...
                   0.31357495979254918, 0.43697244227949977, 0.27805384113645232, 0, 0, 0; ...
                   -0.094561054726895424, -0.13349472881973173, 0.050001942410174825, 0.27805384113645232, 0, 0; ...
                   -0.22208425034346749, -0.097104320164378782, 0.03123381024104438, 0.069900919130349556, 0.27805384113645232, 0; ...
                   -0.89887281059866253, 0.61640883439991601, -0.12228334655798583, -1.566608399277722, 2.693301880898002, 0.27805384113645232];
            B.b = [-0.89887281059866253; 0.61640883439991601; -0.12228334655798583; -1.566608399277722; 2.693301880898002; 0.27805384113645232];
            B.c = [0; 0.55610768227290464; 1.0286012432085012; 0.10000000000000001; 0.059999999999999998; 1];
            B.d = [-0.44739555875238668; 0.70955275848897481; -0.11977427137823622; -1.3819934098208657; 1.9923392723510966; 0.24727120911141712];
            B.p = 5;
            B.q = 3;
        end

        function B = ESDIRK547L2SA()
            % Usage: B = ESDIRK547L2SA()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ESDIRK5(4)7L[2]SA method.
            %
            % Reference: Table 25 of Kennedy & Carpenter, Diagonally implicit
            %            Runge--Kutta methods for ordinary differential equations. A
            %            review, NASA/TM-2016-219173 (2016).
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0; ...
                   0.184, 0.184, 0, 0, 0, 0, 0; ...
                   -0.038107647738324729, -0.038107647738324743, 0.184, 0, 0, 0, 0; ...
                   0.021677664958778542, 0.0216776649587785, 0.29264467008244299, 0.184, 0, 0, 0; ...
                   -0.85104626617351575, -0.85104626617351564, 1.7533038157326979, 0.41794699347257747, 0.184, 0, 0; ...
                   -5.0356161217492197, -5.0356161217492197, 8.9713052937951279, 0.31505839963851934, 1.6408685500647917, 0.184, 0; ...
                   -0.075998114543861517, -0.075998114543861378, 0.42427748359919076, 0.27546898147535387, 0.32051077889797169, -0.052261014884793552, 0.184];
            B.b = [-0.075998114543861517; -0.075998114543861378; 0.42427748359919076; 0.27546898147535387; 0.32051077889797169; -0.052261014884793552; 0.184];
            B.c = [0; 0.36799999999999999; 0.10778470452335051; 0.52000000000000002; 0.6531582768582439; 1.04; 1];
            B.d = [-0.108049345454303; -0.10804934545430295; 0.48372757888653789; 0.23595105756244605; 0.37538336433425512; -0.032306662513724778; 0.15334335263909166];
            B.p = 5;
            B.q = 4;
        end

        function B = ARK548L2SAESDIRK()
            % Usage: B = ARK548L2SAESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ARK5(4)8L[2]SA-ESDIRK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 44 (2003),
            %            doi:10.1016/S0168-9274(02)00138-1.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.20499999999999999, 0.20499999999999999, 0, 0, 0, 0, 0, 0; ...
                   0.10249999999999999, -0.047570415551619845, 0.20499999999999999, 0, 0, 0, 0, 0; ...
                   0.073899440792006915, 0, -0.080748954099503292, 0.20499999999999999, 0, 0, 0, 0; ...
                   0.29921811830801498, 0, 2.4638206661140414, -2.0480387844220567, 0.20499999999999999, 0, 0, 0; ...
                   0.14689238442881303, 0, 0.11740332879881549, -0.22170196800245401, -0.0075937452251744813, 0.20499999999999999, 0, 0; ...
                   0.17845729560319554, 0, 1.0197467452199207, -0.22154535039396367, -0.036124916205265319, -0.54553377422388716, 0.20499999999999999, 0; ...
                   -0.09554858675139874, 0, 0, 2.3386928037652464, -0.14043175608247527, -2.0705877079565589, 0.76287524702518661, 0.20499999999999999];
            B.b = [-0.09554858675139874; 0; 0; 2.3386928037652464; -0.14043175608247527; -2.0705877079565589; 0.76287524702518661; 0.20499999999999999];
            B.c = [0; 0.40999999999999998; 0.25992958444838016; 0.19815048669250362; 0.92000000000000004; 0.23999999999999999; 0.59999999999999998; 1];
            B.d = [-0.09957696480500873; 0; 0; 2.4071628799997749; -0.1601481830855136; -2.1442365964445265; 0.77956562242499827; 0.21723324191027585];
            B.p = 5;
            B.q = 4;
        end

        function B = ARK548L2SAbESDIRK()
            % Usage: B = ARK548L2SAbESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % ARK5(4)8L[2]SAb-ESDIRK method.
            %
            % Reference: Kennedy & Carpenter, Appl. Numer. Math. 136 (2019),
            %            doi:10.1016/j.apnum.2018.10.007.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0, 0; ...
                   0.22222222222222221, 0.22222222222222221, 0, 0, 0, 0, 0, 0; ...
                   0.26824595137478835, 0.26824595137478835, 0.22222222222222221, 0, 0, 0, 0, 0; ...
                   -0.057945592237231995, -0.057945592237231995, 0.0089383968162837328, 0.22222222222222221, 0, 0, 0, 0; ...
                   -0.043305287723547685, -0.043305287723547685, -0.034013891077568637, 0.25515937270676026, 0.22222222222222221, 0, 0, 0; ...
                   0.13179599023759678, 0.13179599023759678, -0.032376726277862332, 0.12385474427672251, 0.14270777930372408, 0.22222222222222221, 0, 0; ...
                   0.30932282100434261, 0.30932282100434261, -0.68291992723367922, -0.058822756149695461, -0.041308613833499437, 0.89718343298596659, 0.22222222222222221, 0; ...
                   0, 0, 0.17366253573581261, 0.25479166260812353, 0.24190176845094791, 0.30740485830222825, -0.19998304731933453, 0.22222222222222221];
            B.b = [0; 0; 0.17366253573581261; 0.25479166260812353; 0.24190176845094791; 0.30740485830222825; -0.19998304731933453; 0.22222222222222221];
            B.c = [0; 0.44444444444444442; 0.75871412497179891; 0.11526943456404197; 0.3567571284043185; 0.71999999999999997; 0.95499999999999996; 1];
            B.d = [0; 0; 0.062724216952707135; 0.25523315714677963; 0.23902754916001318; 0.39907952207535802; -0.14315725125850667; 0.18709280592364871];
            B.p = 5;
            B.q = 4;
        end

        function B = Kvaerno745ESDIRK()
            % Usage: B = Kvaerno745ESDIRK()
            %
            % Utility routine to return the embedded DIRK table corresponding to the
            % Kvaerno(7,4,5)-ESDIRK method.
            %
            % Reference: Kvaerno, BIT Numer. Math. 44 (2004),
            %            doi:10.1023/B:BITN.0000046811.70614.38.
            %
            % Outputs: B.A holds the stage coefficients
            %          B.b holds the solution weights
            %          B.c holds the abscissae
            %          B.d holds the embedding weights
            %          B.p holds the method order
            %          B.q holds the embedding order

            B.A = [0, 0, 0, 0, 0, 0, 0; ...
                   0.26000000000000001, 0.26000000000000001, 0, 0, 0, 0, 0; ...
                   0.13, 0.84033320996790806, 0.26000000000000001, 0, 0, 0, 0; ...
                   0.22371961478320504, 0.47675532319799702, -0.064708953631126151, 0.26000000000000001, 0, 0, 0; ...
                   0.16648564323248322, 0.1045001884159172, 0.036314822720987149, -0.13090704451073998, 0.26000000000000001, 0, 0; ...
                   0.13855640231268224, 0, -0.042453372017520433, 0.024466578980031409, 0.61943039072480677, 0.26000000000000001, 0; ...
                   0.13659751177640292, 0, -0.054969087965383759, -0.041186267283210461, 0.629933048990164, 0.069624794482027283, 0.26000000000000001];
            B.b = [0.13659751177640292; 0; -0.054969087965383759; -0.041186267283210461; 0.629933048990164; 0.069624794482027283; 0.26000000000000001];
            B.c = [0; 0.52000000000000002; 1.2303332099679081; 0.89576598435007604; 0.436393609858648; 1; 1];
            B.d = [0.13855640231268224; 0; -0.042453372017520433; 0.024466578980031409; 0.61943039072480677; 0.26000000000000001; 0];
            B.p = 5;
            B.q = 4;
        end

    end
end
