classdef AdvectionDiffusion
    % This file defines the RHS and Jacobian functions associated with the
    % viscous Burgers equation,
    %
    %    u_t + (u^2/2)_x = nu*u_xx,  x in [-1,1], t in [0,1],
    %    u(t,-1) = g(-1,t),  u(t,1) = g(1,t),
    %    u(0,x) = g(x,0),
    %
    % that has the traveling-wave analytical solution
    %
    %    g(x,t) = (1 - tanh((x - t/2)/(4*nu)))/2,
    %
    % a viscous shock connecting u=1 to u=0 that moves to the right with speed 1/2.
    % We discretize in space using centered, second-order differences on a uniform
    % mesh, with the Dirichlet boundary values taken from g.  So that computed
    % errors measure only the temporal error, we add a forcing function Phi(t) to
    % the spatially-discretized problem,
    %
    %    u' = -D1 (u^2/2) + nu*D2 u + Phi(t),
    %
    % chosen so that the semi-discrete solution is exactly u_j(t) = g(x_j,t) at
    % every interior grid point.  For this problem,
    %
    %    Phi(t) = g_t + D1 (g^2/2) - nu*D2 g,
    %
    % where g and g_t = sech^2((x - t/2)/(4*nu))/(16*nu) are evaluated on the
    % spatial grid, and where D1 and D2 include the boundary values of g.  Since g
    % solves the PDE, Phi(t) is just the spatial truncation error.
    %
    % Note that the Jacobian of the advection term is -D1 diag(u).
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    methods (Static)
        function p = problem()
            % Problem parameters.
            p.t0 = 0.0;
            p.tf = 1.0;
            p.xl = -1.0;
            p.xr = 1.0;
            p.Nx = 401;
            p.nu = 5.e-2;
            p.dx = (p.xr-p.xl)/(p.Nx-1);
            p.xgrid = linspace(p.xl+p.dx, p.xr-p.dx, p.Nx-2).';

            % first- and second-derivative matrices for the interior unknowns (in sparse format)
            p.D1 = spdiags([-ones(p.Nx-2,1)/(2*p.dx), ones(p.Nx-2,1)/(2*p.dx)], ...
                [-1, 1], p.Nx-2, p.Nx-2);
            p.D2 = spdiags([ones(p.Nx-2,1)/(p.dx^2), -2*ones(p.Nx-2,1)/(p.dx^2), ...
                ones(p.Nx-2,1)/(p.dx^2)], [-1, 0, 1], p.Nx-2, p.Nx-2);

            % constant Jacobian of the implicit portion
            p.JI_matrix = p.nu*p.D2;
        end

        function val = utrue(x, t)
            % True solution to the spatially-discretized problem.

            p = AdvectionDiffusion.problem();
            val = 0.5*(1.0 - tanh((x(:) - 0.5*t)/(4*p.nu)));
        end

        function val = utrue_t(x, t)
            % Time derivative of the true solution.

            p = AdvectionDiffusion.problem();
            val = 1.0./(16*p.nu*cosh((x(:) - 0.5*t)/(4*p.nu)).^2);
        end

        function val = advection(t, u)
            % Centered approximation of -(u^2/2)_x, including the boundary values.

            p = AdvectionDiffusion.problem();
            u = u(:);
            F = 0.5*u.^2;
            val = -(p.D1*F);
            val(1) = val(1) + 0.5*AdvectionDiffusion.utrue(p.xl, t)^2/(2*p.dx);
            val(end) = val(end) - 0.5*AdvectionDiffusion.utrue(p.xr, t)^2/(2*p.dx);
        end

        function val = diffusion(t, u)
            % Centered approximation of nu*u_xx, including the boundary values.

            p = AdvectionDiffusion.problem();
            u = u(:);
            val = p.JI_matrix*u;
            val(1) = val(1) + p.nu*AdvectionDiffusion.utrue(p.xl, t)/(p.dx^2);
            val(end) = val(end) + p.nu*AdvectionDiffusion.utrue(p.xr, t)/(p.dx^2);
        end

        function val = Phi(t)
            % Forcing function for the spatially-discretized problem.

            p = AdvectionDiffusion.problem();
            g = AdvectionDiffusion.utrue(p.xgrid, t);
            val = AdvectionDiffusion.utrue_t(p.xgrid, t) - AdvectionDiffusion.advection(t, g) ...
                - AdvectionDiffusion.diffusion(t, g);
        end

        function val = fE(t, u)
            % Explicit advection portion of the right-hand side.
            val = AdvectionDiffusion.advection(t, u);
        end

        function val = fI(t, u)
            % Implicit diffusion and forcing portion of the right-hand side.
            val = AdvectionDiffusion.diffusion(t, u) + AdvectionDiffusion.Phi(t);
        end

        function val = JI(t, u)
            % Jacobian of the implicit diffusion and forcing portion of the right-hand side.

            p = AdvectionDiffusion.problem();
            val = p.JI_matrix;
        end

        function val = f(t, u)
            % Full right-hand side function for the IVP.
            val = AdvectionDiffusion.fE(t, u) + AdvectionDiffusion.fI(t, u);
        end

        function val = J(t, u)
            % Jacobian of the full right-hand side function.

            p = AdvectionDiffusion.problem();
            u = u(:);
            val = -(p.D1*spdiags(u, 0, numel(u), numel(u))) + p.JI_matrix;
        end

        function val = u0()
            % Initial condition.

            p = AdvectionDiffusion.problem();
            val = AdvectionDiffusion.utrue(p.xgrid, p.t0);
        end

        function val = explicit_stability_limit(cfl)
            % Step-size limit for the explicit advection term, h_E = cfl*dx/max|u|, where
            % max|u| = 1 for this solution.  The default cfl value is slightly below the
            % stability limit (approximately 2.58) of the explicit ARK3(2)4L[2]SA table on
            % the imaginary axis.
            if nargin < 1
                cfl = 2.4;
            end
            p = AdvectionDiffusion.problem();
            val = cfl*p.dx/1.0;
        end
    end
end
