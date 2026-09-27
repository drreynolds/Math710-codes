classdef ReactionDiffusion
    % This file defines the RHS and Jacobian functions associated with the
    % reaction-diffusion problem,
    %
    %    u_t = u_{xx} + 1/(1+u^2) + Phi(x,t),  (x,t) in [0,1]^2
    %    u(t,0) = u(t,1) = 0,  t in [0,1]
    %    u(0,x) = x(1-x)
    % that has analytical solution u(x,t) = x(1-x)e^t.  For this problem,
    %    Phi(x,t) = u_t - u_{xx} - 1/(1+u^2)
    %             = x(1-x)e^t + 2e^t - 1/(1+u^2)
    %             = u + 2e^t - 1/(1+u^2)
    %
    % Note that the Jacobian of this RHS equals
    %    D - diag(2u/(1+u^2)^2)
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    methods (Static)
        function p = problem()
            % Problem parameters.
            p.t0 = 0.0;
            p.tf = 1.0;
            p.xl = 0.0;
            p.xr = 1.0;
            p.Nx = 201;
            p.dx = (p.xr-p.xl)/(p.Nx-1);
            p.xgrid = linspace(p.xl+p.dx, p.xr-p.dx, p.Nx-2).';

            % Diffusion matrix in sparse format.
            p.D = spdiags([ones(p.Nx-2,1), -2*ones(p.Nx-2,1), ones(p.Nx-2,1)]/p.dx^2, ...
                [-1, 0, 1], p.Nx-2, p.Nx-2);
        end

        function val = utrue(x, t)
            % True solution to problem.
            val = x(:).*(1-x(:))*exp(t);
        end

        function val = Phi(x, t)
            % Forcing function for the IVP.

            u = ReactionDiffusion.utrue(x, t);
            val = u + 2*exp(t) - 1./(1+u.^2);
        end

        function val = f(t, u)
            % RHS function for the IVP.

            p = ReactionDiffusion.problem();
            u = u(:);
            val = p.D*u + 1./(1+u.^2) + ReactionDiffusion.Phi(p.xgrid, t);
        end

        function val = fE(t, u)
            % Explicit reaction portion of the right-hand side.

            u = u(:);
            val = 1./(1+u.^2);
        end

        function val = fI(t, u)
            % Implicit diffusion and forcing portion of the right-hand side.

            p = ReactionDiffusion.problem();
            u = u(:);
            val = p.D*u + ReactionDiffusion.Phi(p.xgrid, t);
        end

        function val = JI(t, u)
            % Jacobian of the implicit diffusion and forcing portion of the right-hand side.

            p = ReactionDiffusion.problem();
            val = p.D;
        end

        function val = J(t, u)
            % Jacobian (in sparse matrix format) of the right-hand side
            % function, J(t,y) = df/dy, for the IVP.

            p = ReactionDiffusion.problem();
            u = u(:);
            val = p.D - spdiags(2*u./((1+u.^2).^2), 0, numel(u), numel(u));
        end

        function val = u0()
            % Initial condition

            p = ReactionDiffusion.problem();
            val = ReactionDiffusion.utrue(p.xgrid, p.t0);
        end
    end
end
