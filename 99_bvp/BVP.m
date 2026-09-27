classdef BVP
    % BVP.m
    %
    % Class containing parameters and functions to define the second-order, scalar-valued BVP:
    %
    %    u'' - 2*lam*u' + lam^2*u = r(t),  0<t<1,
    %
    %    r(t) = (4*lam^2*exp(lam*(1-t)))/(1+2*exp(lam))
    %           + (lam^2-pi^2)*cos(pi*t) + 2*lam*pi*sin(pi*t)
    %    u(0) = (1+exp(lam))/(1+2*exp(lam)) + 1
    %    u(1) = (1+exp(lam))/(1+2*exp(lam)) - 1
    %
    % This problem has analytical solution
    %
    %    u(t) = exp(lam)/(1+2*exp(lam))*(exp(lam*(t-1))
    %            + exp(-lam*t)) + cos(pi*t)
    %
    % and the stiffness may be adjusted using the real-valued
    % parameter lam<0
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % The one argument when constructing a BVP class is the stiffness parameter, lam.

    properties
        lam = -10.0
        a = 0.0
        b = 1.0
        ua, ub
    end

    methods
        function self = BVP(lam)
            if nargin >= 1 && ~isempty(lam), self.lam = lam; end
            % Store the stiffness parameter and the exact boundary values.
            self.ua = (1.0+exp(self.lam))/(1.0+2.0*exp(self.lam)) + 1.0;
            self.ub = (1.0+exp(self.lam))/(1.0+2.0*exp(self.lam)) - 1.0;
        end

        function val = r(self, t)
            val = 4*self.lam^2*exp(self.lam*(1-t))/(1+2*exp(self.lam)) ...
                + (self.lam^2 - pi*pi)*cos(pi*t) ...
                + 2*self.lam*pi*sin(pi*t);
        end

        function val = p(self, t)
            val = 2*self.lam;
        end

        function val = q(self, t)
            val = -self.lam^2;
        end

        function val = utrue(self, t)
            val = exp(self.lam)/(1+2*exp(self.lam)) ...
                .* (exp(self.lam*(t-1)) + exp(-self.lam*t)) ...
                + cos(pi*t);
        end
    end
end
