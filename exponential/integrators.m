function methods = integrators()
% integrators.m
%
% Convenience registry for the exponential and classical Runge--Kutta methods.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

    methods.expRK1 = @expRK1;
    methods.expRK2s2a = @expRK2s2a;
    methods.expRK3s3a = @expRK3s3a;
    methods.expRK4s5 = @expRK4s5;
    methods.expRB2 = @expRB2;
    methods.expRB3s3 = @expRB3s3;
    methods.RK2 = @RK2;
    methods.RK4 = @RK4;
end
