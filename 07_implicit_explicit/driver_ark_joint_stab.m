% Driver that uses ARK_joint_stab_region to plot the joint stability
% region of the ARS(1,1,1), ARS(1,2,2), ARS(2,2,2), and ARS(3,4,3)
% Butcher table pairs from Ascher et al. (1997).
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
clear
clc

thetas = [0, 15, 30, 45, 60, 75, 90];
rmax = 1e5;

% ARS (1,1,1): forward-backward Euler
box = [-3, 1, -2, 2];
BE.A = [0, 0; 1, 0];
BE.b = [1; 0];
BI.A = [0, 0; 0, 1];
BI.b = [0; 1];
fig = figure();
ARK_joint_stab_region(BE, BI, thetas, rmax, box, fig);
title('ARS(1,1,1) joint stability region');
saveas(fig, 'ARS111_joint_stability.png');

% ARS (1,2,2): explicit and (padded) implicit midpoint
box = [-3, 1, -2, 2];
BE.A = [0, 0; 1/2, 0];
BE.b = [0; 1];
BI.A = [0, 0; 0, 1/2];
BI.b = [0; 1];
fig = figure();
ARK_joint_stab_region(BE, BI, thetas, rmax, box, fig);
title('ARS(1,2,2) joint stability region');
saveas(fig, 'ARS122_joint_stability.png');

% ARS (2,2,2)
box = [-3, 1, -3, 3];
g = (2 - sqrt(2))/2;
d = 1 - 1/(2*g);
BE.A = [0, 0, 0; g, 0, 0; d, 1-d, 0];
BE.b = [d; 1-d; 0];
BI.A = [0, 0, 0; 0, g, 0; 0, 1-g, g];
BI.b = [0; 1-g; g];
fig = figure();
ARK_joint_stab_region(BE, BI, thetas, rmax, box, fig);
title('ARS(2,2,2) joint stability region');
saveas(fig, 'ARS222_joint_stability.png');

% ARS (3,4,3), using the 10-digit published coefficients
box = [-4, 1, -4, 4];
g = 0.4358665215;
b1 = 1.208496649;
b2 = -0.644363171;
BE.A = [0.0, 0.0, 0.0, 0.0;
        g, 0.0, 0.0, 0.0;
        0.3212788860, 0.3966543747, 0.0, 0.0;
       -0.105858296, 0.5529291479, 0.5529291479, 0.0];
BE.b = [0.0; b1; b2; g];
BI.A = [0.0, 0.0, 0.0, 0.0;
        0.0, g, 0.0, 0.0;
        0.0, 0.2820667392, g, 0.0;
        0.0, b1, b2, g];
BI.b = [0.0; b1; b2; g];
fig = figure();
ARK_joint_stab_region(BE, BI, thetas, rmax, box, fig);
title('ARS(3,4,3) joint stability region');
saveas(fig, 'ARS343_joint_stability.png');

