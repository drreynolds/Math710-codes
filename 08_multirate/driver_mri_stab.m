% Driver that uses MRI_stab_region to plot the slow stability regions
% S_{rho,alpha} of the explicit MRI-GARK-ERK33a method, for rho = Inf
% (reproducing Figure 1(b) of Sandu, SIAM J. Numer. Anal. 57(5), 2019)
% and for rho = 10.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
clear

alphas = [10, 45, 80];
box = [-3.5, 0.5, -3, 3];
C = MRI.MRIGARKERK33a();

fig = figure('Position', [100, 100, 1000, 600]);
ax1 = subplot(1, 2, 1);
MRI_stab_region(C, alphas, Inf, box, ax1);
title(ax1, 'MRI-GARK-ERK33a: $\mathcal{S}_{\rho=\infty,\alpha}$', 'Interpreter', 'latex');
ax2 = subplot(1, 2, 2);
MRI_stab_region(C, alphas, 10, box, ax2);
title(ax2, 'MRI-GARK-ERK33a: $\mathcal{S}_{\rho=10,\alpha}$', 'Interpreter', 'latex');
saveas(fig, 'MRIGARKERK33a_stability.png');
