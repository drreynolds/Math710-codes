% Script to plot the linear stability regions of fractional-step
% Runge--Kutta (FSRK) methods, i.e., fractional-step (operator-splitting)
% methods in which each sub-flow is approximated by a Runge--Kutta method.
% For the scalar test problem
%      y' = lambda^{1} y + ... + lambda^{M} y,
% each sub-step simply multiplies the solution by the stability function of
% its sub-integrator, so that the FSRK stability function is the product
%      R = prod_k prod_l R^{l}( alpha_k^{l} z^{l} ),   z^{l} = H lambda^{l}
% (Spiteri & Wei, J. Comput. Phys. 476:111900, 2023, Theorem 3.2).  To plot
% this in a single complex variable z, we write z^{l} = mu_l z for given
% multipliers mu_l, as in Spiteri & Wei's examples.
%
% This script reproduces two of their examples (experiments 3 and 4 in
% the lecture notes):
%   1. Strang--Marchuk splitting of y' = -20 y with Heun's method for
%      partition 1 and the L-stable SDIRK(2,2) method for partition 2, for
%      the 50-50, 10-90 and 90-10 splits (their Figure 1), and with the roles
%      of the two sub-integrators exchanged;
%   2. Ruth's splitting with Kutta's RK3 method for partition 1 and the
%      SDIRK(2,3) method for partition 2, with lambda^{1} = lambda^{2} (their
%      Figure 5), and with the sub-integrators exchanged.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../04_explicit_one_step');
addpath('../05_implicit_one_step');
addpath('../utilities');

% sub-integrators
gamma22 = (2 - sqrt(2))/2;       % L-stable SDIRK(2,2)
gamma23 = (3 + sqrt(3))/6;       % third-order SDIRK(2,3)
SDIRK22 = DIRK.SDIRK2(gamma22);
SDIRK23 = DIRK.SDIRK2(gamma23);
HeunB = ERK.Heun();
RK3 = ERK.ERK3();

% Experiment 3: Strang--Marchuk splitting of y' = -20 y, with
% z^{1} = mu_1 z for the Heun partition and z^{2} = mu_2 z for the SDIRK
% partition, where z = -H as in Spiteri & Wei
splits = {'50-50 split', [10.0, 10.0];
          '10-90 split', [2.0, 18.0];
          '90-10 split', [18.0, 2.0]};
colors = {[0.839, 0.153, 0.157], [0.173, 0.627, 0.173], [0.122, 0.467, 0.706]};
box = [-3.8, 0.6, -4.5, 4.5];
box2 = [-3.8, 6.5, -4.5, 4.5];
fig = figure('Position', [100, 100, 1300, 500]);
ax1 = subplot(1, 2, 1);  hold(ax1, 'on');
ax2 = subplot(1, 2, 2);  hold(ax2, 'on');
for i = 1:size(splits, 1)
    name = splits{i,1};
    mu = splits{i,2};
    % Heun takes the two half steps, SDIRK(2,2) takes the full step
    FSRK_stab_region(FractionalStep.StrangMarchuk(), {HeunB, SDIRK22}, mu, box, ax1, colors{i}, name);
    % roles exchanged: SDIRK(2,2) takes the two half steps, Heun the full step
    FSRK_stab_region(FractionalStep.StrangMarchuk(), {SDIRK22, HeunB}, [mu(2), mu(1)], box2, ax2, colors{i}, name);
end
% with the roles exchanged, the 10-90 stability region is unbounded: it is
% the EXTERIOR of its curve, since R tends to 2(1-2 gamma)^2/(81 gamma^4) < 1
% as |z| -> infinity
fprintf('Strang, SDIRK(2,2) half steps + Heun, 10-90 split: the stability region is the exterior of its curve\n');
title(ax1, 'Strang: Heun (half steps) + SDIRK(2,2)');
title(ax2, 'Strang: SDIRK(2,2) (half steps) + Heun');
finish_axes(ax1, box);
finish_axes(ax2, box2);
saveas(fig, 'FSRK_Strang_stability.png');

% Experiment 4: Ruth splitting with lambda^{1} = lambda^{2}, so z^{1} = z^{2} = z
box = [-10, 5, -10, 10];
fig = figure('Position', [100, 100, 1100, 650]);
ax1 = subplot(1, 2, 1);  hold(ax1, 'on');
ax2 = subplot(1, 2, 2);  hold(ax2, 'on');
FSRK_stab_region(FractionalStep.Ruth(), {RK3, SDIRK23}, [1.0, 1.0], box, ax1, colors{3}, 'Ruth (RK3 + SDIRK(2,3))');
FSRK_stab_region(FractionalStep.Ruth(), {SDIRK23, RK3}, [1.0, 1.0], box, ax2, colors{3}, 'Ruth (SDIRK(2,3) + RK3)');
finish_axes(ax1, box);
finish_axes(ax2, box);
saveas(fig, 'FSRK_Ruth_stability.png');

% report the left half-plane poles for experiment 4
names = {'RK3 + SDIRK(2,3)', 'SDIRK(2,3) + RK3'};
tables = {{RK3, SDIRK23}, {SDIRK23, RK3}};
for i = 1:2
    poles = FSRK_poles(FractionalStep.Ruth(), tables{i}, [1.0, 1.0]);
    poles = poles(real(poles) < 0);
    strs = cell(1, numel(poles));
    for j = 1:numel(poles)
        if (abs(imag(poles(j))) < 1e-12)
            strs{j} = sprintf('%.4g', real(poles(j)));
        else
            strs{j} = sprintf('%.4g%+.4gi', real(poles(j)), imag(poles(j)));
        end
    end
    fprintf('Ruth (%s): poles in the left half-plane at z = %s\n', names{i}, strjoin(strs, ', '));
end


function R = FSRK_stability_function(S, B, mu, z)
    % Usage: R = FSRK_stability_function(S, B, mu, z)
    %
    % Evaluates the stability function of the FSRK method with fractional-step
    % coefficient table S (from FractionalStep.m), where partition l uses
    % the Runge--Kutta method with Butcher table B{l}, at every entry of the
    % array z, where z^{l} = mu(l) z.
    alpha = double(S.alpha);
    R = ones(size(z));
    for k = 1:size(alpha, 2)
        for l = 1:size(alpha, 1)
            if (alpha(l,k) ~= 0.0)
                R = R .* RK_stability_function(B{l}, alpha(l,k)*mu(l)*z);
            end
        end
    end
end


function poles = FSRK_poles(S, B, mu)
    % Usage: poles = FSRK_poles(S, B, mu)
    %
    % Returns the poles of the FSRK stability function in the variable z.  The
    % stability function of each Runge--Kutta method has poles at z = 1/a,
    % for each nonzero eigenvalue a of its matrix A, and so the sub-step
    % with coefficient alpha_k^{l} contributes poles at z = 1/(alpha_k^{l} mu_l a).
    % These lie in the left half-plane when alpha_k^{l} < 0 (for a > 0).
    alpha = double(S.alpha);
    poles = [];
    for l = 1:size(alpha, 1)
        evals = eig(double(B{l}.A));
        evals = evals(abs(evals) > 1e-14);
        for a = evals.'
            for k = 1:size(alpha, 2)
                if (alpha(l,k) ~= 0.0)
                    poles(end+1) = 1/(alpha(l,k)*mu(l)*a);
                end
            end
        end
    end
    poles = unique(round(poles, 12));
end


function FSRK_stab_region(S, B, mu, box, ax, color, label)
    % Usage: FSRK_stab_region(S, B, mu, box, ax, color, label)
    %
    % Plots the boundary of the stability region { z : |R(z)| <= 1 } of the FSRK
    % method (S, B) with multipliers mu, over the box [xl, xr, yl, yr] in the
    % complex plane, on the axes ax, using the given color and legend label.
    % Poles in the left half-plane are marked with an 'x', since a stability
    % region may have holes too small to resolve on the mesh.
    N = 801;
    x = linspace(box(1), box(2), N);
    y = linspace(box(3), box(4), N);
    [X, Y] = meshgrid(x, y);
    R = abs(FSRK_stability_function(S, B, mu, X + Y*sqrt(-1)));
    contour(ax, x, y, R, [1.0, 1.0], 'LineColor', color, 'LineWidth', 2, 'DisplayName', label);
    poles = FSRK_poles(S, B, mu);
    for p = poles
        if ((real(p) < 0) && (box(1) <= real(p)) && (real(p) <= box(2)) && (box(3) <= imag(p)) && (imag(p) <= box(4)))
            plot(ax, real(p), imag(p), 'x', 'Color', color, 'MarkerSize', 8, 'HandleVisibility', 'off');
        end
    end
end


function finish_axes(ax, box)
    % Usage: finish_axes(ax, box)
    %
    % Utility routine to label the axes, and set their limits, aspect ratio,
    % grid and legend.
    xlabel(ax, 'Re(z)');
    ylabel(ax, 'Im(z)');
    axis(ax, 'equal');
    axis(ax, box);
    grid(ax, 'on');
    legend(ax, 'Location', 'northeast');
    hold(ax, 'off');
end
