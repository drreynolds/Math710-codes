function MRI_stab_region(C, alphas, rho, box, ax)
  % Usage: MRI_stab_region(C, alphas, rho, box, ax)
  %
  % Computes the slow stability region of an explicit MRI-GARK method
  % applied to the scalar, additive test problem,
  %
  %    y'(t) = lF*y + lS*y,
  %
  % using slow time step H, where lF,lS \in \C, and Re(lF) < 0, Re(lS) < 0.
  % Defining zF = H*lF and zS = H*lS, each MRI stage solves a linear fast
  % IVP with polynomial forcing exactly, and so (see the lecture notes)
  %
  %    z_1 = 1,
  %    z_i = phi_0(dc_i zF) z_{i-1}
  %          + zS sum_{j<i} ( sum_k (k-1)! G(k,i,j) phi_k(dc_i zF) ) z_j,
  %
  % for i = 2,...,s, where dc_i = c_i - c_{i-1}, and R(zF,zS) = z_s.
  %
  % For a radius rho > 0 (possibly Inf) and an angle 0 <= alpha < 90
  % degrees, the slow stability region is
  %
  %    S_{rho,alpha} = { zS \in \C : |R(zF,zS)| <= 1 for all zF in W }, where
  %    W = { zF \in \C : |zF| <= rho, |arg(zF) - pi| <= alpha }.
  %
  % We use two facts to avoid sampling all of the wedge W:
  % (a) For fixed zS, R is an analytic and bounded function of zF on W,
  %     so the maximum of |R| over W is attained on its boundary (the
  %     maximum modulus principle; for rho = Inf, its extension to
  %     sectors, the Phragmen--Lindelof principle).  We therefore only
  %     sample the two rays arg(zF) = pi -/+ alpha out to |zF| = rho, and
  %     the arc |zF| = rho between them.
  % (b) Since the MRI coefficients are real, R(conj(zF),conj(zS)) =
  %     conj(R(zF,zS)).  We therefore sample only the upper half of the
  %     boundary of W (one ray and half of the arc) over the full zS
  %     mesh, and then account for the lower half by reflecting the
  %     result across the real axis.
  % For each fast sample, we evaluate R over the whole zS mesh at once.
  %
  % The input 'alphas' is array-valued -- we plot the slow stability
  % region S_{rho,alpha} for each value in this array, and overlay these
  % plots on the stability region of the slow base method (zF = 0).
  %
  % Inputs:
  %    C      -- explicit MRI coupling table structure, with fields G
  %              and c (as in MRI.m)
  %    alphas -- array of wedge angles (in degrees, < 90) to use in
  %              creating overlaid plots
  %    rho    -- radius of the wedge W (Inf is allowed)
  %    box    -- [xl, xr, yl, yr] is the bounding box for the sub-region
  %              of the complex plane in which to perform the test.  We
  %              assume that yl=-yr, so that the stability region is
  %              symmetric across the real axis.
  %    ax     -- axes handle to use
  %
  % Daniel R. Reynolds
  % Math & Stat @ UMBC

  % set general parameters
  NS = 401;  % zS mesh points in each direction (must be odd)
  NR = 400;  % zF samples along the ray
  NA = 50;   % zF samples along the half-arc (for rho < Inf)
  CM = lines(length(alphas));

  % extract MRI coefficients, and check that the method is explicit
  G = C.G;
  c = C.c;
  K = size(G, 1);
  s = length(c);
  dc = diff(c);
  for k = 1:K
    Gk = reshape(G(k,:,:), s, s);
    if (norm(Gk - tril(Gk,-1), inf) > 1e-14)
      error('MRI_stab_region: only explicit MRI methods are supported')
    end
  end

  % set mesh of zS sample points
  xl = box(1);
  xr = box(2);
  yl = box(3);
  yr = box(4);
  x = linspace(xl, xr, NS);
  y = linspace(yl, yr, NS);
  [X, Y] = meshgrid(x, y);
  zS = X + Y*sqrt(-1);

  % create axes and plot the base method stability region (zF = 0)
  hold(ax, 'on')
  xax = plot(ax, linspace(xl,xr,10), zeros(1,10), 'k:');
  yax = plot(ax, zeros(1,10), linspace(yl,yr,10), 'k:');
  set(get(get(xax,'Annotation'),'LegendInformation'), 'IconDisplayStyle','off');
  set(get(get(yax,'Annotation'),'LegendInformation'), 'IconDisplayStyle','off');
  contour(ax, x, y, absR(0, zS, G, dc, K, s), [1+eps 1+eps], 'LineColor', 'k', ...
          'LineWidth', 2, 'DisplayName', 'Base');

  % loop over alpha values, creating contour plot data for each
  for ialpha = 1:length(alphas)
    alpha = alphas(ialpha)*pi/180;  % convert to radians

    % set array of zF sample points on the upper half of the boundary of W;
    % for rho = Inf we sample the ray to |zF| = 1e6, where |R| has reached
    % its limiting value
    if (isinf(rho))
      r = logspace(-3, 6, NR);
      zF = r*exp((pi - alpha)*sqrt(-1));
    else
      r = logspace(-3, log10(rho), NR);
      beta = linspace(pi - alpha, pi, NA);
      zF = [r*exp((pi - alpha)*sqrt(-1)), rho*exp(beta*sqrt(-1))];
    end

    % compute max|R| over these zF samples, and reflect across the real axis
    Rmax = absR(0, zS, G, dc, K, s);
    for k = 1:length(zF)
      Rmax = max(Rmax, absR(zF(k), zS, G, dc, K, s));
    end
    Rmax = max(Rmax, flipud(Rmax));

    % create contour and add to figure
    lstring = ['$\alpha =\;$', sprintf('%g', alphas(ialpha)), '$^\circ$'];
    contour(ax, x, y, Rmax, [1+eps 1+eps], 'LineColor', CM(ialpha,:), ...
            'LineWidth', 2, 'DisplayName', lstring);
  end

  % finish up figure
  axis(ax, box);
  axis(ax, 'equal');
  axis(ax, box);
  xlabel(ax, 'Re(zS)');
  ylabel(ax, 'Im(zS)');
  legend(ax, 'Location', 'northwest', 'Interpreter', 'latex');
  hold(ax, 'off')

end


function Rabs = absR(zF, zS, G, dc, K, s)
  % Usage: Rabs = absR(zF, zS, G, dc, K, s)
  %
  % Utility routine to evaluate |R(zF,zS)| over the whole zS mesh.
  z = cell(s, 1);
  z{1} = ones(size(zS));
  for i = 2:s
    phi = phi_functions(dc(i-1)*zF, K);
    znew = phi(1)*z{i-1};
    for j = 1:i-1
      gamma = 0;
      for k = 1:K
        gamma = gamma + factorial(k-1)*G(k,i,j)*phi(k+1);
      end
      znew = znew + zS*gamma.*z{j};
    end
    z{i} = znew;
  end
  Rabs = abs(z{s});
end


function phi = phi_functions(z, n)
  % Usage: phi = phi_functions(z, n)
  %
  % Returns the array [phi_0(z), phi_1(z), ..., phi_n(z)], where
  %
  %    phi_0(z) = e^z,
  %    phi_k(z) = 1/(k-1)! int_0^1 e^{(1-theta)z} theta^{k-1} dtheta,  k >= 1.
  %
  % These are computed together from the exponential of the
  % (n+1)x(n+1) matrix
  %
  %    [ z 1 0 ... 0 ]
  %    [ 0 0 1 ... 0 ]
  %    [ ...     ... ]
  %    [ 0 0 0 ... 1 ]
  %    [ 0 0 0 ... 0 ],
  %
  % whose first row is [phi_0(z), ..., phi_n(z)].  Unlike the recurrence
  % phi_{k+1}(z) = (phi_k(z) - 1/k!)/z, this is accurate for small |z|.
  A = diag(ones(n,1), 1);
  A(1,1) = z;
  E = expm(A);
  phi = E(1,:);
end
