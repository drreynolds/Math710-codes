function ARK_joint_stab_region(BE,BI,thetas,rmax,box,fig)
  % Usage: ARK_joint_stab_region(BE,BI,thetas,rmax,box,fig)
  %
  % Computes the joint stability region for an ARK method applied
  % to the scalar, additive test problem,
  %
  %    y'(t) = lI*y + lE*y,
  %
  % using time step h, where lI,lE \in \C, and Re(lI) < 0, Re(lE) < 0.
  % For an s-stage ARK method applied to this problem, then defining
  % the step-size scaled inputs zI = h*lI, zE = h*lE, the function
  % is given by
  %
  %   R(zE,zI) = 1 + (zE*bE + zI*bI)*((eye(s)-zE*AE-zI*AI)\e)
  %
  % where e is a vector of all ones in \R^s.
  %
  % For a given angle 0 <= theta <= pi/2, we define the joint stability
  % region as
  %
  %   Sj(theta) = { zE \in \C : |R(zE,zI)|<1 forall zI in S(theta) }, where
  %   S(theta) = { zI = -a+i*b : a>0, and atan(|b|/a) <= theta }
  %
  % We note that the sector S(theta) contains infinitely many points:
  % (a) it extends arbitrarily far into the complex left half-plane,
  %     i.e., |zI| < infty, and
  % (b) it contains infinitely many angles -theta <= alpha <= theta.
  %
  % However, we need not test all of them.  Since the implicit table is
  % diagonally implicit, the poles of R in zI are at 1/AI(i,i) > 0,
  % outside of S(theta), and so for fixed zE the maximum of |R| over
  % S(theta) is attained on its boundary (the maximum modulus principle),
  % i.e., on the two rays alpha = -theta and alpha = theta.  Moreover,
  % since the Butcher tables are real, R(conj(zE),conj(zI)) =
  % conj(R(zE,zI)).  We therefore test only zI = 0 and the ray
  % alpha = theta, using NI points, with distance logarithmically-
  % scaled away from the origin, to a maximum distance of rmax, over the
  % full NE^2 mesh of points zE within the pre-defined "box" in the
  % complex plane, and then account for the ray alpha = -theta by
  % reflecting the result across the real axis.  Finally, since
  % I - zE*AE - zI*AI is lower triangular, we evaluate R by forward
  % substitution over the whole zE mesh at once for each zI sample.
  %
  % The input 'thetas' is array-valued -- we plot the joint stability
  % region Sj(theta) for each value in this array, and overlay these plots.
  %
  % Inputs:
  %    BE     -- ERK Butcher table structure, with fields A, b
  %    BI     -- DIRK Butcher table structure, with fields A, b
  %    thetas -- array of sector angles (in degrees) to use in creating
  %              overlaid plots
  %    rmax   -- maximum distance from the origin for the DIRK sample points
  %    box    -- [xl, xr, yl, yr] is the bounding box for the sub-region
  %              of the complex plane in which to perform the test.  We
  %              assume that yl=-yr, so that the mesh is symmetric
  %              across the real axis.
  %    fig    -- figure handle to use
  %
  %
  % Daniel R. Reynolds
  % Math & Stat @ UMBC

  % set general parameters
  NE = 101;  % must be odd
  NI = 100;
  CM = lines(length(thetas));

  % check Butcher tables for compatibility
  s = length(BE.b);
  AE = BE.A;  bE = reshape(BE.b,1,s);
  AI = BI.A;  bI = reshape(BI.b,1,s);
  [rowsE,colsE] = size(AE);
  [rowsI,colsI] = size(AI);
  if (length(bE) ~= length(bI))
      error('ARK_joint_stab_region: bI and bI are incompatible')
  end
  if ((s ~= rowsE) || (s ~= colsE))
      error('ARK_joint_stab_region: incompatible explicit Butcher table inputs')
  end
  if ((s ~= rowsI) || (s ~= colsI))
      error('ARK_joint_stab_region: incompatible implicit Butcher table inputs')
  end

  % convert Butcher tables to double precision once (whether they were
  % stored in floating-point or symbolically)
  AEd = double(AE);  bEd = double(bE);
  AId = double(AI);  bId = double(bI);
  if ((norm(AEd - tril(AEd,-1), inf) > 1e-14) || (norm(AId - tril(AId), inf) > 1e-14))
      error('ARK_joint_stab_region: requires an explicit and a diagonally implicit table')
  end

  % set mesh of ERK sample points
  xl = box(1);
  xr = box(2);
  yl = box(3);
  yr = box(4);
  x = linspace(xl,xr,NE);
  y = linspace(yl,yr,NE);
  [X,Y] = meshgrid(x,y);
  ZE = X + Y*sqrt(-1);

  % create new figure window
  xlim = box(1:2);  ylim = box(3:4);
  xax = plot(linspace(xlim(1),xlim(2),10),zeros(1,10),'k:'); hold on
  yax = plot(zeros(1,10),linspace(ylim(1),ylim(2),10),'k:');

  % loop over theta values, creating contour plot data for each
  for itheta = 1:length(thetas)
    theta = thetas(itheta)*pi/180;  % convert to radians

    % initialize max|R| over box
    Rmax = zeros(NE,NE);

    % set array of DIRK sample points: zI = 0, and the upper ray of S(theta)
    r = -logspace(-1,log10(rmax),NI);
    zI = [0, r*(cos(theta)-sin(theta)*sqrt(-1))];

    % compute max|R| over the zI samples, for the whole zE mesh at once,
    % and then reflect across the real axis to account for the lower ray
    for k=1:length(zI)
      Rmax = max(Rmax, abs(ARK_R(ZE,zI(k),AEd,bEd,AId,bId,s)));
    end
    Rmax = max(Rmax, flipud(Rmax));

    % create contours and add to figure
    c = contourc(x,y,Rmax,[1+eps 1+eps]);

    % assemble plot
    figure(fig);
    hold on
    idx = 1;
    while idx < size(c,2)
      cols = idx+1:idx+c(2,idx);
      X = c(1,cols);
      Y = c(2,cols);
      lstring = ['$\theta =\;$', sprintf('%g', thetas(itheta))];
      plot(X, Y, 'color', CM(itheta,:), 'DisplayName', lstring, 'LineWidth', 2)
      idx = idx + c(2,idx) + 1;
    end
    hold off

  end

  % finish up figure
  set(get(get(xax,'Annotation'),'LegendInformation'), 'IconDisplayStyle','off');
  set(get(get(yax,'Annotation'),'LegendInformation'), 'IconDisplayStyle','off');
  axis(box);
  xlabel('Re(zE)');
  ylabel('Im(zE)');
  legend('Location', 'northwest', 'Interpreter', 'latex');
  hold off

% end of function


function R = ARK_R(ZE,zI,AE,bE,AI,bI,s)
  % Usage: R = ARK_R(ZE,zI,AE,bE,AI,bI,s)
  %
  % Evaluates the ARK stability function over the whole zE mesh ZE, for a
  % single zI value: since I - zE*AE - zI*AI is lower triangular, the stages
  %    z_i = (1 + sum_{j<i} (zE*AE(i,j) + zI*AI(i,j)) z_j) / (1 - zI*AI(i,i))
  % follow by forward substitution, and R = 1 + sum_i (zE*bE(i) + zI*bI(i)) z_i
  z = cell(s,1);
  for i=1:s
    zi = ones(size(ZE));
    for j=1:i-1
      zi = zi + (ZE*AE(i,j) + zI*AI(i,j)).*z{j};
    end
    z{i} = zi/(1 - zI*AI(i,i));
  end
  Rsum = zeros(size(ZE));
  for i=1:s
    Rsum = Rsum + (ZE*bE(i) + zI*bI(i)).*z{i};
  end
  R = 1 + Rsum;
