function [w, stats] = kiops(tau_out, A, u, tol, m_init, mmin, mmax, iop, task1)
% kiops.m
%
% MATLAB port/adaptation of the original Python kiops.py file supplied with
% these exponential-integrator examples.
%
% Usage:
%     [w, stats] = kiops(tstops, A, u, ...)
%
% Evaluate a linear combination of the phi functions evaluated at t*A acting on
% vectors from u, that is
%
%     w(i,:) = phi_0(t(i) A) u(1,:) + phi_1(t(i) A) u(2,:) ...
%
% The size of the Krylov subspace is changed dynamically during the integration.
% The Krylov subspace is computed using the incomplete orthogonalization method.
%
% Arguments:
%     tau_out = array of output times
%     A       = matrix argument of the phi functions
%     u       = matrix with rows representing the vectors to be multiplied by
%               the phi functions
%
% Optional arguments:
%     tol          = convergence tolerance required
%     mmin, mmax   = let the Krylov size vary between mmin and mmax
%     m_init       = estimate of the appropriate Krylov size
%     iop          = length of incomplete orthogonalization procedure
%     task1        = if true, divide the result by the output time
%
% Returns:
%     w        = linear combination of phi functions evaluated at t*A acting on
%                the vectors from u
%     stats(1) = number of substeps
%     stats(2) = number of rejected steps
%     stats(3) = number of Krylov steps
%     stats(4) = number of matrix exponentials
%     stats(5) = error estimate
%     stats(6) = Krylov size of the last substep
%
% n is the size of the original problem.
% p is the highest index of the phi functions.
%
% Original KIOPS references retained from kiops.py:
% Gaudreault, S., Rainwater, G. and Tokman, M., 2018. KIOPS: A fast adaptive
% Krylov subspace solver for exponential integrators. Journal of Computational
% Physics. Based on the PHIPM and EXPMVP codes
% (http://www1.maths.leeds.ac.uk/~jitse/software.html).
% https://gitlab.com/stephane.gaudreault/kiops.
%
% Niesen, J. and Wright, W.M., 2011. A Krylov subspace method for option
% pricing. SSRN 1799124.
%
% Niesen, J. and Wright, W.M., 2012. Algorithm 919: A Krylov subspace algorithm
% for evaluating the phi-functions appearing in exponential integrators. ACM
% Transactions on Mathematical Software (TOMS), 38(3), p.22.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

    if nargin < 4 || isempty(tol), tol = 1e-10; end
    if nargin < 5 || isempty(m_init), m_init = 10; end
    if nargin < 6 || isempty(mmin), mmin = 10; end
    if nargin < 7 || isempty(mmax), mmax = 128; end
    if nargin < 8 || isempty(iop), iop = 2; end
    if nargin < 9 || isempty(task1), task1 = false; end

    tau_out = tau_out(:);
    [ppo, n] = size(u);
    p = ppo - 1;

    if p == 0
        p = 1;
        % Add extra row of zeros.
        u = [u; zeros(1, numel(u))];
    end

    % We only allow m to vary between mmin and mmax.
    m = max(mmin, min(m_init, mmax));

    % Preallocate matrix.
    V = zeros(mmax + 1, n + p);
    H = zeros(mmax + 1, mmax + 1);

    step = 0;
    krystep = 0;
    ireject = 0;
    reject = 0;
    exps = 0;
    sgn = sign(tau_out(end));
    tau_now = 0.0;
    tau_end = abs(tau_out(end));
    happy = false;
    j = 0;

    conv = 0.0;

    numSteps = numel(tau_out);

    % Initial condition.
    w = zeros(numSteps, n);
    w(1,:) = u(1,:);

    % Compute the 1-norm of u.
    if size(u,1) > 1
        local_nrmU = sum(abs(u(2:end,:)), 2);
        global_normU = local_nrmU;
        normU = max(global_normU);
    else
        normU = 0;
    end

    % Normalization factors.
    if ppo > 1 && normU > 0
        ex = ceil(log2(normU));
        nu = 2^(-ex);
        mu = 2^ex;
    else
        nu = 1.0;
        mu = 1.0;
    end

    % Flip the rest of the u matrix.
    u_flip = nu * flipud(u(2:end,:));

    % Compute an initial starting approximation for the step size.
    tau = tau_end;

    % Setting the safety factors and tolerance requirements.
    if tau_end > 1
        gamma = 0.2;
        gamma_mmax = 0.1;
    else
        gamma = 0.9;
        gamma_mmax = 0.6;
    end

    delta = 1.4;

    % Used in the adaptive selection.
    oldm = -1;
    oldtau = NaN;
    omega = NaN;
    orderold = true;
    kestold = true;

    l = 0;

    while tau_now < tau_end
        % Compute necessary starting information.
        if j == 0

            H(:,:) = 0.0;

            V(1,1:n) = w(l+1,:);

            % Update the last part of w.
            for k = 0:(p-2)
                idx = p - k + 1;
                V(1,n+k+1) = (tau_now^idx) / factorial(idx) * mu;
            end
            V(1,n+p) = mu;

            % Normalize initial vector (this norm is nonzero).
            local_sum = V(1,1:n) * V(1,1:n).';
            global_sum_nrm = local_sum;
            beta = sqrt(global_sum_nrm + V(1,n+1:n+p) * V(1,n+1:n+p).');

            % The first Krylov basis vector.
            V(1,:) = V(1,:) / beta;
        end

        % Incomplete orthogonalization process.
        while j < m

            j = j + 1;

            % Augmented matrix-vector product.
            V(j+1,1:n) = (A * V(j,1:n).').' + V(j,n+1:n+p) * u_flip;
            if p > 1
                V(j+1,n+1:n+p-1) = V(j,n+2:n+p);
            end
            V(j+1,end) = 0.0;

            % Classical Gram-Schmidt.
            ilow = max(0, j - iop);
            rows = (ilow+1):j;

            local_sum = V(rows,1:n) * V(j+1,1:n).';
            global_sum = local_sum;
            H(rows,j) = global_sum + V(rows,n+1:n+p) * V(j+1,n+1:n+p).';

            V(j+1,:) = V(j+1,:) - (V(rows,:).' * H(rows,j)).';

            local_sum = V(j+1,1:n) * V(j+1,1:n).';
            global_sum_nrm = local_sum;
            nrm = sqrt(global_sum_nrm + V(j+1,n+1:n+p) * V(j+1,n+1:n+p).');

            % Happy breakdown.
            if nrm < tol
                happy = true;
                break;
            end

            H(j+1,j) = nrm;
            V(j+1,:) = (1.0 / nrm) * V(j+1,:);

            krystep = krystep + 1;
        end

        % To obtain the phi_1 function which is needed for error estimate.
        H(1,j+1) = 1.0;

        % Save h_{j+1,j} and remove it temporarily to compute the exponential of H.
        nrm = H(j+1,j);
        H(j+1,j) = 0.0;

        % Compute the exponential of the augmented matrix.
        F = expm(sgn * tau * H(1:j+1,1:j+1));
        exps = exps + 1;

        % Restore the value of H_{m+1,m}.
        H(j+1,j) = nrm;

        if happy
            % Happy breakdown wrap up.
            omega = 0.0;
            err = 0.0;
            tau_new = min(tau_end - (tau_now + tau), tau);
            m_new = m;
            happy = false;

        else

            % Local truncation error estimation.
            err = abs(beta * nrm * F(j,j+1));

            % Error for this step.
            oldomega = omega;
            omega = tau_end * err / (tau * tol);

            % Estimate order.
            if m == oldm && tau ~= oldtau && ireject >= 1
                order = max(1, log(omega/oldomega) / log(tau/oldtau));
                orderold = false;
            elseif orderold || ireject == 0
                orderold = true;
                order = j / 4;
            else
                orderold = true;
            end

            % Estimate k.
            if m ~= oldm && tau == oldtau && ireject >= 1
                kest = max(1.1, (omega/oldomega)^(1/(oldm-m)));
                kestold = false;
            elseif kestold || ireject == 0
                kestold = true;
                kest = 2;
            else
                kestold = true;
            end

            if omega > delta
                remaining_time = tau_end - tau_now;
            else
                remaining_time = tau_end - (tau_now + tau);
            end

            % Krylov adaptivity.
            same_tau = min(remaining_time, tau);
            tau_opt = tau * (gamma / omega)^(1 / order);
            tau_opt = min(remaining_time, max(tau/5, min(5*tau, tau_opt)));

            m_opt = ceil(j + log(omega / gamma) / log(kest));
            m_opt = max(mmin, min(mmax, max(floor(3/4*m), min(m_opt, ceil(4/3*m)))));

            if j == mmax
                if omega > delta
                    m_new = j;
                    tau_new = tau * (gamma_mmax / omega)^(1 / order);
                    tau_new = min(tau_end - tau_now, max(tau/5, tau_new));
                else
                    tau_new = tau_opt;
                    m_new = m;
                end
            else
                m_new = m_opt;
                tau_new = same_tau;
            end
        end

        % Check error against target.
        if omega <= delta

            % Yep, got the required tolerance; update.
            reject = reject + ireject;
            step = step + 1;

            % Update for tau_out in the interval (tau_now, tau_now + tau).
            blownTs = 0;
            nextT = tau_now + tau;
            for k = l:(numSteps-1)
                if abs(tau_out(k+1)) < abs(nextT)
                    blownTs = blownTs + 1;
                end
            end

            if blownTs ~= 0
                % Copy current w to w we continue with.
                w(l+blownTs+1,:) = w(l+1,:);

                for k = 0:(blownTs-1)
                    tauPhantom = tau_out(l+k+1) - tau_now;
                    F2 = expm(sgn * tauPhantom * H(1:j,1:j));
                    w(l+k+1,:) = beta * (F2(1:j,1).' * V(1:j,1:n));
                end

                % Advance l.
                l = l + blownTs;
            end

            % Using the standard scheme.
            w(l+1,:) = beta * (F(1:j,1).' * V(1:j,1:n));

            % Update tau_out.
            tau_now = tau_now + tau;

            j = 0;
            ireject = 0;

            conv = conv + err;

        else
            % Nope, try again.
            ireject = ireject + 1;

            % Restore the original matrix.
            H(1,j+1) = 0.0;
        end

        oldtau = tau;
        tau = tau_new;

        oldm = m;
        m = m_new;
    end

    if task1
        for k = 1:numSteps
            w(k,:) = w(k,:) / tau_out(k);
        end
    end

    m_ret = m;

    stats = [step, reject, krystep, exps, conv, m_ret];
end
