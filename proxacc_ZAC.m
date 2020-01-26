% (C) 2014 Joseph Fernandez. Please see license.txt for license information.
%
% function to implement proximal gradient descent for computing ZACF
%---INPUTS-----
% optfxn - matlab function specifying the optimization function used for
% the cost function.
%
% gradfxn - matlab function specifying the gradient of the optimization
% function.
%
% H - initial "guess" of filter. Usually this is the base filter e.g.
% MACE for ZAMACE.
%
% proxopt - structure containing options for the optimization.
% -----proxopt.t_init: intial t to use (step size).
% -----proxopt.beta: beta parameter (set between 0.1 and 0.8) MUST BE between 0 and 1.
% -----proxopt.tol: used to stop algorithm, normalized error vector with respect to previous iteration, usually 1e-5
% -----proxopt.maxiter: maximum number of iterations to run.
% -----proxopt.N: FFT size used to compute filter
% -----proxopt.Nx: desired filter size (i.e. the size of training images)
% -----proxopt.T, proxopt.p: filter parameters may be passed into the
% function that will be used for numerical stability type adjustments.

function [hspatial, H, info] = proxacc_ZAC(optfxn, gradfxn, H, proxopt, N, Nx)

%% set default options if they do not exist.
if ~exist('proxopt', 'var');
    proxopt.beta = 0.5;
    proxopt.maxiter = inf;
    proxopt.tol = 1e-10;
    proxopt.t_init = prod(N); 
else
    if ~isfield(proxopt, 't_init');
        proxopt.t_init = prod(N);
    end
    if ~isfield(proxopt, 'beta');
        proxopt.beta = 0.5;
    end
    if ~isfield(proxopt, 'maxiter');
        proxopt.maxiter = inf;
    end
    if ~isfield(proxopt, 'tol');
        proxopt.tol = 1e-10;
    end
end

proxopt.V = proxopt.Xspat/(proxopt.Xspat.'*proxopt.Xspat);  % precomputed matrix

%% error check
if proxopt.beta < 0 || proxopt.beta > 1
    error('beta out of range');
end
if proxopt.t_init < 0
    error('t_init must be positive');
end

% put image size and fft size info into proxopt.
proxopt.N = N;
proxopt.Nx = Nx;

%% initialize stuff for H_init
if abs(proxopt.p(1))/max(abs(proxopt.p)) < 1e-5
    H(1) = 0;       % for numerical stability
end

%% Proximal Gradient Method
% initialize
stop = 0; 
iter = 0;
H = h_prox(H, proxopt);     % initially will take the initial H and do a proximal step to zero tail.
H_old = H;                  % initialize the variable "H_old"

x = H;
y = H;

info.tlist = [];
info.obj_val_list = real(optfxn(H_old));
info.rel_err_vect = [];
info.rel_obj_err_vect = [];

while stop == 0
    iter = iter + 1;
    
    % this next statement was added 12/2/13
    if mod(iter,100) == 0
        proxopt.t_init = max(info.tlist(end-98:end));       % helps get to the t value faster if necessary.
    end
   
    % use backtracking to determine t
    t = backtracking_line_search(H_old, optfxn, gradfxn, @h_prox, proxopt);      % use t = 1 to start.---- also, changed first argument to "H_old" 12/2/13
    xplus = h_prox(y - t*gradfxn(y),proxopt);
    yplus = xplus + ((iter-1)/(iter+2))*(xplus-x);
    
    obj_val = real(optfxn(xplus));

    info.tlist = [info.tlist; t];
    info.obj_val_list = [info.obj_val_list; obj_val];
    
    rel_err = norm(H_old-xplus)/norm(H_old);
    info.rel_err_vect = [info.rel_err_vect, rel_err];
    
    rel_obj_err = abs(info.obj_val_list(end-1) - info.obj_val_list(end))/abs(info.obj_val_list(end-1));
    info.rel_obj_err_vect = [info.rel_obj_err_vect, rel_obj_err];

    if (iter >= proxopt.maxiter) || (rel_obj_err < proxopt.tol)
        if iter >= proxopt.maxiter
            warning('gradient descent terminated due to maxiter');
        end
        stop = 1;
        H = xplus;
    end
 
    H_old = xplus;
    x = xplus;
    y = yplus;
    
end

if iter == 1
%     display(['.......Converged in one step. N = ' num2str(N) ', Nx = ' num2str(Nx)]);
else
%     display(['.......number of iterations to converge: ' num2str(iter)]);
end

% final outputs:
Htemp = reshape(H, proxopt.N);
hspatial = ifftn(Htemp,'symmetric');


function [t,H] = backtracking_line_search(H, g_objective, g_gradient, h_prox, proxopt)
    beta = proxopt.beta;  % (0 < beta < 1)
    t = proxopt.t_init;
    f_H = real(g_objective(H));
    grad = g_gradient(H);

    stop1 = 0;
    while stop1 == 0
        Ht = h_prox(H - t*grad, proxopt);
        G = (1/t)*(H-Ht);
        f_Ht = real(g_objective(Ht));
        if f_Ht > (f_H - t*grad'*G + (t/2)*(G'*G))
            t = beta*t;
        else
            stop1 = 1;
            H = Ht;
        end
    end
end

function [H,hspatial] = h_prox(H, proxopt)  
    Hreshaped = reshape(H, proxopt.N);
    hspatial = ifftn(Hreshaped,'symmetric')*sqrt(proxopt.N(1)*proxopt.N(2));
    %----- first proximal step
    hspatial(proxopt.Nx(1)+1:proxopt.N(1),:) = 0;
    hspatial(:,proxopt.Nx(2)+1:proxopt.N(2)) = 0;
    
    %----- second proximal step
    hvect = hspatial(1:proxopt.Nx(1), 1:proxopt.Nx(2));
    hvect = hvect(:);       % vectorize
    
    initialdotprod = proxopt.Xspat.'*hvect;
    val = median(initialdotprod);
    vect = val*proxopt.u; % use this as the u vector, for numerical stability.
%     val=1;
    
    % solve for h_delta, minimal difference to meet constraints
    h_delta = proxopt.V*(vect-initialdotprod);  % proxopt.V is precomputed once
    
    % new template
    hvectnew = hvect+h_delta;
    
%     newdotprod = proxopt.Xspat.'*hvectnew;
    
    % reshape
    hspatial(1:proxopt.Nx(1), 1:proxopt.Nx(2)) = reshape(hvectnew, proxopt.Nx(1), proxopt.Nx(2));
    
    H = fft2(hspatial)/sqrt(proxopt.N(1)*proxopt.N(2));
    H = H(:);
end

end
