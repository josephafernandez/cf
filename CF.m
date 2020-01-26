% This function computes a ZACF for a given set of training signals.

% list of parameters here
% trainset: a cell array of the raw training signals that are to be
%           used to train a filter.
% N: the size DFT to use to train the ZACF
% opt: options structure for determining the details of the correlation
%      filter to be trained. Details below.
% crop: logical, determines if the output filter is cropped or not.
% normalize: logical, determines if output filter is normalized or not

% options structure - only need to include necessary options for the filter
%                     you want!
% opt.filtertype - the type of correlation filter to train
%                  Options include:
%                     MACE - minimum average correlation energy filter
%                     OTSDF - optimal tradeoff synthetic discriminant function filter
%                     UCF - unconstrained correlation filter
%                     CCF - constrained correlation filter
% opt.u - the constraint vector for constrained filters
% opt.alpha - alpha parameter (different purpose for each filter)
% opt.beta - beta parameter (different purpose for each filter)
% opt.gamma - gamma parameter (different purpose for each filter)
% opt.lambda - lambda parameter (different purpose for each filter)
% opt.desiredg - desired correlation output plane (in freq. domain)

% stat - statistics about the output.

% ==============outputs====================
% h_za: space domain zero-aliasing template
% h_base: space domain standard template
% CNh output structure holding cropped/normalized filters
% vo = various output. structure to give the flexibility of outputing
% different things for different filters.
%
%
%-----development notes
% 11/6/13: added Naresh's proximal gradient and accelerated proximal
% gradient descent algorithms.

function [h_base, CNh, vo] = CF(trainset, N, opt, crop, normalize, Sza)
    
    %% error check code
    
        
%     if strcmp(type, '2D') == 0 && strcmp(type, '3D') == 0
%         error('Error, invalid argument for "type" input');
%     end
%     
%     if strcmp(type, '2D')
%         if numel(Nx) ~= 2
%             error('dims vector must be 2x1 or 1x2');
%         end
%     else
%         if numel(Nx) ~= 3
%             error('dims vector must be 3x1 or 1x3');
%         end
%     end
%     
%     if strcmp(method, 'normal') == 0 && strcmp(method, 'memory_efficient') == 0 && strcmp(method, 'MEGA') == 0
%         error('Error, invalid argument for "method" input');
%     end

%% Process Inputs, Check for Errors
if ~isfield(opt, 'method')
    opt.method = 'standard';        % set to standard for legacy reasons
end

if ~isfield(opt, 'zeromean')
    opt.zeromean = 0;        % set to 0 for default.
end

trainset = trainset(:);     % "vectorize" the cell array
Q = numel(trainset);

% make sure the u field is a vector
if isfield(opt, 'u')
    opt.u = opt.u(:);
end


sizes = cell2mat(cellfun(@size, trainset, 'UniformOutput', false));

% determine the dimensionality of the signals.
% use the first signal
if ndims(size(trainset{1})) == 2 && min(size(trainset{1})) == 1
    dimensions = 1;
elseif ndims(size(trainset{1})) == 2 && min(size(trainset{1})) ~= 1
    dimensions  = 2;
else
    dimensions = ndims(size(trainset{1}));
end

% for 1D only - make N equal to 2 elements, and vectorize all training signals
if dimensions == 1    
    % for purposes of coding, treat this as a "2D" signal. Make N 2D
    N = [N,1];
    % now, make sure that all signals are vectorized to avoid issues.
    for i0 = 1:numel(trainset);
        trainset{i0} = trainset{i0}(:);
    end
    % update the sizes since we vectorized.
    sizes = cell2mat(cellfun(@size, trainset, 'UniformOutput', false));

end

% 1D constraint
if ((sum(sizes(:,1) == sizes(1,1))) ~= Q)
    error('All elements of trainset must be the same - dimension 1');
end

% 2D constraint
% do this for all signals
if ((sum(sizes(:,2) == sizes(1,2))) ~= Q)
    error('All elements of trainset must be the same - dimension 2');
end


% 3D constraint
if dimensions == 3
    if ((sum(sizes(:,3) == sizes(1,3))) ~= Q)
        error('All elements of trainset must be the same - dimension 3');
    end
end

% now set Nx, the size of the input signals
Nx = sizes(1,:);

% check the inputs for N
if numel(N) ~= numel(Nx)
    error('dimension of N must be the same as the dimensions of the the training set');
end


%% process training signals

% get mean of training set
msignal = zeros(Nx);
for q = 1:Q
    msignal = msignal + (1/Q)*trainset{q,1};
end
Msignal = fftn(msignal,N);      % frequency domain mean
Msignal = Msignal(:);           % vectorize the mean

X = zeros(prod(N),Q);
Di = zeros(prod(N),Q);
Si = zeros(prod(N),Q);
for q = 1:Q
    IM = fftn(trainset{q,1}, N);                          % takes n-D fft of the signal
    X(:,q) = IM(:);                        % vectorizes this freq-domain image
    Di(:,q) = X(:,q).*conj(X(:,q));      % vector storing the energy of a single training image (same as abs(X).^2)
    Si(:,q) = (X(:,q)-Msignal).*conj(X(:,q)-Msignal);
end

D = sum(Di,2)/(prod(N)*Q);              % D is an average of all the energies in the training images 
S = sum(Si,2)/(prod(N)*Q);              % S is an average of the images minus the means.
P = ones(size(D))/(prod(N));            % P matrix, used for ONV.
M = Msignal.*conj(Msignal)/prod(N);     % M matrix, used for CCF/UCF formulations

if opt.zeromean == 1
    X(1,:) = 0;
    D(1) = 1;
end

clear('Di', 'Si', 'IM');         % memory management
    
% now, calculate the desired filter.
%% MACE, OTSDF
if strcmp(opt.filtertype, 'MACE') || strcmp(opt.filtertype, 'OTSDF')
    if strcmp(opt.filtertype, 'MACE');
        T = D; % diag matrix and take inverse in the same step  
    elseif strcmp(opt.filtertype, 'OTSDF');
        T = opt.alpha * D + (1-opt.alpha) * P;
    end
    
    u = prod(N)*opt.u;
    
    % ------- base
    Xw = bsxfun(@rdivide, X, T);        % same as inv(T)*X
    H_base = Xw* ((X'*Xw)\u);
    h_base = real(ifftn(reshape(H_base, N)));
    
    % output ACE.
    vo.ACE_base = real(H_base'*diag(D)*H_base);
    
%% CCF
elseif strcmp(opt.filtertype, 'CCF');
    T = opt.lambda*D - opt.lambda*(1-opt.psi)*M + opt.psi*(1-opt.lambda)*P;
    u = prod(N)*opt.u;
    if isfield(opt, 'var')
        % if a variance is specified, build desired output
        % assumes desired peak at origin
        % assume all will have the same desired G
        if opt.var == 0
            opt.desiredG = prod(N)*ones(size(X)); %a delta function in space is a flat function in frequency
        else
            % center the Gaussian at coordinates Nx1, Nx2
            gindx = GetMatIndices(N);
            gim = mvnpdf(gindx,Nx,[opt.var 0; 0 opt.var]);
            gim = reshape(gim,N);
            gim = circshift(gim,floor((1-N)/2)); %move the center to the 1,1 location (see thesis notes for explanation)
            gim = prod(N)*gim/max(gim(:)); %to ensure that the peak value is 1 in space domain
            gimF = fft2(gim);
            opt.desiredG = repmat(gimF(:),1,Q);
        end     
    elseif isempty(opt.desiredG)
        % by default, use impulses for all training images
        opt.desiredG = prod(N)*ones(size(X));       % impulse is constant in FD
    end
    
    % solve for (little) p
    p = 1/(prod(N)*Q) * sum(X.*opt.desiredG,2);       % X is a matrix, each column is a FFT of a training image
                                                      % opt.desiredG is a NxQ matrix, each column is the desired correlation 
                                                      % output for that training image, in the freq. domain
   
                                      
    Xw = bsxfun(@rdivide, X,T);      % same as inv(T)*X
    H_base = bsxfun(@rdivide, p,T) + (Xw/(X'*Xw))*(u-Xw'*p);
    H_base = reshape(H_base, N);
    h_base = real(ifftn(H_base));
    

%% UCF
elseif strcmp(opt.filtertype, 'UCF');
    T = opt.lambda*D - opt.lambda*(1-opt.psi)*M + opt.psi*(1-opt.lambda)*P;
    
    if isempty(opt.desiredG)
        % by default, use impulses for all training images
        opt.desiredG = ones(size(X));
    end
    
    % solve for (little) p
    p = 1/(prod(N)*Q) * sum(X.*opt.desiredG,2);       % X is a matrix, each column is a FFT of a training image
                                                      % opt.desiredG is a NxQ matrix, each column is the desired correlation 
                                                      % output for that training image, in the freq. domain
    % base filter
    H_base = bsxfun(@rdivide, p,T);      % same as inv(T)*p
    H_base = reshape(H_base, N);
    h_base = real(ifftn(H_base));
    
%% MMCF
% elseif strcmp(opt.filtertype, 'MMCF');
%     % this implementation of MMCF assumes that p is a delta function, always.    
%     Tvect = opt.lambda*D - opt.lambda*(1-opt.psi)*M + opt.psi*(1-opt.lambda)*P;
%     [H_za, H_base] = ZAMMCF(X, A, Tvect, opt.numT, opt.POSSCALE, opt.NEGSCALE, opt.C, opt.usebias, opt.desiredG);
%     
%     h_base = real(ifftn(reshape(H_base, N)));
% %     h_za = real(ifftn(reshape(H_za, N)));
   
%% ASEF
% elseif strcmp(opt.filtertype, 'ASEF');
%     % ASEF's exact filters are UCF filters, but embed this here to avoid
%     % having to put all the code outside ZACF function.
%     
%     % exact filter options
%     Eopt.lambda = 1;
%     Eopt.psi = 1;
%     Eopt.filtertype = 'UCF';
%   
%     % assume all will have the same desired G
%     if opt.var == 0
%         Eopt.desiredG = ones(prod(N),1); %a delta function in space is a flat function in frequency
%     else
%         % center the Gaussian at coordinates Nx1, Nx2
%         gindx = GetMatIndices(N);
%         gim = mvnpdf(gindx,Nx,[opt.var 0; 0 opt.var]);
%         gim = reshape(gim,N);
%         gim = circshift(gim,floor((1-N)/2)); %move the center to the 1,1 location (see thesis notes for explanation)
%         gim = gim/max(gim(:)); %to ensure that the peak value is 1 in space domain
%         gimF = fft2(gim);
%         Eopt.desiredG = gimF(:);
%     end
%     
%     % now train exact filters
%     vo.EF = cell(Q,1);
%     vo.EFZA = cell(Q,1);
%     for q = 1:Q
%         [vo.EFZA{q}, vo.EF{q}] = ZACF({trainset{q}}, N, Eopt, false, false);
%     end
%     
%     % form ASEF and ZAASEF
%     h_base = zeros(size(vo.EF{1}));
%     h_za = zeros(size(vo.EFZA{1}));
%     for q = 1:Q
%         h_base = h_base + (1/Q)*vo.EF{q};
%         h_za = h_za + (1/Q)*vo.EFZA{q};
%     end
%% MOSSE           
elseif strcmp(opt.filtertype, 'MOSSE');
    % MOSSE is just a UCF filter with lambda = psi = 1, with desired
    % output given by a gaussian. Later will update code here to allow user
    % to specify this output.
    
    % I have this separate from UCF so user doesn't have to put in the
    % desired outputs outside of ZACF()
    if isfield(opt, 'lambda')       % useful for when you have zero mean, normalized images, where numerical issues may be present.
        T = opt.lambda*D + (1-opt.lambda)*P;
    else
        T = D;
    end
        
  
    % assume all will have the same desired G
    if opt.var == 0
        opt.desiredG = ones(prod(N),1); %a delta function in space is a flat function in frequency
    else
        % center the Gaussian at coordinates Nx1, Nx2
        gindx = GetMatIndices(N);
        gim = mvnpdf(gindx,Nx,[opt.var 0; 0 opt.var]);
        gim = reshape(gim,N);
        gim = circshift(gim,floor((1-N)/2)); %move the center to the 1,1 location (see thesis notes for explanation)
        gim = gim/max(gim(:)); %to ensure that the peak value is 1 in space domain
        gimF = fft2(gim);
        opt.desiredG = gimF(:);
    end
    
    % solve for (little) p
    p = 1/(prod(N)*Q) * sum(X.*repmat(opt.desiredG,1,Q),2);       % X is a matrix, each column is a FFT of a training image
                                                      % opt.desiredG is a NxQ matrix, each column is the desired correlation 
                                                      % output for that training image, in the freq. domain

    H_base = bsxfun(@rdivide, p,T);      % same as inv(D)*p
    H_base = reshape(H_base, N);
    h_base = real(ifftn(H_base));

else
    %% Error
    error('Unspecified filter requested');
end
%% final output stuff
% copy the outputs (this is so user can have originals and not call ZACF() multiple times).
CNh.h_base = reshape(h_base,N);

if crop
    if numel(Nx) == 1
        CNh.h_base = CNh.h_base(1:Nx(1));
    elseif numel(Nx) == 2
        CNh.h_base = CNh.h_base(1:Nx(1), 1:Nx(2));
    else
        error('Crop feature not supported for 3D or higher training signals at this time');
    end
end

if normalize
    if numel(Nx) > 2
        error('Normalize feature not supported for 3D or higher training signals at this time');
    end
    CNh.h_base = CNh.h_base/sqrt(sum(sum(abs(CNh.h_base).^2)));
end