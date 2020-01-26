% (C) 2014 Joseph Fernandez. Please see license.txt for license information.
%

% buildMMCFbias.m
% Author: Joseph Fernandez and Andres Rodriguez
% Date: August 23, 2013
% This function returns the MMCF and ZAMMCF filters

function [h_za h_base a] = ZAMMCF(Xmat, A, Tvect,numT,POSSCALE, NEGSCALE,C, usebias, desiredG)
% Xmat - training matrix
% T - weighted matrix (trade off ratio) for adjusting the filter by ONV and ACE
% numT - number of true class images
% numF - number of false class images
% Fout - weight for false class

%% administrative stuff
if ~exist('C','var'); C=Inf; end
if ~exist('lam','var'); lam=1;end
if ~exist('sig','var'); sig=1; end
[d numim] = size(Xmat);     % get the total dimensionality
numF = numim-numT;          % number of false class images

if POSSCALE < 0
    error('POSSCALE must be positive');
end
if NEGSCALE < 0
    error('NEGSCALE must be positive');
end
    

%% compute the UCF filter
% assumes p = mean of images - this is only true for when there is only
% positive class images. If there are negative class images, then we need
% to weight the negative samples by some amount, which is -NEGSCALE. We
% assume the desired output is always a delta function.
weightvect = [POSSCALE*ones(numT,1); -NEGSCALE*ones(numF,1)];       % added 10/2/13
Xweightedsum = (1/d)*(1/numim)*Xmat*weightvect;     % this was formerly p = (1/d)*mean(Xmat,2);

if isempty(desiredG)
    % by default, use impulses for all training images
    desiredG = ones(size(Xweightedsum));
elseif desiredG == 0
    desiredG = zeros(size(Xweightedsum));   % this effectively removes the h'*p term
else
    % code here for making sure the peak doesn't account for class...
    % check size of the desired output.
    error('This feature not supported yet');    
end
p = Xweightedsum.*desiredG;


hUCF = p.*Tvect.^-1;        % used to be hUCF = p.*Tvect;

% h_UCF_andres = buildUCF(Xmat, Tvect, numT, -NEGSCALE, ones(size(Xmat,1),1), 0);        % lambda and sigma here appear to be different from andres thesis pp 86, eq 3.10

%% Add Origin as false class image
% code to add the origin as a false class image: if there are no false
% class images, or if it is hard coded to do so
ADD_ORIGIN = 0; %if 1 adds origin as an additional false-class point
if numF == 0 || ADD_ORIGIN
    Xmat(:,end+1) = zeros(d,1);
    numF = numF+1;
    numim = numim+1;
end

%% Set up variables for quadprog
Lvect = [ones(numT,1); -1*ones(numF,1)]; % labels (note: L = diag(Lvect)
u = d*[POSSCALE*ones(numT,1); -NEGSCALE*ones(numF,1)];     % peak values: +POSSCALE for positive class, -NEGSCALE for negative class

% intermediate variables used for MMCF and ZAMMCF
XL = Xmat*diag(Lvect);                          % XL contains negated images for false class                   
TinvXL = bsxfun(@rdivide, XL, Tvect);           % this is when T is a vector, represented as a

% some variables useful for ZAMMCF only
Aw = bsxfun(@rdivide, A,Tvect);      % same as inv(T)*A
Kw = real(A'*Aw);
Rw = chol(Kw);   
R = (Rw\(Rw'\A'));
% miniDelt2 = eye(d)-diag(Tvect.^-1)*A*R;  % slower
miniDelt = eye(d) - bsxfun(@rdivide, A,Tvect)*R;

%% MMCF

%---------------inputs for quadprog, MMCF
H = XL'*TinvXL;
H=real(H); H=(H'+H)/2; % although H is symmetric & real this avoids numerical errors

uUCF = real(XL'*hUCF);      % uUCF = L*X'*Tinv*p
Lu = Lvect.*u;
f = uUCF-Lu;       % negative of term L*u - X'*Tinv*p

%---------------Quad Program, MMCF
epsilon = 1e-6;
if sum(abs(f))<epsilon, f=epsilon*u; end
lb = zeros(numim,1);
ub = C*ones(numim,1);
options = optimset('LargeScale','off','Display','off','MaxIter',100000, 'Algorithm', 'active-set'); 
% options = optimset('LargeScale','off','Display','off','MaxIter',100000); % <<---- old setup

if usebias == 1
    a = quadprog(H,f,[],[],Lvect',0,lb,ub,[],options); %min 0.5*alp'*H*alp + f'*alp s.t. 0<=alp, y'*alp=0
else
    a = quadprog(H,f,[],[],[],[],lb,ub,[],options); %min 0.5*alp'*H*alp + f'*alp s.t. 0<=alp, y'*alp=0
end
h = TinvXL*a; % note that Yww contains negated training images already
h_base = h+hUCF; % This is the base filter, MMCF.

epsilon = 1e-5;

%% ZAMMCF

%---------------inputs for quadprog, ZAMMCF
M = -XL'*miniDelt*TinvXL;
M=real(M); M=(M'+M)/2; % although M is symmetric & real this avoids numerical errors
b = 2*(Lu - XL' * miniDelt * hUCF);   % was formerly  b = 2*(Lu + XL' * miniDelt * hUCF);
b = real(b);        % b should be real, so eliminate rounding errors
%---------------Quad Program, ZAMMCF
epsilon = 1e-6;
if sum(abs(b))<epsilon, b=epsilon*u; end
lb = zeros(numim,1);
ub = C*ones(numim,1);
options = optimset('LargeScale','off','Display','off','MaxIter',100000, 'Algorithm', 'active-set'); 
% options = optimset('LargeScale','off','Display','off','MaxIter',100000); % <<---- old setup
if usebias == 1
    a = quadprog(-2*M,-b,[],[],Lvect',0,lb,ub,[],options); %min 0.5*a'*M*a + b'*a s.t. 0<=a, y'*a=0
else
    a = quadprog(-2*M,-b,[],[],[],[],lb,ub,[],options); %min 0.5*a'*M*a + b'*a s.t. 0<=a, y'*a=0
end
omega = -R*(hUCF + TinvXL*a);
h_za = hUCF + TinvXL*a + Tvect.^-1 .* (A*omega); % note that Yww contains negated training images already

