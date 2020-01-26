close all
clear all
clc

% load in some images to train filters.
load('C:\Users\jafernan\Documents\My Research\Surveillance\Matlab\3dFFT\timedomainMACE\ORLface_norm_scale_0.25.mat')
trainset = facedata(1,1:3); testim = facedata{1,10};
Nx = size(trainset{1});
N = size(trainset{1})*2-1;


%% CCF vs. CCF prox gradient
opt.filtertype = 'CCF';         % UCF or CCF
opt.psi = 1;
opt.lambda = 0.9;
opt.u = ones(numel(trainset),1);
opt.desiredG = [];

% this next option is either "standard", "load", or "proxacc".  Standard forms 
% the A matrix from scratch, which takes a long time. Load will look for
% a mat file that contains the matrix. if it doesn't exist, it will
% generate it. Typically, loading in the matrix from a mat file is a lot
% faster (assuming the mat file already exists), so it's the preferred 
% closed form method. Proxacc (see below) invokes the numerical proximal 
% gradient method, which will be the fastest option.
opt.method = 'load';        

% builds the ZACF. h_base is the standard filter.
tic
[h_za_CF,h_base] = ZACF(trainset, N, opt, 0,0);
toc

% this code shows how to invoke the proximal gradient method.
opt2 = opt;
opt2.filtertype = opt.filtertype;
opt2.method = 'proxacc';    % accelerated proximal gradient descent
opt2.proxopt.t_init = 100;  % keep t_init and beta the same as this
opt2.proxopt.beta = 0.5;
opt2.proxopt.maxiter = Inf;     % depending on the situation, sometimes I make this a finite number. It gives total number of iterations algorithm is allowed to run.
opt2.proxopt.tol = 1e-8;        % tolerance for error on stop condition
tic
[h_za_PAC,~,~,vo2] = ZACF(trainset, N, opt2, 0,0);
toc
    
% normalize templates
h_za_CF = h_za_CF/sqrt(sum(sum(abs(h_za_CF).^2)));
h_za_PAC = h_za_PAC/sqrt(sum(sum(abs(h_za_PAC).^2)));
h_base = h_base/sqrt(sum(sum(abs(h_base).^2)));

% plot the templates
figure; 
subplot 131
imagesc(h_base); title('Original Formulation');
subplot 132
imagesc(h_za_CF);   title('closed form');
subplot 133
imagesc(h_za_PAC);  title('proximal gradient descent');

% plot correlation outputs for TRAINING images)
figure; 
subplot 131
surf(fftcorr2(h_base(1:Nx(1), 1:Nx(2)), trainset{1}));
title('Correlaton output, original formulation');
subplot 132
surf(fftcorr2(h_za_CF(1:Nx(1), 1:Nx(2)), trainset{1}));
title('Correlaton output, closed form');
subplot 133
surf(fftcorr2(h_za_PAC(1:Nx(1), 1:Nx(2)), trainset{1}));
title('Correlaton output, prox grad descent');



