% script to test idea for training filters using column/row tail
% minimization
close all
clear all
clc

% load in 1D data
% load('beatdata.mat', 'data');
% data = data.';      % put it so each heartbeat is a column.
% trainset1D = cell(10,1);
% for i0 = 1:10
%     trainset1D{i0,1} = data(:,i0+10);
% end
% clear('data');


% load in some images to train filters.
load('C:\Users\jafernan\Documents\My Research\Surveillance\Matlab\3dFFT\timedomainMACE\ORLface_norm_scale_0.25.mat')
trainset = facedata(1,2:10); testim = facedata{1,10};
Nx = size(trainset{1});
N = size(trainset{1})*2-1;

%% train MACE filter with new function
% % temp N
% p = 0;
% % method 2a (see notebook pp 11)
% Szaa = ones(N(1),N(2));
% if p > 0
%     Szaa(Nx(1)+1:min(Nx(1)+p, N(1)), 1:min(Nx(2)+p, N(2))) = 0;
%     Szaa(1:min(Nx(1)+p, N(1)), Nx(2)+1:min(Nx(2)+p, N(2))) = 0;
% end
% 
% % method 2b (see notebook pp 11)
% Szab = ones(N(1),N(2));
% if p > 0
%     Szab(max(N(1)-(p-1), Nx(1)+1):N(1),:) = 0;
%     Szab(:, max(N(2)-(p-1), Nx(2)+1):N(2)) = 0;
% end
% 
% opt.filtertype = 'OTSDF';
% opt.alpha = 0.5;
% opt.u = [1,1,1];

% [h_za, h_base] = ZACF(trainset, N, opt, false, false, Sza(:));  %<-- remove Sza input to do conventional ZACF


% train several filters

% %------------ mace, padded with 0x0
% [h_za_m1_1, h_base1] = ZACF(trainset, Nx, opt, false, false);  %<-- remove Sza input to do conventional ZACF
% %------------ mace, padded with 27x22
% trainsetrand = cell(1,3);
% trainsetrand{1,1} = randn(28,23);
% trainsetrand{1,2} = randn(28,23);
% trainsetrand{1,3} = randn(28,23);
% 
% [h_za_m1_2, h_base2] = ZACF_play(trainset, N, opt, false, false);  %<-- remove Sza input to do conventional ZACF
% %------------ custom padded filter, method 2, (each edge)
% p = 0;
% % method 2a (see notebook pp 11)
% Szaa = ones(N(1),N(2));
% if p > 0
%     Szaa(Nx(1)+1:min(Nx(1)+p, N(1)), 1:min(Nx(2)+p, N(2))) = 0;
%     Szaa(1:min(Nx(1)+p, N(1)), Nx(2)+1:min(Nx(2)+p, N(2))) = 0;
% end
% % method 2b (see notebook pp 11)
% Szab = ones(N(1),N(2));
% if p > 0
%     Szab(max(N(1)-(p-1), Nx(1)+1):N(1),:) = 0;
%     Szab(:, max(N(2)-(p-1), Nx(2)+1):N(2)) = 0;
% end
% [h_za_m2a_1] = ZACF(trainset, N, opt, false, false, Szaa(:));  %<-- remove Sza input to do conventional ZACF
% [h_za_m2b_1] = ZACF(trainset, N, opt, false, false, Szab(:));  %<-- remove Sza input to do conventional ZACF
% 
% 
% p = 27;
% % method 2a (see notebook pp 11)
% Szaa = ones(N(1),N(2));
% if p > 0
%     Szaa(Nx(1)+1:min(Nx(1)+p, N(1)), 1:min(Nx(2)+p, N(2))) = 0;
%     Szaa(1:min(Nx(1)+p, N(1)), Nx(2)+1:min(Nx(2)+p, N(2))) = 0;
% end
% % method 2b (see notebook pp 11)
% Szab = ones(N(1),N(2));
% if p > 0
%     Szab(max(N(1)-(p-1), Nx(1)+1):N(1),:) = 0;
%     Szab(:, max(N(2)-(p-1), Nx(2)+1):N(2)) = 0;
% end
% [h_za_m2a_2] = ZACF(trainset, N, opt, false, false, Szaa(:));  %<-- remove Sza input to do conventional ZACF
% [h_za_m2b_2] = ZACF(trainset, N, opt, false, false, Szab(:));  %<-- remove Sza input to do conventional ZACF
% 

% figure;
% subplot 241
% imagesc(h_base1); title('MACE, 0x0 padding');
% subplot 245
% imagesc(h_base2); title('MACE, 27x22 padding');
% subplot 242
% imagesc(h_za_m1_1); title('ZAMACE method 1, 0x0 padding');
% subplot 246
% imagesc(h_za_m1_2); title('ZAMACE method 1, 27x22 padding');
% subplot 243
% imagesc(h_za_m2a_1); title('ZAMACE method 2a, p = 0');
% subplot 247
% imagesc(h_za_m2a_2); title('ZAMACE method 2a, p = 27');
% subplot 244
% imagesc(h_za_m2b_1); title('ZAMACE method 2b, p = 0');
% subplot 248
% imagesc(h_za_m2b_2); title('ZAMACE method 2b, p = 27');
% % 
% [h_za, h_base] = ZACF(trainset, N, opt, false, false, Sza(:));  %<-- remove Sza input to do conventional ZACF
% 
% 
% 
% figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% subplot 121
% imagesc(h_base); title('MACE'); colorbar;
% subplot 122
% imagesc(h_za); title('ZAMACE'); colorbar;



%% train OTSDF filter with new function
% opt.filtertype = 'OTSDF';
% opt.u = ones(size(trainset));
% opt.alpha = 0.9;
% opt.method = 'standard'
% [h_za, h_base] = ZACF(trainset, N, opt, false, false);
% 
% figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% subplot 121
% imagesc(h_base); title(['OTSDF, \alpha = ' num2str(opt.alpha, 9)]); colorbar;
% subplot 122
% imagesc(h_za); title(['ZAOTSDF, \alpha = ' num2str(opt.alpha, 9)]); colorbar;
% saveas(gcf, [cd filesep 'images_for_different_ZAfilters' filesep 'otsdf_alpha_' num2str(opt.alpha, 9) '.eps'])
% saveas(gcf, [cd filesep 'images_for_different_ZAfilters' filesep 'otsdf_alpha_' num2str(opt.alpha, 9) '.tiff'])
% saveas(gcf, [cd filesep 'images_for_different_ZAfilters' filesep 'otsdf_alpha_' num2str(opt.alpha, 9) '.fig'])

% troubleshooting code for MVSDF, P
% [ones(floor(size(D,1)/2),1); 0.001*ones(ceil(size(D,1)/2),1)];

%% UCF vs. CCF vs. OTSDF code
% opt.filtertype = 'UCF';
% opt.psi = 1;
% opt.lambda = 0.5;
% opt.desiredG = [];
% 
% opt2.filtertype = 'CCF';
% opt2.psi = 1;
% opt2.lambda = opt.lambda;
% opt2.u = [1,1,1];
% opt2.desiredG = [];
% 
% opt3.filtertype = 'OTSDF';
% opt3.u = [1,1,1];
% opt3.alpha = opt.lambda;
% 
% tic
% [ucf_za, ucf_base] = ZACF(trainset, N, opt, false, false); toc; tic;
% [ccf_za, ccf_base] = ZACF(trainset, N, opt2, false, false); toc; tic;
% [otsdf_za, otsdf_base] = ZACF(trainset, N, opt3, false, false); toc;
% 
% ucf_base = ucf_base(1:Nx(1), 1:Nx(2));
% ucf_base = ucf_base/norm(ucf_base(:));
% ccf_base = ccf_base(1:Nx(1), 1:Nx(2));
% ccf_base = ccf_base/norm(ccf_base(:));
% ucf_za = ucf_za(1:Nx(1), 1:Nx(2));
% ucf_za = ucf_za/norm(ucf_za(:));
% ccf_za = ccf_za(1:Nx(1), 1:Nx(2));
% ccf_za = ccf_za/norm(ccf_za(:));
% otsdf_base = otsdf_base(1:Nx(1), 1:Nx(2));
% otsdf_base = otsdf_base/norm(otsdf_base(:));
% otsdf_za = otsdf_za(1:Nx(1), 1:Nx(2));
% otsdf_za = otsdf_za/norm(otsdf_za(:));

% 
% figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% subplot 121
% imagesc(ucf_base); title(['UCF, \lambda = ' num2str(opt.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% subplot 122
% imagesc(ucf_za); title(['ZAUCF, \lambda = ' num2str(opt.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% 
% figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% subplot 121
% imagesc(ccf_base); title(['CCF, \lambda = ' num2str(opt2.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% subplot 122
% imagesc(ccf_za); title(['ZACCF, \lambda = ' num2str(opt2.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% 
% figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% subplot 121
% imagesc(otsdf_base); title(['OTSDF, \alpha = ' num2str(opt3.alpha, 9)]); colorbar;
% subplot 122
% imagesc(otsdf_za); title(['ZAOTSDF, \alpha = ' num2str(opt3.alpha, 9)]); colorbar;
% 
% 
% 
% for i0 = 1:numel(trainset)
%     figure;
%     subplot 221
%     out = fftcorr2(ucf_base(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of UCF and training image ' num2str(i0)]);
%     subplot 222
%     out = fftcorr2(ucf_za(1:Nx(1), 1:Nx(2)), trainset{i0}); %AFR
%     surf(out);
%     title(['output of ZAUCF and training image ' num2str(i0)]); %AFR
%     subplot 223
%     out = fftcorr2(ccf_base(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of CCF and training image ' num2str(i0)]);
%     subplot 224
%     out = fftcorr2(ccf_za(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of ZACCF and training image ' num2str(i0)]);
%     set(gcf, 'Position', [72, -108, 1714, 892]);
% end

%% CCF vs. CCF prox gradient
opt.filtertype = 'CCF';         % UCF or CCF
opt.psi = 1;
opt.lambda = 0.9;
opt.u = ones(numel(trainset),1);
opt.desiredG = [];
opt.method = 'load';
tic
[h_za_CF,~,h_base] = ZACF(trainset, N, opt, 0,0);
toc

% use this code for doing proximal method.
iter = 1;
stoplist = Inf
for stop = stoplist
    
    opt2 = opt;
    opt2.filtertype = opt.filtertype;
    opt2.method = 'proxacc';    % prox (UCF only), proxacc, or standard
    opt2.proxopt.t_init = 100;
    opt2.proxopt.beta = 0.5;
    opt2.proxopt.maxiter = stop;
    opt2.proxopt.tol = 1e-8;
    tic
    [h_za_PAC,~,~,vo2] = ZACF(trainset, N, opt2, 0,0);
    toc
    numel(vo2.proxinfo.tlist)
    


    h_za_CF = h_za_CF/sqrt(sum(sum(abs(h_za_CF).^2)));
    h_za_PAC = h_za_PAC/sqrt(sum(sum(abs(h_za_PAC).^2)));
    mse(iter) = sum(sum(abs(h_za_CF-h_za_PAC).^2))
    iter = iter+1;
end

for q = 1:numel(trainset)
    dotprod = sum(sum(h_za_CF(1:Nx(1),1:Nx(2)).*trainset{q}));
    display(['Dot product of h_za_CF with training image ' num2str(q) ': ' num2str(dotprod)]);
end
for q = 1:numel(trainset)
    dotprod = sum(sum(h_za_PAC(1:Nx(1),1:Nx(2)).*trainset{q}));
    display(['Dot product of h_za_PAC with training image ' num2str(q) ': ' num2str(dotprod)]);
end


figure; 
subplot 121
imagesc(h_za_CF);   title('closed form');
subplot 122
imagesc(h_za_PAC);  title('proximal gradient descent');

figure; semilogy(stoplist, mse); grid on; title('MSE as function of iteration number');

figure; 
subplot 121
surf(fftcorr2(h_za_CF(1:Nx(1), 1:Nx(2)), trainset{1}));
title('Correlaton output, closed form');
subplot 122
surf(fftcorr2(h_za_PAC(1:Nx(1), 1:Nx(2)), trainset{1}));
title('Correlaton output, prox grad descent');

figure; 
subplot 121
surf(fftcorr2(h_za_CF(1:Nx(1), 1:Nx(2)), testim));
title('Correlaton output, closed form, true class (not in trainset)');
subplot 122
surf(fftcorr2(h_za_PAC(1:Nx(1), 1:Nx(2)), testim));
title('Correlaton output, prox grad descent, true class (not in trainset)');


figure; 
subplot 121
surf(fftcorr2(h_za_CF(1:Nx(1), 1:Nx(2)), facedata{4,7}));
title('Correlaton output, closed form, false class (not in trainset)');
subplot 122
surf(fftcorr2(h_za_PAC(1:Nx(1), 1:Nx(2)), facedata{4,7}));
title('Correlaton output, prox grad descent, false class (not in trainset)');


%% MMCF code
% opt.filtertype = 'MMCF';
% opt.numT = numel(trainset);
% opt.POSSCALE = 1;
% opt.NEGSCALE = 1;
% opt.C = Inf;
% opt.lambda = 0.5;
% opt.psi = 1;
% opt.usebias = 0;
% opt.desiredG = [];       % use 0 for excluding p term, [] for default (impulse).
% 
% tic
% [mmcf_za, mmcf_base] = ZACF(trainset, N, opt, false, false); toc; tic;
% 
% mmcf_base = mmcf_base(1:Nx(1), 1:Nx(2));
% mmcf_base = mmcf_base/norm(mmcf_base(:));
% mmcf_za = mmcf_za(1:Nx(1), 1:Nx(2));
% mmcf_za = mmcf_za/norm(mmcf_za(:));
% 
% figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% subplot 121
% imagesc(mmcf_base); title(['MMCF, \lambda = ' num2str(opt.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% subplot 122
% imagesc(mmcf_za); title(['ZAMMCF, \lambda = ' num2str(opt.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% 
% for i0 = 1:numel(trainset)
%     figure;
%     subplot 121
%     out = fftcorr2(mmcf_base(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of MMCF and training image ' num2str(i0)]);
%     subplot 122
%     out = fftcorr2(mmcf_za(1:Nx(1), 1:Nx(2)), trainset{i0}); %AFR
%     surf(out);
%     title(['output of ZAMMCF and training image ' num2str(i0)]); %AFR
% end
%% ASEF
% opt.filtertype = 'ASEF';
% opt.var = 1;
% 
% [asef_za, asef_base] = ZACF(trainset, N, opt, false, false);
% 
% 
% figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% subplot 121
% imagesc(asef_base); title(['ASEF var = ' num2str(opt.var, 9)]); colorbar;
% subplot 122
% imagesc(asef_za); title(['ASEF, var = ' num2str(opt.var, 9)]); colorbar;

% for i0 = 1:numel(trainset)
%     figure;
%     subplot 121
%     out = fftcorr2(asef_base(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of ASEF and training image ' num2str(i0)]);
%     subplot 122
%     out = fftcorr2(asef_za(1:Nx(1), 1:Nx(2)), trainset{i0}); %AFR
%     surf(out);
%     title(['output of ASEF and training image ' num2str(i0)]); %AFR
% end

%% MOSSE
% opt.filtertype = 'MOSSE';
% opt.var = 1;
% 
% [mosse_za, mosse_base] = ZACF(trainset, N, opt, false, false);
% 
% 
% figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% subplot 121
% imagesc(mosse_base); title(['MOSSE var = ' num2str(opt.var, 9)]); colorbar;
% subplot 122
% imagesc(mosse_za); title(['MOSSE, var = ' num2str(opt.var, 9)]); colorbar;
% 
% 
% for i0 = 1:numel(trainset)
%     figure;
%     subplot 121
%     out = fftcorr2(mosse_base(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of MOSSE and training image ' num2str(i0)]);
%     subplot 122
%     out = fftcorr2(mosse_za(1:Nx(1), 1:Nx(2)), trainset{i0}); %AFR
%     surf(out);
%     title(['output of MOSSE and training image ' num2str(i0)]); %AFR
% end

%% MSESDF
% msesdf is a type of ccf, use lambda = psi = 1, desired output is an
% impulse.
% opt.filtertype = 'CCF';
% opt.psi = 1;
% opt.lambda = 1;
% opt.u = [1,1,1];
% % opt.desiredG = [];      % default, impulse.
% opt.var=5;
% [ccf_za, ccf_base] = ZACF(trainset, N, opt, false, false);
% 
% ccf_base = ccf_base(1:Nx(1), 1:Nx(2));
% ccf_base = ccf_base/norm(ccf_base(:));
% 
% figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% subplot 121
% imagesc(ccf_base); title(['CCF, \lambda = ' num2str(opt.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% subplot 122
% imagesc(ccf_za); title(['ZACCF, \lambda = ' num2str(opt.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% 
% 
% for i0 = 1:numel(trainset)
%     figure;
%     subplot 121
%     out = fftcorr2(ccf_base(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of CCF and training image ' num2str(i0)]);
%     subplot 122
%     out = fftcorr2(ccf_za(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of ZACCF and training image ' num2str(i0)]);
%     set(gcf, 'Position', [72, -108, 1714, 892]);
% end
%% MACH
% opt.filtertype = 'UCF';
% opt.psi = 1;
% opt.lambda = 0.9;
% opt.desiredG = [];      % default, impulse.
% 
% % use this code for doing proximal method.
% opt2 = opt;
% opt2.method = 'prox';    % prox = gradient descent, otherwise, not including opt.method will just do the standard computation.
% opt2.proxopt.t_init = 100;
% opt2.proxopt.alpha = 0.3;
% opt2.proxopt.beta = 0.5;
% opt2.proxopt.maxiter = inf;
% opt2.proxopt.tol = 1e-5;
% 
% % [ucf_za, ucf_base] = ZACF(trainset, N, opt, false, false);
% [ucf_za2, ucf_base2] = ZACF(trainset, N, opt2, false, false);
% 
% % ucf_base = ucf_base(1:Nx(1), 1:Nx(2));
% % ucf_base = ucf_base/norm(ucf_base(:));
% 
% % figure; set(gcf, 'Position', [300, 400, 1330, 420]);
% % subplot 121
% % imagesc(ucf_base); title(['UCF, \lambda = ' num2str(opt.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% % subplot 122
% % imagesc(ucf_za); title(['ZAUCF, \lambda = ' num2str(opt.lambda, 9) ' \psi = ' num2str(opt.psi, 9)]); colorbar;
% 
% figure;
% subplot 121
% imagesc(ucf_base2); title(['UCF (prox), \lambda = ' num2str(opt2.lambda, 9) ' \psi = ' num2str(opt2.psi, 9)]); colorbar;
% subplot 122
% imagesc(ucf_za2); title(['ZAUCF (prox), \lambda = ' num2str(opt2.lambda, 9) ' \psi = ' num2str(opt2.psi, 9)]); colorbar;

% for i0 = 1:numel(trainset)
%     figure;
%     subplot 121
%     out = fftcorr2(ucf_base(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of MACH and training image ' num2str(i0)]);
%     subplot 122
%     out = fftcorr2(ucf_za(1:Nx(1), 1:Nx(2)), trainset{i0});
%     surf(out);
%     title(['output of ZAMACH and training image ' num2str(i0)]);
%     set(gcf, 'Position', [72, -108, 1714, 892]);
% end

%% OTSDF 1D
% opt.filtertype = 'OTSDF';
% opt.u = ones(10,1);
% opt.alpha = 0.1;
% [h_za, h_base] = ZACF(trainset1D, 601, opt, false, false);
% 
% figure;
% plot(h_base);
% hold on;
% plot(h_za, 'r');
% plot(h_za2, 'g');
% legend('OTSDF', 'ZA OTSDF', 'ZA2 OTSDF');

%% MACE with different padding code
% opt.filtertype = 'MACE';
% opt.alpha = 1;
% opt.u = [1,1,1];
% [h_za1, h_base1] = ZACF(trainset, [28 23], opt, false, false); 
% [h_za2, h_base2] = ZACF(trainset, [42 35], opt, false, false);
% [h_za3, h_base3] = ZACF(trainset, [55 45], opt, false, false);
% 
% p = 10;
% Sza = ones(N(1),N(2));
% if p > 0
%     Sza(max(N(1)-(p-1), Nx(1)+1):N(1),:) = 0;
%     Sza(:, max(N(2)-(p-1), Nx(2)+1):N(2)) = 0;
% end
% Sza = Sza(:);
% h_za_m2 = ZACF(trainset, [55 45], opt, false, false, Sza);
% 
% globalmin = min([min2d(h_base1), min2d(h_base2), min2d(h_base3)]);
% globalmax = max([max2d(h_base1), max2d(h_base2), max2d(h_base3)]);
% 
% figure; 
% imshow(h_base1, [min2d(h_base1), max2d(h_base1)], 'InitialMagnification', 1000)
% colormap(jet)
% 
% figure; 
% imshow(h_base2, [min2d(h_base2), max2d(h_base2)], 'InitialMagnification', 1000)
% colormap(jet)
% 
% figure; 
% imshow(h_base3, [min2d(h_base3), max2d(h_base3)], 'InitialMagnification', 1000)
% colormap(jet)
% %---za
% figure; 
% imshow(h_za1, [min2d(h_za1), max2d(h_za1)], 'InitialMagnification', 1000)
% colormap(jet)
% 
% figure; 
% imshow(h_za2, [min2d(h_za2), max2d(h_za2)], 'InitialMagnification', 1000)
% colormap(jet)

% figure; 
% imshow(h_za3, [min2d(h_za3), max2d(h_za3)], 'InitialMagnification', 1000)
% colormap(jet)
% 
% figure; 
% imshow(h_za_m2, [min2d(h_za_m2), max2d(h_za_m2)], 'InitialMagnification', 1000)
% colormap(jet)

