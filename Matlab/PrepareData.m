clear;
close all;

%%
%Select phantom  
phantom = 2;

if phantom == 1,
    copyfile heartNlungs_phantom1.m heartNlungs.m
    tmpl = 'L1H02';
else
    copyfile heartNlungs_phantom2.m heartNlungs.m
    tmpl = 'L5H2';
end
n = 2;
phs = 4; %phase [0,7]
phase =phs*pi/8;

heartNlungs_plot;
%%
currentFolder = pwd;

pt = sprintf('%s/../Data/InputData',currentFolder);
file = sprintf('Grid1k%sn%d_phase_%d.mat',tmpl,n,phs);
%%

NumOfRefine = 7; %number of mesh refinement iteration
marg = 10;       %image margin
N = 1000;        % u is NxN matrix
Nb = 12000;      % number of boundary points
noiseUb = 0;    % boundary noise
save data/BoundaryDataN n phase;
global GlobalSigma;
verbose = 1;
%%
%Preparing mesh
g = 'circleg';
[p,e,t]=initmesh(g);

for i=1:NumOfRefine
    [p,e,t]=refinemesh(g,p,e,t);
end


%%
%Arrange Sigma as GlobNxGlobN matrix
GlobalN = 1000;
xx = linspace(-1,1,GlobalN);
yy = linspace(-1,1,GlobalN);
GlobalSigma = zeros(GlobalN,GlobalN); 
   
for ii=1:GlobalN
    for jj=1:GlobalN
        GlobalSigma(jj,ii) = heartNlungs(xx(ii)+1i*yy(jj));
    end
end
fil = fspecial('gaussian',200,20); 
SigmaSmooth = imfilter(GlobalSigma,fil,'symmetric','same');
GlobalSigma = SigmaSmooth;
[Globalsx,Globalsy] = gradient(GlobalSigma);
save('Sigma1000.mat','GlobalSigma'); 
if verbose,
    figure; imagesc(GlobalSigma);axis equal;title('sigma'); colorbar;
end


%%
% solving the equation by the finite element method
u = assempde('BoundaryData',p,e,t,'FEMconductivitySmooth',0,0);

[~,g,~,~] = BoundaryData(p,e); 

if verbose,
    figure;
    pdesurf(p,t,(u))
    title('u');
    drawnow
end
%%
%Represent the finite element method solution to caresian grid
[center,h,radius,U1,G1,dudn1,sdudn1,W,Sigma,n_hat_x,n_hat_y,phi,XOmeg,YOmeg,U1nan,Omega,XOmegSmall,YOmegSmall,XOmegWide,YOmegWide,sx,sy] = ...
    ExtractImages(p,e,t,u,g,N,marg,Globalsx,Globalsy);



CorrectS = (0.5*(max(XOmeg)-min(XOmeg)));
CorrectB = (min(XOmeg)+CorrectS);
U1 = U1.*Omega;
if verbose
    figure;
    subplot(121); imagesc(Omega); axis equal;title('Omega');
    subplot(122); imagesc(W); axis equal;title('W');
    figure; imagesc(U1);axis equal; title('U1');
end


%%
Xnor = (XOmeg-CorrectB)/CorrectS;
Ynor = (YOmeg-CorrectB)/CorrectS;
factor = 8;

g = zeros(N); %the current

for ii=1:length(Xnor)
    xx = XOmeg(ii); yy = YOmeg(ii);
    theta = angle(Xnor(ii)+1i*Ynor(ii));
    g(yy,xx) = real( factor* (1/sqrt(2*pi)*exp(1i*n*(theta+phase))));
end
G = g.*W;
if verbose
    figure; 
    subplot(121); imagesc(G);title('current g');axis equal; colorbar;
    subplot(122); imagesc((U1).*W);title('voltage on the boundary u0');axis equal; colorbar;
end
%%
%Preparing the gradients, with a special care on the boundary

[ux,uy] = BoundaryGradient_ver1(U1,W,Omega,CorrectB,CorrectS);
[uxx,~] = BoundaryGradient_ver1(ux,W,Omega,CorrectB,CorrectS);
[~,uyy] = BoundaryGradient_ver1(uy,W,Omega,CorrectB,CorrectS);

if verbose,
    figure; 
    subplot(121); imagesc(ux);axis equal; title('ux');
    subplot(122); imagesc(uy);axis equal; title('uy');
end

%%
%Boundary conditions
rng(1);
ix = find(W>0);
ix1 = randperm(length(ix),min(Nb,length(ix)));
ix2 = ix(ix1);
[Yb,Xb] = ind2sub(size(W),ix2);
Ub = U1(ix2);

if noiseUb > 0,
   Ub = Ub+noiseUb*rand(size(Ub));
end


Gb = G(ix2);
n_hat_xb = n_hat_x(ix2);
n_hat_yb = n_hat_y(ix2);
A1 = zeros(size(W));
A2 = zeros(size(W));
for i=1:length(ix2)
    A1(Yb(i),Xb(i))=Ub(i);
    A2(Yb(i),Xb(i))=Gb(i);
end
if verbose,
    figure; 
    subplot(121); imagesc(A1);axis equal;title('Dirichlet Boundary conditions');
    subplot(122); imagesc(A2);axis equal;title('Neumann Boundary conditions');
end
%%
if verbose,
    NOmega = 1000;
    Nbound = 400;
    ix = randperm(length(XOmeg),NOmega);
    xx = XOmeg(ix); yy = YOmeg(ix);
    figure;
    plot(xx,yy,'bx'); hold on; plot(Xb(1:Nbound),Yb(1:Nbound),'rx');axis equal; axis off;
end
%%
fprintf('saving %s\n',fullfile(pt,file));
save(fullfile(pt,file),'W','Sigma','sx','sy','U1','ux','Omega','XOmeg','YOmeg','h','Xb','Yb','Ub','Gb','CorrectS','CorrectB','n_hat_xb','n_hat_yb');
 

 

 
 
