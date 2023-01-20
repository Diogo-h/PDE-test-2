function [] = PackU()

    close all;
    tmpl = 'L5H2';
    %tmpl = 'L1H02';
    
   Orig=0;

    currentFolder = pwd;
    pt = sprintf('%s/../Data/InputData',currentFolder);
    ptout = sprintf('%s/../Data/OutputData',currentFolder);
    
      
    fileName = sprintf('Grid1k%sn1_phase_0.mat',tmpl);
    filebase = fullfile(pt,fileName);

    f1 = load(filebase);
    W = f1.W;
    Omega = f1.Omega;
    XOmeg = f1.XOmeg;
    YOmeg = f1.YOmeg;
    Sigma = f1.Sigma;
    h = f1.h;
    CorrectB = f1.CorrectB;
    CorrectS = f1.CorrectS;
    
    rng(1);
    ix = find(W>0);
    ix1 = randperm(length(ix),min(2000,length(ix)));
    ix2 = ix(ix1);
    [Yb,Xb] = ind2sub(size(W),ix2); 


    
    fileName1 = sprintf('Grid1k%sn1_phase_0_output.mat',tmpl);
    fileName2 = sprintf('Grid1k%sn1_phase_4_output.mat',tmpl);
    fileName3 = sprintf('Grid1k%sn2_phase_0_output.mat',tmpl);
    fileName4 = sprintf('Grid1k%sn2_phase_4_output.mat',tmpl);
    
    outfileName = sprintf('Uest1k%sall.mat',tmpl);
   
    
    file1 = fullfile(ptout,fileName1);
    file2 = fullfile(ptout,fileName2);
    file3 = fullfile(ptout,fileName3);
    file4 = fullfile(ptout,fileName4);
    outfile = fullfile(ptout,outfileName);
    
  
    f1 = load(file1);
    [u1,ux1,uy1,uxx1,uyy1] = GetDers(f1,W,Omega,CorrectB,CorrectS,Orig);
    
    f2 = load(file2);
    [u2,ux2,uy2,uxx2,uyy2] = GetDers(f2,W,Omega,CorrectB,CorrectS,Orig);

    f3 = load(file3);
    [u3,ux3,uy3,uxx3,uyy3] = GetDers(f3,W,Omega,CorrectB,CorrectS,Orig);

    f4 = load(file4);
    [u4,ux4,uy4,uxx4,uyy4] = GetDers(f4,W,Omega,CorrectB,CorrectS,Orig);
    

    uxList{1} = ux1;
    uxList{2} = ux2;
    uxList{3} = ux3;
    uxList{4} = ux4;


    uyList{1} = uy1;
    uyList{2} = uy2;
    uyList{3} = uy3;
    uyList{4} = uy4;


    uxxList{1} = uxx1;
    uxxList{2} = uxx2;
    uxxList{3} = uxx3;
    uxxList{4} = uxx4;


    uyyList{1} = uyy1;
    uyyList{2} = uyy2;
    uyyList{3} = uyy3;
    uyyList{4} = uyy4;

    
    figure(1);
    title('u');
    subplot(221); imagesc(u1); axis equal; axis off; title('1');
    subplot(222); imagesc(u2); axis equal; axis off; title('2');
    subplot(223); imagesc(u3); axis equal; axis off; title('3');
    subplot(224); imagesc(u4); axis equal; axis off; title('4');
    
    
    figure(2);
    title('ux');
    subplot(221); imagesc(uxList{1}); axis equal; axis off; title('ux1');
    subplot(222); imagesc(uxList{2}); axis equal; axis off;title('ux2');
    subplot(223); imagesc(uxList{3}); axis equal; axis off;title('ux3');
    subplot(224); imagesc(uxList{4}); axis equal; axis off;title('ux4');


    figure(3);
    title('uxx');
    subplot(221); imagesc(uxxList{1}); axis equal; axis off;title('uxx1');
    subplot(222); imagesc(uxxList{2}); axis equal; axis off;title('uxx2');
    subplot(223); imagesc(uxxList{3}); axis equal; axis off;title('uxx3');
    subplot(224); imagesc(uxxList{4}); axis equal; axis off;title('uxx4');

    fprintf('saving to %s\n',outfile);    
    save(outfile,'uxList','uyList','uxxList','uyyList','Xb','Yb','W','Omega','XOmeg','YOmeg','h','Sigma','CorrectS','CorrectB');
end

function [u,ux,uy,uxx,uyy] = GetDers(f,W,Omega,CorrectB,CorrectS,Orig)


    if Orig,
        u = f.U1;
        [ux,uy] = BoundaryGradient_ver1(f.U1,W,Omega,CorrectB,CorrectS);
        [uxx,~] = BoundaryGradient_ver1(ux,W,Omega,CorrectB,CorrectS);
        [~,uyy] = BoundaryGradient_ver1(uy,W,Omega,CorrectB,CorrectS);
    else
        u = f.UEST;
        ux = f.UESTx;
        uy = f.UESTy;
        uxx = f.UESTxx;
        uyy = f.UESTyy;
        
    end

end

function [u,ux,uy,uxx,uyy] = GetDersNoise(f,W,Omega,CorrectB,CorrectS)

    sigma = 1e-5;
   
    u = f.UEST;
    u_noise = u + sigma*randn(size(u));
    
    [ux,uy] = BoundaryGradient_ver1(u_noise,W,Omega,CorrectB,CorrectS);
    [uxx,~] = BoundaryGradient_ver1(ux,W,Omega,CorrectB,CorrectS);
    [~,uyy] = BoundaryGradient_ver1(uy,W,Omega,CorrectB,CorrectS);
    

end

