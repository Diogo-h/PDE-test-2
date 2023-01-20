function [] = DisplayResultsU()


    clear;
    close all;
    
    writeOutput = 0;
    currentFolder = pwd;
    in_pt = sprintf('%s/../Data/InputData',currentFolder);
    pt = sprintf('%s/../Data/OutputData',currentFolder);
    pt_write = sprintf('%s/../Data/OutputData/Figures',currentFolder);


    tmpl = 'L5H2';
    %tmpl = 'L1H02';
    n = 2;
    phs = 4;
    
    
    filename1 = sprintf('Grid1k%sn%d_phase_%d.mat',tmpl,n,phs);
    filename2 = sprintf('Grid1k%sn%d_phase_%d_output.mat',tmpl,n,phs);
    
        
    f1 = load(fullfile(in_pt,filename1));
    f2 = load(fullfile(pt,filename2));

    
    Omega = f1.Omega;
    W = f1.W;


    f1.ux = f1.ux.*(1-W);
    f2.UESTx = f2.UESTx.*(1-W);

    [~,s,ex] = fileparts(filename2);
    DrawUResults(f1,f2,pt_write,s,writeOutput,Omega)

end

function [] = DrawUResults(f1,f2,pt,fltmp,writeOutput,Omega)

    [mse,ssim,psnr] = CalcError(f1.U1,f2.UEST,Omega);
    tu = sprintf('U: mse=%.2e ssim=%.3f, psnr=%.2f\n',mse,ssim,psnr);

     [mse,ssim,psnr] = CalcError(f1.ux,f2.UESTx,Omega);
     tux = sprintf('Ux: mse=%.2e ssim=%.3f psnr=%.2f\n',mse,ssim,psnr);
    
     
    umin = min(min(f1.U1)); 
    umax = max(max(f1.U1)); 
    uxmin = min(f1.ux(:));
    uxmax = max(f1.ux(:));
    
    uerrmin=-0.1;
    uerrmax = 0.1;
    uxerrmin=-0.3;
    uxerrmax=0.3;
    
    figure;
    subplot(221); imagesc(f1.U1,[umin,umax]); axis equal; axis off;title('FE U');
    subplot(222); imagesc(f2.UEST,[umin,umax]); axis equal; axis off;title(tu);
    subplot(223); imagesc(f1.ux,[uxmin,uxmax]); axis equal; axis off;title('FE Ux');
    subplot(224); imagesc(f2.UESTx,[uxmin,uxmax]); axis equal; axis off;title(tux);
    
    
    if writeOutput,
        
        mx1 = max(f1.U1(:));
        mx2 = max(f1.ux(:));
        
        nm = sprintf('Ugt%s.png',fltmp);
        WriteImage(f1.U1, pt, nm,umin,umax);
        nm = sprintf('U%s.png',fltmp);
        WriteImage(f2.UEST, pt, nm,umin,umax);
        nm = sprintf('Uerr%s.png',fltmp); 
        WriteImage((f1.U1-f2.UEST)./mx1, pt, nm,uerrmin,uerrmax);
        %WriteImage((f1.U1-f2.UEST)./mx1, pt, nm);
        nm = sprintf('Ux_gt%s.png',fltmp);
        WriteImage(f1.ux, pt, nm,uxmin,uxmax);
        nm = sprintf('Ux%s.png',fltmp);
        WriteImage(f2.UESTx, pt, nm,uxmin,uxmax);
        nm = sprintf('Uxerr%s.png',fltmp); 
        %WriteImage((f1.ux-f2.UESTx)./mx2, pt, nm,uxerrmin,uxerrmax);
        WriteImage((f1.ux-f2.UESTx), pt, nm);
        
    end
    
end




function WriteImage(A, pt, nm,umin,umax)

    if ~exist('umin','var'),
        umin=min(A(:));
        umax=max(A(:));
    end
    fileName = fullfile(pt,nm);
    fprintf('Writing %s\n',fileName);
   
    figure; imagesc(A,[umin,umax]); axis equal;  axis off; 
    
  
    colorbar1 = colorbar(...
       'FontSize',20);
   
   
    saveas(gcf,fileName);

    
    
end
