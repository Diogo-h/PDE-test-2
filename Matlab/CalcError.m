function [mse,ssim_val,psnr] = CalcError(f,fest,Omega)
    
    mx = max(f(:));
    if mx == 0,
        mx = max(abs(f(:)));
    end
    e = (f-fest).*Omega;
    mse = mean(e(:).^2);
    rmse = mean((e(:)/mx).^2);
    psnr = 10*log10((mx^2)/mse);

    ssim_val = ssim(double(fest.*Omega),double(f.*Omega));
    %rmse = corr2(f,fest);
    
end