function [center,h,radius,U,G,dudn,sdudn,W,Sigma,n_hat_x,n_hat_y,phi,XOmeg,YOmeg, Unan, Omega,XOmegSmall,YOmegSmall,XOmegWide,YOmegWide,Sigmax,Sigmay] = ...
    ExtractImages(p,e,t,u,g,N,marg,Globalsx,Globalsy)

   
    center = (N+1)/2;
    h = 2.0/(N-1-2*marg);
    radius = (N-1-2*marg)/2; 
    %phiThresh=2.2;
    phiThresh = 0.1;
    %phiThresh = 0.005;
    
    fprintf('center=%.f h=%f radius=%.f\n',center,h,radius);

    Nedges = size(e,2);
    Npoints = size(p,2);
   
    W = zeros(N);
    
    
    
    x = linspace(-1-h*marg,1+h*marg,N);
    y = linspace(-1-h*marg,1+h*marg,N);
     
    U = tri2grid(p,t,u,x,y);
    %U(isnan(U)) = 0;
    
    
    
    gg = zeros(Npoints,1);
    for ii=1:Nedges
         pt_idx = e(1,ii);
         gg(pt_idx) = g(ii);
     end
     G = tri2grid(p,t,gg,x,y);
     G(isnan(G)) = 0;
    
     W(G~=0) = 1;  
    
   
    
    
   s = zeros(Npoints,1);
    for iii=1:size(p,2)
        xx = p(1,iii); yy=p(2,iii);
        s(iii)=heartNlungs(xx+1i*yy);
    end
    %Sigma = tri2grid(p,t,s,x,y);
    
    Sigma = zeros(N,N);
    Sigmax = zeros(N,N);
    Sigmay = zeros(N,N);
    
    x = linspace(-1-h*marg,1+h*marg,N);
    y = linspace(-1-h*marg,1+h*marg,N);
    for ii=1:length(x)
        for jj=1:length(y)
            xx = x(ii); yy=y(jj);
            [i,j] = XToGrid(xx,yy,N,marg);

            
            Sigma(i,j) = MySigma(xx+1i*yy);
            [itmp,jtmp] = XToGrid(xx,yy,size(Globalsx,1),0);
            Sigmax(i,j) = Globalsx(itmp,jtmp);
            Sigmay(i,j) = Globalsy(itmp,jtmp);
            
        end
    end


    phi=zeros(N);
    for x=1:N
        for y=1:N
            phi(y,x) = radius-sqrt( (x-center).^2 + (y-center).^2 );
        end
    end
    
    psi = zeros(N);
    for x=1:N
        for y=1:N
            psi(y,x) = radius-( (x-center).^2 + (y-center).^2 );
        end
    end
    
    
    
    
    W(phi > -1*phiThresh & phi < phiThresh) = 1;
    
    [phi_x,phi_y] = gradient(phi);
    magphi = sqrt(phi_x.^2+phi_y.^2);
    n_hat_x = -phi_x./(magphi+1e-20); n_hat_y = -phi_y./(magphi+1e-20);
    
    
    IX = find(phi >=-phiThresh);
    [YOmeg,XOmeg] = ind2sub(size(U),IX);
     
    [ux,uy] = gradient(U,h);
    dudn = ux.*n_hat_x +uy.*n_hat_y;
    
    Unan = U;
    U(isnan(U)) = 0;
    dudn(isnan(dudn))=0;
    Sigma(isnan(Sigma)) = 0;
    sdudn = Sigma.*dudn;
    
    Omega = zeros(size(U));
    for i=1:length(XOmeg)
        xx = XOmeg(i); yy = YOmeg(i);
        Omega(yy,xx)=1;
    end
    
    
    d = bwdist(1-Omega);
    W = zeros(size(d));
    W(d>0 & d <3) = 1;
    
    
    
    r = sqrt((XOmeg-center).^2+(YOmeg-center).^2);
    ix = find(r < radius-10);
    XOmegSmall = XOmeg(ix); YOmegSmall = YOmeg(ix);
    
    Xall = zeros(N*N,1);
    Yall = zeros(N*N,1);
    idx=1;
    for i=1:N,
        for j=1:N
            Xall(idx) = i; Yall(idx)=j;
            idx = idx+1;
        end
    end
    r = sqrt((Xall-center).^2+(Yall-center).^2);
    ix = find(r < radius+marg);
    XOmegWide = Xall(ix); YOmegWide = Yall(ix);    
    
    %% Jaanuary 13
    W(U==0) = 0;
    Omega(U==0)=0;
    %%
    
    
    %%%%%%%%
%     dx = 0.001;
%     for xx=-1:dx:1
%         for yy=-1:dx:1
%             r = sqrt(xx.^2+yy.^2);
%             if r >= 1, continue; end;
%             %v = cos(xx*2*pi/1.5).*cos(yy*2*pi/1.5);
%             v = sin(xx*2*pi);
%             [iR,jC] = XToGrid(xx,yy,N,marg);
%             U(iR,jC) = v;
%         end
%     end
    
    %%%%%%%
    
    
end

function [iR,jC] = XToGrid(x,y,N,marg)

%     center = round((N+1)/2);
%     h = 2/(N-1);
% 
%     iR = (y/h+center); iR = round(iR); if iR<=0, iR=1; end; if iR > N, iR=N; end;
%     jC = (x/h+center); jC = round(jC); if jC<=0, jC=1; end; if jC > N, jC=N; end;

    
    center = (N+1)/2;
    h = 2/(N-1-2*marg);
    iR = (y/h+center); iR = round(iR); if iR<=0, iR=1; end; if iR > N, iR=N; end;
    jC = (x/h+center); jC = round(jC); if jC<=0, jC=1; end; if jC > N, jC=N; end;
    
    
    %fprintf('(%.2f,%.2f)-->(%.2f,%.2f)\n',x,y,jC,iR);

end