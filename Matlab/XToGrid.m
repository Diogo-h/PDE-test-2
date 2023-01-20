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