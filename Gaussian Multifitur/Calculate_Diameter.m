function [grad D] = Calculate_Diameter(BW, i, j)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
N = 16;
Diameter = zeros(N,2);

k=1;
for x=0:N/4
    y = N/4;
    Diameter(k,1) = y/x;
    Diameter (k,2) = Cal_D(BW, i, j, Diameter (k,1), 'Y', 1);
    k = k+1;
end
for x=N/4-1:-1:-N/4+1
    y = N/4;
    Diameter(k,1) = x/y;
    Diameter (k,2) = Cal_D(BW, i, j, Diameter (k,1), 'X', 1);
    k = k+1;
end
for x=N/4:-1:1
    y = -N/4;
    Diameter(k,1) = y/x;
    Diameter (k,2) = Cal_D(BW, i, j, Diameter (k,1), 'Y', 1);
    k = k+1;
end

min = Inf; idx = 0;
for x=1:N
    if (Diameter (x,2) < min)
        min = Diameter (x,2);
        idx = x;
    end
end
grad = Diameter(idx,1);
D = min; 
end

