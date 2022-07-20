function [found X Y] = Calculate_Segment(citra, i, j)
if (citra(i-1,j-1)==1)
    found = 1; 
    X = i-1; 
    Y = j-1;
elseif (citra(i-1,j)==1)
    found = 1; 
    X = i-1; 
    Y = j;
elseif (citra(i-1,j+1)==1)
    found = 1; 
    X = i-1; 
    Y = j+1;
elseif (citra(i,j-1)==1)
    found = 1; 
    X = i; 
    Y = j-1;
elseif (citra(i,j+1)==1)
    found = 1; 
    X = i; 
    Y = j+1;
elseif (citra(i+1,j-1)==1)
    found = 1; 
    X = i+1; 
    Y = j-1;
elseif (citra(i+1,j)==1)
    found = 1; 
    X = i+1; 
    Y = j;
elseif (citra(i+1,j+1)==1)
    found = 1; 
    X = i+1; 
    Y = j+1;
else
    found = 0;
    X = i;
    Y = j;
end
end

