function D = Cal_D(BW, i, j, grad, cond, inc)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
tepi = 0; x1 = i; y1 = j;
while not(tepi)
    if (cond == 'X')
        x1 = x1+inc;
        if (grad ~= Inf)
            y1 = round(grad*(x1-i)+j);
        end
    elseif (cond == 'Y')
        y1 = y1+inc;
        if (grad ~= 0)
            x1 = round((y1-j)/grad+i);
        end
    end
    if (BW(x1,y1)==0)
        tepi = 1;
        if (cond == 'X')
            x1 = x1-inc;
            if (grad ~= Inf)
                y1 = round(grad*(x1-i)+j);
            end
        elseif (cond == 'Y')
            y1 = y1-inc;
            if (grad ~= 0)
                x1 = round((y1-j)/grad+i);
            end
        end
    end
end

tepi = 0; x2 = i; y2 = j;
while not(tepi)
    if (cond == 'X')
        x2 = x2-inc;
        if (grad ~= Inf)
            y2 = round(grad*(x2-i)+j);
        end
    elseif (cond == 'Y')
        y2 = y2-inc;
        if (grad ~= 0)
            x2 = round((y2-j)/grad+i);
        end
    end
    if (BW(x2,y2)==0)
        tepi = 1;
    end
end

D = sqrt((x1-x2)^2 + (y1-y2)^2);
end

