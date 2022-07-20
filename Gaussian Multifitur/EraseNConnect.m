function [BW_erased] = EraseNConnect(BW, N_con)

[x, y] = find(BW); BW_erased = BW;
for k=1:length(x)
    con = [x(k), y(k), 0]; init = [x, y]; init(k,:) = [];
    bool_con = FindConnec(init, con, N_con);
    if bool_con==0
        BW_erased(x(k),y(k)) = 0;
    end
end

end

function [bool_con] = FindConnec(init, con, N_con)

initArray = init; conArray = con;
idx = find(conArray(:,3)==0); n = 1; 

while (n<=N_con)&&(length(idx)>0)
    conArray(idx(1),3) = 1; i = conArray(idx(1),1); j = conArray(idx(1),2);

    i_con = find(initArray(:,1)==i);
    if (length(i_con)>0)
        j_con = find(initArray(i_con,2)==j+1);
        if (length(j_con)>0)
            conArray = [conArray; i j+1 0]; n = n+1;
            initArray(i_con(j_con(1)),:) = []; 
        end
    end

    i_con = find(initArray(:,1)==i);
    if (length(i_con)>0)
        j_con = find(initArray(i_con,2)==j-1);
        if (length(j_con)>0)
            conArray = [conArray; i j-1 0]; n = n+1;
            initArray(i_con(j_con(1)),:) = []; 
        end
    end

    j_con = find(initArray(:,2)==j);
    if (length(j_con)>0)
        i_con = find(initArray(j_con,1)==i+1);
        if (length(i_con)>0)
            conArray = [conArray; i+1 j 0]; n = n+1;
            initArray(j_con(i_con(1)),:) = []; 
        end
    end

    j_con = find(initArray(:,2)==j);
    if (length(j_con)>0)
        i_con = find(initArray(j_con,1)==i-1);
        if (length(i_con)>0)
            conArray = [conArray; i-1 j 0]; n = n+1;
            initArray(j_con(i_con(1)),:) = []; 
        end
    end
    idx = find(conArray(:,3)==0);
end

if (n>N_con)
    bool_con = 1;
else 
    bool_con = 0;
end

end