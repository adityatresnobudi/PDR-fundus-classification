function [Segmen] = DivideSkel(BW)

[x, y] = find(BW);
initArray = [x, y]; conArray = []; 
segArray = []; seg = 1;
while isempty(initArray)==0
    conArray = [conArray; initArray(1,1) initArray(1,2) 0]; bool_con = 1;
    initArray(1,:) = [];
    while (bool_con)
        [bool_con, initArray, conArray] = FindConnec(initArray, conArray);
    end
    j = length(segArray); k = size(conArray,1); conArray(:,3) = ones(k,1);
    segArray(j+1:k,1) = seg*ones(k-j,1); seg = seg+1;
%     Segmen = [conArray(:,1) conArray(:,2) segArray];
end
Segmen = [conArray(:,1) conArray(:,2) segArray];

end

function [bool_con, initArray, conArray] = FindConnec(init, con)

initArray = init; conArray = con;
idx = find(conArray(:,3)==0); conArray(idx(1),3) = 1; 
i = 1;
while i<=size(initArray,1)
    r = sqrt( (initArray(i,1)-conArray(idx(1),1))^2 + (initArray(i,2)-conArray(idx(1),2))^2 );
    if (r==1)||(r==sqrt(2))
        conArray = [conArray; initArray(i,1) initArray(i,2) 0];
        initArray(i,:) = []; 
    else
        i = i+1;
    end
end

idx = find(conArray(:,3)==0);
if (isempty(idx) || isempty(initArray))
    bool_con = 0;
else
    bool_con = 1;
end

end