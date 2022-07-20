function [Segmen] = DivideSegmen2(divSkel, BW)

% divSkel = S; BW = BW_final_2;

n_seg = max(divSkel(:,3)); seg = divSkel(:,3); Segmen = [];
segmen_x = divSkel(:,2); segmen_y = divSkel(:,1); tempBW = BW;
for i=1:n_seg
    idx = find(seg==i);
    x = segmen_x(idx);
    y = segmen_y(idx);
    if length(idx)>30
        tempSeg = divSkel(idx(1),:); j = 1; tempBW(tempSeg(j,1),tempSeg(j,2)) = 0;
        while j<=size(tempSeg,1) && sum(tempBW(:))>0
            if tempBW(tempSeg(j,1)+1,tempSeg(j,2))==1
                tempSeg = [tempSeg; tempSeg(j,1)+1 tempSeg(j,2) i];
                tempBW(tempSeg(j,1)+1,tempSeg(j,2)) = 0;
            end
            if tempBW(tempSeg(j,1),tempSeg(j,2)+1)==1
                tempSeg = [tempSeg; tempSeg(j,1) tempSeg(j,2)+1 i];
                tempBW(tempSeg(j,1),tempSeg(j,2)+1) = 0;
            end
            if tempBW(tempSeg(j,1)-1,tempSeg(j,2))==1
                tempSeg = [tempSeg; tempSeg(j,1)-1 tempSeg(j,2) i];
                tempBW(tempSeg(j,1)-1,tempSeg(j,2)) = 0;
            end
            if tempBW(tempSeg(j,1),tempSeg(j,2)-1)==1
                tempSeg = [tempSeg; tempSeg(j,1) tempSeg(j,2)-1 i];
                tempBW(tempSeg(j,1),tempSeg(j,2)-1) = 0;
            end
            j=j+1;
        end
        Segmen = [Segmen;tempSeg];
    end
end

end
