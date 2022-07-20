function sum_neigh = Calculate_Neighbour(citra, i, j)
sum = 0;
for v = i-1:i+1
    for w = j-1:j+1
        sum = sum + citra(v,w);
    end
end
sum_neigh = sum;
end

