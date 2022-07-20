function M = MaskSTARE(RGB_R)

choose = RGB_R;
[A, B] = imhist(choose);
low = min(A(30:80)); idx = find(A==low);
i = 1; D = idx(i);
while (D<30)
    i = i+1; D = idx(i);
end
C = roicolor(choose,D,255);
C = imfill(C,'holes');
for i=1:3
    C = masked(C);
end
% figure(1), imshowpair(choose,C,'montage')
% figure(2), imshow(V.*double(C))

M = C;
end
