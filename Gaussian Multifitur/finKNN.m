
load('KoreksiEXU_20051019_38557_0100_PP');
X =  tabel(:, 3:11);
Y =  tabel(:, 12);

Mdl = fitcknn(X,Y,'NumNeighbors',5);  %fitcsvm(X,Y);
fout = strcat('ModelkNN5_8.mat');
save(fout,'Mdl');

