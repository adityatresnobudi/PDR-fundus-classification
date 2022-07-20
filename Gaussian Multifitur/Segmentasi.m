%% PEMILIHAN FILE CITRA
clear; clc;
%%
% Membaca/input citra mentah dan vessel segmentasi (pilih file)
[fnameRAW, pnameRAW] = uigetfile('*.tif','Please select RAW IMAGE(s)', 'MultiSelect', 'on');
fullnameRAW = strcat(pnameRAW,fnameRAW);
n_RAW = length(fullnameRAW);
%%
[fnameVES, pnameVES] = uigetfile({'*.png'},'Please select VESSEL IMAGE(s)', 'MultiSelect', 'on');
fullnameVES = strcat(pnameVES,fnameVES);
%% 
[fnameMASK, pnameMASK] = uigetfile({'*.gif'},'Please select MASK IMAGE(s)', 'MultiSelect', 'on');
fullnameMASK = strcat(pnameMASK,fnameMASK);
%% 
[fnameLES, pnameLES] = uigetfile({'*.gif'},'Please select RED LESION IMAGE(s)', 'MultiSelect', 'on');
fullnameLES = strcat(pnameLES,fnameLES);
%% MAT FILE
[fnameSEG, pnameSEG] = uigetfile('*.mat','Please select MAT file(s)', 'MultiSelect', 'on');
fullnameSEG = strcat(pnameSEG,fnameSEG);
%% TIFF FILE
[fnameSEG, pnameSEG] = uigetfile('*.tiff','Please select VESSEL SEGMENTATION file(s)', 'MultiSelect', 'on');
fullnameSEG = strcat(pnameSEG,fnameSEG);
%% RESOULTION CHECK
raw_size_data = [];
hitung = 0;
% les_size_data = [];
% ves_size_data = [];
for h=1:2
    fprintf('%i\n', h)
    image = imread(cell2mat(fullnameRAW(h)));
%     lesion = imread(cell2mat(fullnameLES(h)));
%     ves = imread(cell2mat(fullnameVES(h)));
%     [row_v,col_v,lay_v] = size(ves);
    [row,col,lay] = size(image);
%     if col_v > 1700
%         hitung = hitung + 1;
%     end
%     [row_l,col_l,lay_l] = size(lesion);
%     ves = imresize(ves, [row,col]);
    
    raw_size_data = [raw_size_data; fnameRAW(h), row, col, lay];
%     les_size_data = [les_size_data; fnameLES(h), row_l, col_l, lay_l];
%     ves_size_data = [ves_size_data; fnameVES(h), row_v, col_v, lay_v];
end
%% CLAHE dan LESI MERAH
tabel_data = [];
for h=1:120
    tic
    fprintf('%i\n', h)
    % Membaca/input citra mentah dan ekstraksi kanal RGB
    rgbImage = imread(cell2mat(fullnameRAW(h)));
    LAB = rgb2lab(rgbImage); L = LAB(:,:,1)/100; L = adapthisteq(L,'ClipLimit',0.01); 
    LAB(:,:,1) = L*100; ClaheRGB = lab2rgb(LAB);
%     figure(h);
%     subplot(2,2,1), imshow(rgbImage);
%     subplot(2,2,2), imhist(rgbImage);
%     subplot(2,2,3), imshow(ClaheRGB);
%     subplot(2,2,4), imhist(ClaheRGB);
%     fig = figure(h);
%     fig.WindowState = 'maximized';
    [counts_I,binLocations_I] = imhist(rgbImage);
%     [counts_CLAHE,binLocations_CLAHE] = imhist(ClaheRGB);
    sum_counts_I = 0;
    for i=128:255
        sum_counts_I = sum_counts_I + counts_I(i);
    end
    tabel_data = [tabel_data; fnameRAW(h), sum_counts_I];
%     figure (h), imshowpair(rgbImage,L,'montage'), hold on;
    RGB_R = rgbImage(:,:,1);
    RGB_G = rgbImage(:,:,2); 
    RGB_B = rgbImage(:,:,3);
    [row col] = size(RGB_G);
    [counts,binLocations] = imhist(RGB_G);
    B = 125/find(max(counts(50:end))==counts); 
    if B(1) == 0
        RGB_G_Norm = RGB_G*B(2);
    else
        RGB_G_Norm = RGB_G*B(1);
    end
    
    vessImage = imread(cell2mat(fullnameVES(h)));
    [row,col,lay] = size(rgbImage);
    vessImage = imresize(vessImage, [row,col]);
    vessImage = im2gray(vessImage);
    T_vI = graythresh(vessImage);
    vI = imbinarize(vessImage, T_vI);
%     vI = 1-vI;

    Cliplimit = 0.01;
    Clahe = adapthisteq(RGB_G_Norm,'ClipLimit',Cliplimit); % CLAHE
    num_iter = 5;   % number of iterations
    delta_t = 0.14; % integration constant (0 <= delta_t <= 1/7)
    kappa = 3;      % gradient modulus threshold that controls the conduction
    option = 2;     % conduction coefficient functions proposed by Perona & Malik:
                    % 1 - c(x,y,t) = exp(-(nablaI/kappa).^2),
                    %     privileges high-contrast edges over low-contrast ones. 
                    % 2 - c(x,y,t) = 1./(1 + (nablaI/kappa).^2),
                    %     privileges wide regions over smaller ones.
    ad = anisodiff2D(Clahe,num_iter,delta_t,kappa,option); % Filter Difusi Anisotropik
    sigmas = [0.1 1.1 2.1 3.1 4.1 5.1 5.5];
    if col > 1000
        sigmas = 3.5:1:5.5;   % vector of scales on which the vesselness is computed
        tau = 0.9;
        if sum_counts_I >= 600000
            tau = 0.95;
        end
    end
    spacing = [1;1];        % input image spacing resolution
    tau = 0.8;              % (between 0.5 and 1) lower tau -> more intense output response
    if sum_counts_I >= 600000
        tau = 0.85;
    end
    V = vesselness2D(ad, sigmas, spacing, tau, false); % Hessian-based
    pinggiran = imread(cell2mat(fullnameMASK(h)));
    pinggiranL = logical(pinggiran);
    V2 = pinggiranL.*V; V2 = V2/max(V2(:));
%     T = 0.6;
    BW = imbinarize(V2); % Thresholding
%     se = strel('disk',15); tophat = imtophat((255-uint8(ad)),se); 
%     tophat = double(tophat)/double(max(tophat(:))); BW2 = imbinarize(tophat);
%     figure(h), imshowpair(ClaheRGB,BW,'montage')
%     idx = find(BW2==1); A = length(find(vI(idx)==1)); B = A*100/length(find(vI==1))
    
%     se = strel('disk',12);
%     close_G = imclose(RGB_G_Norm,se); 
%     exud = vesselness2D(close_G, [0.5:1:7.5], spacing, 0.65, false); % Hessian-based
%     exud = pinggiranL.*exud; exud = exud/max(exud(:));
% %     T = 0.5;
%     BW_exud = imbinarize(exud); % Thresholding
%     %figure (h), imshowpair(exud,V2,'montage'), hold on;
    
    lesi = imread(cell2mat(fullnameLES(h))); % imshow(lesi)
    [i, j] = find(BW>0); 
%     tabel = zeros(length(i),15);
    tabel = zeros(length(i),12);
    for x=1:length(i)
        if i(x) >= 3 & j(x) >= 3
            if i(x) > max(i)-2 & j(x) > max(j)-2
%                 window = double(exud(i(x)-2:max(i),j(x)-2:max(j)));
                window2 = double(V2(i(x)-2:max(i),j(x)-2:max(j)));
                window3 = double(lesi(i(x)-2:max(i),j(x)-2:max(j)));
                window4 = double(RGB_G_Norm(i(x)-2:max(i),j(x)-2:max(j)));
            elseif i(x) <= max(i)-2 & j(x) > max(j)-2
%                 window = double(exud(i(x)-2:i(x)+2,j(x)-2:max(j)));
                window2 = double(V2(i(x)-2:i(x)+2,j(x)-2:max(j)));
                window3 = double(lesi(i(x)-2:i(x)+2,j(x)-2:max(j)));
                window4 = double(RGB_G_Norm(i(x)-2:i(x)+2,j(x)-2:max(j)));
            elseif i(x) > max(i)-2 & j(x) <= max(j)-2
%                 window = double(exud(i(x)-2:max(i),j(x)-2:j(x)+2));
                window2 = double(V2(i(x)-2:max(i),j(x)-2:j(x)+2));
                window3 = double(lesi(i(x)-2:max(i),j(x)-2:j(x)+2));
                window4 = double(RGB_G_Norm(i(x)-2:max(i),j(x)-2:j(x)+2));
            else
%                 window = double(exud(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
                window2 = double(V2(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
                window3 = double(lesi(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
                window4 = double(RGB_G_Norm(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
            end
        elseif i(x) >= 3 & j(x) < 3
            if i(x) > max(i)-2
%                 window = double(exud(i(x)-2:max(i),1:j(x)+2));
                window2 = double(V2(i(x)-2:max(i),1:j(x)+2));
                window3 = double(lesi(i(x)-2:max(i),1:j(x)+2));
                window4 = double(RGB_G_Norm(i(x)-2:max(i),1:j(x)+2));
            else
%                 window = double(exud(i(x)-2:i(x)+2,1:j(x)+2));
                window2 = double(V2(i(x)-2:i(x)+2,1:j(x)+2));
                window3 = double(lesi(i(x)-2:i(x)+2,1:j(x)+2));
                window4 = double(RGB_G_Norm(i(x)-2:i(x)+2,1:j(x)+2));
            end
        elseif i(x) < 3 & j(x) >=3
            if j(x) > max(j)-2
%                 window = double(exud(1:i(x)+2,j(x)-2:max(j)));
                window2 = double(V2(1:i(x)+2,j(x)-2:max(j)));
                window3 = double(lesi(1:i(x)+2,j(x)-2:max(j)));
                window4 = double(RGB_G_Norm(1:i(x)+2,j(x)-2:max(j)));
            else
%                 window = double(exud(1:i(x)+2,j(x)-2:j(x)+2));
                window2 = double(V2(1:i(x)+2,j(x)-2:j(x)+2));
                window3 = double(lesi(1:i(x)+2,j(x)-2:j(x)+2));
                window4 = double(RGB_G_Norm(1:i(x)+2,j(x)-2:j(x)+2));
            end
        else
%             window = double(exud(1:i(x)+2,1:j(x)+2));
            window2 = double(V2(1:i(x)+2,1:j(x)+2));
            window3 = double(lesi(1:i(x)+2,1:j(x)+2));
            window4 = double(RGB_G_Norm(1:i(x)+2,1:j(x)+2));
        end
        tabel(x,1) = i(x);                      tabel(x,2) = j(x);
        tabel(x,3) = double(V2(i(x),j(x)));     tabel(x,4) = mean(window2(:));  tabel(x,5) = std(window2(:));
%         tabel(x,6) = double(exud(i(x),j(x)));   tabel(x,7) = mean(window(:));   tabel(x,8) = std(window(:));
%         tabel(x,9) = double(RGB_G_Norm(i(x),j(x)))/255;     tabel(x,10) = mean(window4(:))/255; 
%         tabel(x,11) = std(window4(:))/255;
        tabel(x,6) = double(RGB_G_Norm(i(x),j(x)))/255;     tabel(x,7) = mean(window4(:))/255; 
        tabel(x,8) = std(window4(:))/255;
%         tabel(x,12) = double(lesi(i(x),j(x)));   tabel(x,13) = mean(window3(:));   tabel(x,14) = std(window3(:));
%         tabel(x,15) = vI(i(x),j(x));
        tabel(x,9) = double(lesi(i(x),j(x)));   tabel(x,10) = mean(window3(:));   tabel(x,11) = std(window3(:));
        tabel(x,12) = vI(i(x),j(x));
    end
    
    temp_name = cell2mat(fnameRAW(h));
    temp_name = temp_name(1,1:length(temp_name)-4);
    fout1 = strcat('KoreksiEXU_', temp_name, '.mat');
    save(fout1,'tabel');
    fout2 = strcat('Hessian_', temp_name, '.tiff');
    imwrite(BW, fout2, 'tiff');
%     fout3 = strcat('CLAHE_', temp_name, '.tiff');
%     imwrite(ClaheRGB, fout3, 'tiff');
%     fout4 = strcat('HistCLAHE_', temp_name, '.jpg');
%     saveas(fig, fout4);
    
%     A = edge(BW); [x y] = find(A==1);
%     figure(h), imshow(rgbImage);
%     figure (h), imshowpair(ClaheRGB,Clahe,'montage');
%     figure (h+3), imshowpair(BW_exud,BW,'montage');
%     figure(h), imshow(ClaheRGB), hold on, scatter(y,x,3,'filled','s','green'), hold off
    toc
end
%% SEGMENTASI 
% fitur = [3:14];
fitur = [3:11];
load('ModelkNN5_8.mat');
for h=1:120
    tic
    % Membaca/input citra mentah dan ekstraksi kanal RGB
%     rgbImage = imread(cell2mat(fullnameRAW(h))); 
%     LAB = rgb2lab(rgbImage); L = LAB(:,:,1)/100; L = adapthisteq(L,'ClipLimit',0.005); 
%     LAB(:,:,1) = L*100; ClaheRGB = lab2rgb(LAB); 
%     RGB_R = rgbImage(:,:,1);
%     RGB_G = rgbImage(:,:,2); 
%     RGB_B = rgbImage(:,:,3);
%     [row col] = size(RGB_G);
%     [counts,binLocations] = imhist(RGB_G);
%     B = 125/find(max(counts(50:end))==counts); RGB_G_Norm = RGB_G*B(1); 
%     
    fprintf('%i\n', h)
    image = imread(cell2mat(fullnameRAW(h)));
    [row,col,lay] = size(image);
    load(cell2mat(fullnameSEG(h)));
    X = tabel(:,fitur);
    [Y, Y_score] = predict(Mdl,X);
    BW2 = zeros(row,col); 
%     BW = BW2;
    for i=1:size(tabel,1)
        BW2(tabel(i,1),tabel(i,2)) = Y(i);
%         BW(tabel(i,1),tabel(i,2)) = 1;
    end
    Morph = bwmorph(BW2,'clean',Inf);
    Morph = bwmorph(Morph,'fill',Inf);
    Morph = bwmorph(Morph,'bridge',Inf);
    Morph = bwmorph(Morph,'diag',Inf);
    Morph = bwmorph(Morph,'close',1);
    Morph = imresize(Morph, [row col]);
%     figure(h), imshowpair(BW1,Morph,'montage')
    
    temp_name = cell2mat(fnameSEG(h));
    temp_name = temp_name(1,12:length(temp_name)-4);
    fout = strcat('Ves5_', temp_name, '.tiff');
    imwrite(Morph, fout, 'tiff');
    
%     Morph = imread(cell2mat(fullnameSEG(h)));
%     vessImage = imread(cell2mat(fullnameVES(h)));
%     [row col] = size(vessImage);
%     Morph = imresize(Morph, [row,col]);
%     vessImage = im2gray(vessImage);
%     T_vI = graythresh(vessImage);
%     vI = imbinarize(vessImage, T_vI);
% %     vI = 1-vI;
% 
%     idx1 = find(Morph==1); 
%     TP = length(find(vI(idx1)==1)); 
%     FP = length(find(vI(idx1)==0));
%     idx2 = find(Morph==0);
%     TN = length(find(vI(idx2)==0)); 
%     FN = length(find(vI(idx2)==1));
%     Acc = (TP+TN)/(TP+TN+FP+FN)*100;
%     Sen = TP/(TP+FN)*100;
%     Spec = TN/(FP+TN)*100;
%     Prec = TP/(TP+FP)*100;
%     Data = [Data; fnameSEG(h), Acc, Sen, Spec, Prec];
% %     Data = [Data; fnameSEG(h), Acc, Sen, Spec];
%     fout = strcat('Performa Run Akhir 3', '.mat');
%     save(fout,'Data');
%     writecell(Data, 'Performa Run Akhir 3.xlsx');
    
%     A = edge(BW2); [x y] = find(A==1);
%     figure(h), imshow(ClaheRGB), hold on, scatter(y,x,3,'filled','s','green'), hold off
    toc
end
%% MEDIAN FILTERING
for h=1:120
    fprintf('%i\n', h)
    Morph = imread(cell2mat(fullnameSEG(h)));
    if col > 1900
        Morph_Medfilt = medfilt2(Morph, [10 10]);
    end
    if col > 1000 & col < 1900
        Morph_Medfilt = medfilt2(Morph, [6 6]);
    end
    Morph_Medfilt = medfilt2(Morph, [4 4]);
    temp_name = cell2mat(fnameRAW(h));
    temp_name = temp_name(1,1:length(temp_name)-4);
    fout3 = strcat('MedFilt_', temp_name, '.tiff');
    imwrite(Morph_Medfilt, fout3, 'tiff');
end
%% PERFORMA SEGMENTASI
Data = [];
for h=1:1200
    fprintf('%i\n', h)
    Morph = imread(cell2mat(fullnameSEG(h)));
    vessImage = imread(cell2mat(fullnameVES(h)));
    [row col] = size(vessImage);
    Morph = imresize(Morph, [row,col]);
    vessImage = im2gray(vessImage);
    T_vI = graythresh(vessImage);
    vI = imbinarize(vessImage, T_vI);
%     vI = 1-vI;

    idx1 = find(Morph==1); 
    TP = length(find(vI(idx1)==1)); 
    FP = length(find(vI(idx1)==0));
    idx2 = find(Morph==0);
    TN = length(find(vI(idx2)==0)); 
    FN = length(find(vI(idx2)==1));
%     Acc = (TP+TN)/(TP+TN+FP+FN)*100;
%     Sen = TP/(TP+FN)*100;
%     Spec = TN/(FP+TN)*100;
    Prec = TP/(TP+FP)*100;
    Rec = TP/(TP+FN)*100;
    F_Score = (2 * Prec * Rec) / (Prec + Rec);
    Data = [Data; fnameSEG(h), Prec, Rec, F_Score];
%     Data = [Data; fnameSEG(h), Acc, Sen, Spec];
    fout = strcat('Performa Taris', '.mat');
    save(fout,'Data');
    writecell(Data, 'Performa Taris.xlsx');
end
%% IMAGE OVERLAY
for h=1:1200
    fprintf('%i\n', h)
    image = imread(cell2mat(fullnameRAW(h)));
    Morph = imread(cell2mat(fullnameSEG(h)));
    vessImage = imread(cell2mat(fullnameVES(h)));
    [row col] = size(vessImage);
    Morph = imresize(Morph, [row,col]);
    image = imresize(image, [row,col]);
    vessImage = im2gray(vessImage);
    T_vI = graythresh(vessImage);
    vI = imbinarize(vessImage, T_vI);
    d = vI-Morph;
    trupos = zeros(row,col);
    falneg = d;
    falpos = d;
    for i=1:row
        for j=1:col
            if falneg(i,j) < 0
                falneg(i,j) = 0;
            end
        end
    end
    for i=1:row
        for j=1:col
            if falpos(i,j) > 0
                falpos(i,j) = 0;
            elseif falpos(i,j) < 0
                falpos(i,j) = 1;
            end
        end
    end
    for i=1:row
        for j=1:col
            if vI(i,j) == 1 & Morph(i,j) == 1
                trupos(i,j) = 1;
            end
        end
    end
    out_1 = imoverlay_old(image, falneg, [1 0 0]);
    out_2 = imoverlay_old(out_1, falpos, [0 1 0]);
    out_3 = imoverlay_old(out_2, trupos, [1, 0.7, 0.1]);
    temp_name = cell2mat(fnameRAW(h));
    temp_name = temp_name(1,1:length(temp_name)-4);
    fout4 = strcat('OverlayTaris_', temp_name, '.tiff');
    imwrite(out_3, fout4, 'tiff');
end