J_value = [];
J_max = [];
for h=1:241
    fprintf('%i\n', h)
    J_value = [mean_tpr2, mean_fpr2, mean_tpr2 + (1 - mean_fpr2)];
end
[val_J idx_J] = max(J_value(:,3));
J_max = [J_value(idx_J,:)];
% features_log = log10(features);
% % features_log = features;
% n = [1 2 4 8 16 32 64 128 256 512 1024 2048];
% log_n = log10(n);
% err = [];
% for h=1:1200
%     fprintf('%i\n', h)
%     Y = features_log(h,1:11);
%     X = log_n(1:11);
%     B = polyfit(X,Y,1);
%     for i=1:11
%         err(i) = features_log(i) - (B(1)*log_n(i)+B(2));
%     end
%     c(h) = std(err);
%     fcap(h) = abs(B(1));
%     fcap_err(h) = c(h)*sqrt(10/9);
% end
% for h=1:1200
%     fprintf('%i\n', h)
%     if features(h,11) ~= 0
%         Y = features_log(h,1:11);
%         X = log_n(1:11);
%         B = polyfit(X,Y,1);
%         for i=1:11
%             err(i) = features_log(i) - (B(1)*log_n(i)+B(2));
%         end
%         c(h) = std(err);
%         fcap(h) = abs(B(1));
%         fcap_err(h) = c(h)*sqrt(10/9);
%     else
%         Y = features_log(h,1:10);
%         X = log_n(1:10);
%         B = polyfit(X,Y,1);        
%         for i=1:10
%             err(i) = features_log(i) - (B(1)*log_n(i)+B(2));
%         end
%         c(h) = std(err);
%         fcap(h) = abs(B(1));
%         fcap_err(h) = c(h)*sqrt(9/8);
%     end
% end