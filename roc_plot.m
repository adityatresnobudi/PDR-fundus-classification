% % line_color = [255, 152, 52] / 255;
% line_color = [112, 190, 110] / 255;
% % line_color = [0, 0, 255] / 255;
% line_style = '--';
% line_width = 10;
% current_legend = [current_legend, ' - AUC=', num2str(ans,'%.4f'), ' $\pm$ ', num2str(std_auc,'%.2f')];
% my_legends = cat(1, my_legends, current_legend);
%     
% plot(mean_fpr_2, mean_tpr_2, 'LineWidth', line_width, 'Color', line_color, 'LineStyle', line_style);
% box on; grid on;
% xlim([0 1]); ylim([0 1]);
% legend(my_legends, 'Location', 'southeast','Interpreter','LaTex');
% xlabel('FPR = 1 - Specificity', 'Interpreter', 'LaTex')
% ylabel('TPR = Sensitivity', 'Interpreter', 'LaTex')
% hold on