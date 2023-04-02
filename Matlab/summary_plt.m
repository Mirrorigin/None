%% 汇总
path = 'E:\学习云盘\毕业论文\程序结果\matlab_Fig\';

wordsize = 16;

hf1 = openfig([path, 'dim_acc_WN18RR.fig'],'reuse');
fig1 = gca;

hf2 = openfig([path, 'dim_acc_FB15k.fig'],'reuse');
fig2 = gca;

figure

s1 = subplot(1,2,1);
xlabel('\fontname{宋体}\fontsize{14}维数');
ylabel('\fontname{宋体}\fontsize{14}推理准确度'); 
% xlim([0 500])
% ylim([0 0.9])
set(gca,'FontSize',wordsize);
set(gca,'Fontname','times new Roman');


s2 = subplot(1,2,2);
xlabel('\fontname{宋体}\fontsize{14}维数');
ylabel('\fontname{宋体}\fontsize{14}推理准确度'); 
%xlim([0 500])
% ylim([0 0.9])
% legend('FML-EG: {\itE} = 2', 'FML-EG: {\itE} = 4', 'FML-EG: {\itE} = 6','FML-EG: {\itE} = 8','FontSize',wordsize)
set(gca,'FontSize',wordsize);
set(gca,'Fontname','times new Roman');

ax1 = get(fig1, 'children');
ax2 = get(fig2, 'children');

copyobj(ax1, s1);
copyobj(ax2, s2);