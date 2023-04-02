path = 'E:\学习云盘\毕业论文\程序结果\losslist\';
% path = 'E:\学习云盘\毕业论文\程序结果\';

%% 画LOSS图！

linewidth = 1.5;
wordsize = 18;
width = 0.8;

%% Iteration LOSS曲线图
% loss_1 = load([path, 'poincare60_WN18RR.mat']).array(1:100);
% loss_2 = load([path, 'distmult60_WN18RR.mat']).array(1:100);
% loss_3 = load([path, 'transE_p260_WN18RR.mat']).array(1:100);
% loss_4 = load([path, 'transE_p160_WN18RR.mat']).array(1:100);

loss_1 = load([path, 'p1_FB15k_loss_list.mat']).array(1:end);
loss_2 = load([path, 'p3_FB15k_loss_list.mat']).array(1:end);
loss_3 = load([path, 'p5_FB15k_loss_list.mat']).array(1:end);

x_epoch = [1:1:2000];
% x_epoch = [1:1:100];

figure(1)
plot(x_epoch, loss_1, 'LineWidth',linewidth);
hold on;
plot(x_epoch, loss_2, 'LineWidth',linewidth);
hold on;
plot(x_epoch, loss_3, 'LineWidth',linewidth);
% hold on;
% plot(x_epoch, loss_4, 'LineWidth',linewidth);

% legend_fig2 = legend('双曲', '乘性', '线性', '加性');

legend_fig2 = legend('\fontname{宋体}推理路径长度\fontname{Times New Roman}L=1', '\fontname{宋体}推理路径长度\fontname{Times New Roman}L=3', '\fontname{宋体}推理路径长度\fontname{Times New Roman}L=5');

xlabel('训练轮次')
ylabel('损失值')

grid on % 显示网格线
