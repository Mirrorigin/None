linewidth = 1.5;
wordsize = 18;

% 不同维度精度

x_dim = [20:20:120, 160, 200];

% WN18RR
% murp = [0.5230, 0.5528, 0.5616, 0.5522, 0.5654, 0.5512, 0.5501, 0.5507];
% transE_p1 = [0.1457, 0.1594, 0.21, 0.2261, 0.2446, 0.2535, 0.2653, 0.2687];
% transE_p2 = [0.2443, 0.2870, 0.3320, 0.3224, 0.3290, 0.3346, 0.3357, 0.3393];
% distmult = [0.0651, 0.1343, 0.2130, 0.2005, 0.2323, 0.2470, 0.2647, 0.2714];

% FB15k
murp = [0.4762, 0.4928, 0.4993, 0.5025, 0.5040, 0.5057, 0.5070, 0.5043];
transE_p1 = [0.1631, 0.1742, 0.1786, 0.1777, 0.1814, 0.1793, 0.1811, 0.1793];
transE_p2 = [0.1777, 0.1846, 0.1913, 0.1894, 0.1917, 0.1894, 0.1880, 0.1909];
distmult = [0.1608, 0.1712, 0.1769, 0.1777, 0.1771, 0.1820, 0.1793, 0.1880];

%% 三次拟合
P1 = polyfit(x_dim, murp, 3);
P2 = polyfit(x_dim, transE_p1, 3);
P3 = polyfit(x_dim, transE_p2, 3);
P4 = polyfit(x_dim, distmult, 3);

xx_dim = 20: 1 : 200;
murp_ = polyval(P1, xx_dim);
transE_p1_ = polyval(P2, xx_dim);
transE_p2_ = polyval(P3, xx_dim);
distmult_ = polyval(P4, xx_dim);

%% 画图
figure(1)
plot(xx_dim, murp_, 'LineWidth',linewidth);
hold on;
plot(xx_dim, distmult_, 'LineWidth',linewidth);
hold on;
plot(xx_dim, transE_p2_, 'LineWidth',linewidth);
hold on;
plot(xx_dim, transE_p1_, 'LineWidth',linewidth);

legend_fig2 = legend('\fontname{宋体}非欧嵌入编码', '\fontname{宋体}乘性嵌入编码', '\fontname{宋体}线性嵌入编码', '\fontname{宋体}加性嵌入编码', 'Fontsize', 14);

xlabel('\fontname{宋体}嵌入维度\fontsize{16}', 'Fontsize', 16)
ylabel('\fontname{宋体}推理精度 \fontname{Time New Roman}(%)', 'Fontsize', 16)