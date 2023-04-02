%% 画SNR图！

linewidth = 1.5;
wordsize = 18;
width = 0.8;

load("murp_ent_embed_analysis_WN18RR_20d.mat");

%% plot

% 生成随机的（可能重复）
% rand_low = randi([1, length(low)], 1, 10);
% rand_mid = randi([1, length(mid)], 1, 10);
% rand_high = randi([1, length(high)], 1, 10);

% 生成随机的（不重复）
rand_low = randperm(length(low));
rand_mid = randperm(length(mid));
rand_high = randperm(length(high));

rand_low = rand_low(1:50);
rand_mid = rand_mid(1:20);
rand_high = rand_high(1:5);
% 
low = low(rand_low,:);
mid = mid(rand_mid,:);
high = high(rand_high,:);

res_low = get_SNR(low);
res_mid = get_SNR(mid);
res_high = get_SNR(high);

snrs = -12:2:12;

figure(1)
semilogy(snrs, res_low, 'MarkerSize',9, 'LineWidth',linewidth); % Y轴调成对数形式
hold on;
semilogy(snrs, res_mid, 'MarkerSize',9, 'LineWidth',linewidth);
hold on;
semilogy(snrs, res_high, 'MarkerSize',9, 'LineWidth',linewidth);

legend_fig2 = legend('低层语义', '中层语义', '高层语义');