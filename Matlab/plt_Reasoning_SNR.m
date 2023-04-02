%% 画SNR图！

linewidth = 1.5;
wordsize = 18;
width = 0.8;

load("diff_ent_embed_analysis_WN18RR_20d.mat");

%% 不推理

SignalLength = 10000000;
SignalSequence = rand(1, SignalLength);
SignalSequence(SignalSequence > 0.5) = 1;
SignalSequence(SignalSequence <= 0.5) = 0;
SignalBpskBit = 1 - 2*SignalSequence;
Bit_tx = SignalBpskBit;
noise_real = normrnd(0,sqrt(0.5),[1 SignalLength]);
noise_image = j*normrnd(0,sqrt(0.5),[1 SignalLength]);

SNR = (-12:2:12);
BER = zeros(1,length(SNR));
for i = 1:length(SNR)
    ratio = 10^(SNR(i)/10);
    Mesage_rx = sqrt(ratio)*Bit_tx+(noise_real + noise_image);
    Bit_rx = zeros(1,SignalLength);
    Bit_rx(Mesage_rx>0) = 1;
    Bit_rx(Mesage_rx<0) = -1;
    SignalBit_rx = 0.5*(1-Bit_rx);
    error = xor(SignalSequence,SignalBit_rx);
    BER(i) = (sum(error)) / SignalLength;
end

%% 不同嵌入方法的SNR

% 生成随机的（不重复）
rand_transE_p1 = randperm(length(transE_p1));
rand_transE_p2 = randperm(length(transE_p2));
rand_distmult = randperm(length(distmult));
rand_poincare = randperm(length(poincare));

rand_transE_p1 = rand_transE_p1(1:40);
rand_transE_p2 = rand_transE_p2(1:30);
rand_distmult = rand_distmult(1:50);
rand_poincare = rand_poincare(1:8);

transE_p1 = transE_p1(rand_transE_p1,:);
transE_p2 = transE_p2(rand_transE_p1,:);
distmult = distmult(rand_distmult,:);
poincare = poincare(rand_poincare,:);

res_transE_p1 = get_SNR(transE_p1);
res_transE_p2 = get_SNR(transE_p2);
res_distmult = get_SNR(distmult);
res_poincare = get_SNR(poincare);

snrs = -12:2:12;

figure(1)
semilogy(snrs, res_poincare, 'MarkerSize',9, 'LineWidth',linewidth);
hold on;
semilogy(snrs, res_distmult, 'MarkerSize',9, 'LineWidth',linewidth);
hold on;
semilogy(snrs, res_transE_p1, 'MarkerSize',9, 'LineWidth',linewidth); % Y轴调成对数形式
hold on;
semilogy(snrs, res_transE_p2, 'MarkerSize',9, 'LineWidth',linewidth);
hold on;
semilogy(snrs, BER, 'MarkerSize',9, 'LineWidth',linewidth);

legend_fig2 = legend('非欧嵌入', '乘性嵌入', '线性嵌入', '加性嵌入','无编码');