% Layered Semantics BER Calculation
% 分层语义信息的BER计算
% ent_low：低层语义实体的向量
% ent_mid：中层语义实体向量
% ent_high：高层语义实体向量
function res = get_SNR(ent_embed)
    %% 初始设置
    recovery_ratio = 0.3;
    n = 50000;
    index = 1:1:length(ent_embed(:, 1)); % 实体数量
    random_index = index(randi(numel(index), 1, n));
    
    snrs = -12:2:12;
    
    ser_list = zeros(1,length(snrs));
    ber_list = zeros(1,length(snrs));
    
    %% BER of Low-layer Semantics  
    for s = 1:1:length(snrs)
        acc = 0;
        for i = 1 :1: n
            signal = ent_embed(random_index(i),:); % 取得向量表示
            noised = awgn(signal, snrs(s),'measured'); % noised是噪声干扰后的向量表示
            temp = ent_embed - noised; % temp是原向量和干扰向量的差值
            mod = [];

            for j = 1:1:length(ent_embed(:,1))
                mod(j) = norm(temp(j,:)); % 对temp每一行进行范数求和
            end

            [minvalue, min_idex] = min(mod);

            if ismember(random_index(i),find(mod==min(mod))) % find(mod==min(mod)) 返回mod中最小值的索引
                % 如果当前的索引值index在干扰差值的最小值里面 则acc+1
                acc = acc+1;
            end
        end
        ser = 1-acc/n;
        ser_list(s) = ser;
    end

    ber_list = 1-nthroot(1-ser_list,8);

    res = ber_list - ber_list.*(1-ser_list(end)).*ber_list;
    % res = ber_list - recovery_ratio.*(1-ber_list).*(1-ser_list(end)).*ber_list;
end