% 第四章说明图

linewidth = 1.5;
wordsize = 18;
width = 0.8;

%———————————————————————————————————————

% PCA - 生成三维图
% embed1 = load('E:\学习云盘\毕业论文\code\Matlab\Chapter4_data\PCA_embed_person_3d.mat');
% embed2 = load('E:\学习云盘\毕业论文\code\Matlab\Chapter4_data\PCA_embed_scientist_3d.mat');
% embed3 = load('E:\学习云盘\毕业论文\code\Matlab\Chapter4_data\PCA_embed_fruit_3d.mat');
% embed4 = load('E:\学习云盘\毕业论文\code\Matlab\Chapter4_data\PCA_embed_tele_3d.mat');
%
% x1 = [embed1.array(1:300, 1)];
% y1 = [embed1.array(1:300, 2)];
% z1 = [embed1.array(1:300, 3)];
% 
% x2 = [embed2.array(1:100, 1)];
% y2 = [embed2.array(1:100, 2)];
% z2 = [embed2.array(1:100, 3)];
% 
% x3 = [embed3.array(1:end, 1)];
% y3 = [embed3.array(1:end, 2)];
% z3 = [embed3.array(1:end, 3)];
% 
% x4 = [embed4.array(1:100, 1)];
% y4 = [embed4.array(1:100, 2)];
% z4 = [embed4.array(1:100, 3)];
% 
% 
% figure(1)
% scatter3(x1, y1, z1, 'filled');
% hold on;
% scatter3(x2, y2, z2, 'filled');
% hold on;
% scatter3(x3, y3, z3, 'filled');
% hold on;
% scatter3(x4, y4, z4, 'filled');

%———————————————————————————————————————

% PCA - 生成二维图
embed1 = load('murp_embed_pca_university.mat');
embed2 = load('murp_embed_pca_book.mat');
embed3 = load('murp_embed_pca_musicalbum.mat');
embed4 = load('murp_embed_pca_chemical.mat');
% label1 = load('label_1.mat'); % 读取出来是char数组（char array）
% label2 = load('label_2.mat');
% label3 = load('label_3.mat');

x1 = [embed1.array(1:end, 1)];
y1 = [embed1.array(1:end, 2)];

x2 = [embed2.array(1:end, 1)];
y2 = [embed2.array(1:end, 2)];

x3 = [embed3.array(1:end, 1)];
y3 = [embed3.array(1:end, 2)];

x4 = [embed4.array(1:100, 1)];
y4 = [embed4.array(1:100, 2)];

figure(1)
% scatter(x1, y1, 'filled');
% hold on;
scatter(x2, y2, 'filled');
hold on;
scatter(x3, y3, 'filled');
hold on;
scatter(x4, y4, 'filled');

legend('实体概念："书籍”', '实体概念："音乐专辑”', '实体概念："化学”')

% 加文字标签
% 转换为字符串形式（否则char是一个一个字母读取的）
% label1 = cellstr(label1.array);
% label2 = cellstr(label2.array);
% 全部打上图标太密集了，选择几个代表性词语
% drug_label = {'viagra','seloken','proactol'};
% city_label = {'bridgetown','marrero','brantford','wellington', 'kansas'};
% for i = 1:max(size(x1))
%     if ismember(label1(i), city_label)
%         text(x1(i),y1(i),label1(i), 'Color','blue','FontSize',10);% 在图上显示文字
%     end
% end
% 
% for i = 1:max(size(x2))
%     if ismember(label2(i), drug_label)
%         text(x2(i),y2(i),label2(i), 'Color','red','FontSize',10);% 在图上显示文字
%     end
% end