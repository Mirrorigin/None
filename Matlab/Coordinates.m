% ������˵��ͼ

linewidth = 1.5;
wordsize = 18;
width = 0.8;

%������������������������������������������������������������������������������

% PCA - ������άͼ
% embed1 = load('E:\ѧϰ����\��ҵ����\code\Matlab\Chapter4_data\PCA_embed_person_3d.mat');
% embed2 = load('E:\ѧϰ����\��ҵ����\code\Matlab\Chapter4_data\PCA_embed_scientist_3d.mat');
% embed3 = load('E:\ѧϰ����\��ҵ����\code\Matlab\Chapter4_data\PCA_embed_fruit_3d.mat');
% embed4 = load('E:\ѧϰ����\��ҵ����\code\Matlab\Chapter4_data\PCA_embed_tele_3d.mat');
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

%������������������������������������������������������������������������������

% PCA - ���ɶ�άͼ
embed1 = load('murp_embed_pca_university.mat');
embed2 = load('murp_embed_pca_book.mat');
embed3 = load('murp_embed_pca_musicalbum.mat');
embed4 = load('murp_embed_pca_chemical.mat');
% label1 = load('label_1.mat'); % ��ȡ������char���飨char array��
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

legend('ʵ����"�鼮��', 'ʵ����"����ר����', 'ʵ����"��ѧ��')

% �����ֱ�ǩ
% ת��Ϊ�ַ�����ʽ������char��һ��һ����ĸ��ȡ�ģ�
% label1 = cellstr(label1.array);
% label2 = cellstr(label2.array);
% ȫ������ͼ��̫�ܼ��ˣ�ѡ�񼸸������Դ���
% drug_label = {'viagra','seloken','proactol'};
% city_label = {'bridgetown','marrero','brantford','wellington', 'kansas'};
% for i = 1:max(size(x1))
%     if ismember(label1(i), city_label)
%         text(x1(i),y1(i),label1(i), 'Color','blue','FontSize',10);% ��ͼ����ʾ����
%     end
% end
% 
% for i = 1:max(size(x2))
%     if ismember(label2(i), drug_label)
%         text(x2(i),y2(i),label2(i), 'Color','red','FontSize',10);% ��ͼ����ʾ����
%     end
% end