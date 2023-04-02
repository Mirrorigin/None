%准备数据
Y = [94, 1407, 20910; 882, 10731, 2928; 38, 2451, 38454];
entity_num = sum(Y, 2);
Y = Y ./ entity_num;
X = 1: 3;
h = bar(X, Y, 1);

% 设置颜色
set(h(1),'FaceColor',[250, 170, 137]/255)     
set(h(2),'FaceColor',[255, 220, 126]/255)    
set(h(3),'FaceColor',[173, 206, 215]/255)    

set(gca,'XTickLabel',{'\fontname{Times New Roman}NELL-995\fontname{宋体}子集','\fontname{Times New Roman}FB15k-237','\fontname{Times New Roman}WN18RR'},'FontSize',10)

%修改x,y轴标签
ylabel('\fontname{宋体}\fontsize{14}实体数量占比(%)');
xlabel('\fontname{宋体}\fontsize{14}不同知识数据集'); 

%修改图例
legend({'\fontname{宋体}高层语义','\fontname{宋体}中层语义','\fontname{宋体}低层语义'},'FontSize',11);