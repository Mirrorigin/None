%准备数据
Y = [0.5427, 0.5293, 0.5478; 0.4950, 0.4225, 0.3625; 0.5057, 0.6703, 0.6602];
X = 1: 3;
h = bar(X, Y, 1);
hold on;

% 设置颜色
set(h(1),'FaceColor',[250, 170, 137]/255)     
set(h(2),'FaceColor',[255, 220, 126]/255)    
set(h(3),'FaceColor',[173, 206, 215]/255)    

set(gca,'XTickLabel',{'\fontname{{Times New Roman}WN18RR','\fontname{{Times New Roman}FB15k-237','\fontname{{Times New Roman}NELL-995'},'FontSize',10)

%修改x,y轴标签
ylabel('\fontname{宋体}\fontsize{14}推理准确度 \fontname{{Times New Roman}(%)');
xlabel('\fontname{宋体}\fontsize{14}不同知识数据集'); 

%修改图例
legend({'\fontname{宋体}推理路径长度\fontname{Times New Roman}L=1','\fontname{宋体}推理路径长度\fontname{Times New Roman}L=3','\fontname{宋体}推理路径长度\fontname{Times New Roman}L=5'},'FontSize',11);