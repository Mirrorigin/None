# import umap
# import umap.plot
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from scipy.io import savemat
import scipy.io as scio

# 读取源数据
metadata = pd.read_csv('./embedding_list.csv', header=None, sep=',')
label = pd.read_csv('./label_list.csv', header=None, sep=',')
# 读取源数据（源文件为tsv版）
# metadata=pd.read_csv('E:\学习云盘\毕业论文\code\Matlab\embed_three_class.csv', header=None, sep='\t')
# label = pd.read_csv('E:\学习云盘\毕业论文\code\Matlab\label_three_class.csv', header=None, sep='\t')

pca = PCA(n_components=2) # 使用PCA
pca.fit(metadata)
embedding = pca.transform(metadata)

# 不同标签类型的实体向量
N = 4 # 标签种类

embed_pca_dic = {}
label_dic = {}

for i in range(N):
    p = 'label_' + str(i)
    embed_pca_dic[p] = []
    label_dic[p] = []

for i in range(len(label)):
    for j in range(N):
        if int(label[0][i]) == j:
            p = 'label_' + str(j)
            embed_pca_dic[p].append((embedding[i]))
            label_dic[p].append(label[1][i])
            break

for i in range(N):
    savemat('ent_embed_pca.mat', embed_pca_dic)
    savemat('label_list.mat', label_dic)


# 使用UMAP
# mapper = umap.UMAP().fit(digits.data)
# embedding = umap.UMAP().fit_transform(metadata)
# np.savetxt('./csv/TransE_2d_40.tsv', embedding, delimiter='\t')
# umap.plot.points(mapper, labels=digits.target)