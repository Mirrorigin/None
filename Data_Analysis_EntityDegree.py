# 对WN18RR的实体度进行分析
import nltk
from nltk.corpus import wordnet
import numpy as np
from scipy.io import savemat
import pandas as pd

def load_data(data_dir, data_type="train"):
    with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:  # 一口气读取完所有数据
        # data形如 ['00260881\t_hypernym\t00260622', '01332730\t_derivationally_related_form\t03122748', ...,]
        data = f.read().strip().split("\n")  # 分割出一行（即一组三元组）
        # 对data中进行空格切分：
        # data 形如 [['00260881', '_hypernym', '00260622'], ['01332730', '_derivationally_related_form', '03122748'], ..., ]
        data = [i.split() for i in data]
        return data

def count_idx(dataset):
    data_dir = "data/%s/" % dataset
    train_data = load_data(data_dir, "train")
    valid_data = load_data(data_dir, "valid")
    test_data = load_data(data_dir, "test")
    data = data = train_data + valid_data + test_data
    degree = {}

    if dataset == "WN18RR":
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        syns = list(wordnet.all_synsets())
        offsets_list = [(s.offset(), s) for s in syns]
        offsets_dict = dict(offsets_list)

    for i in data:
        if i[0] not in degree:
            degree[i[0]] = 1
        if i[2] not in degree:
            degree[i[2]] = 1
        else:
            degree[i[0]] += 1
            degree[i[2]] += 1

    print("总实体数量：%d" % len(degree))

    count_high = []
    count_mid = []
    count_low = []

    for entity in degree:
        if degree[entity] <= 10:
            # print("%s: %d" % (offsets_dict[int(entity)], degree[entity])) # WN18RR的输出
            # if entity.find('chemical') != -1 :
            #     print("%s: %d" % (entity, degree[entity]))
            count_low.append(entity)
        elif degree[entity] >= 100:
            count_high.append(entity)
        else:
            count_mid.append(entity)

    print("数据集：%s 中的度分析——高层语义数量：%d，中层语义数量：%d，低层语义数量：%d" % (dataset, len(count_high), len(count_mid), len(count_low)))
    print('Anlaysis End')

    return count_low, count_mid, count_high

def Embedding_read(dataset = 'WN18RR', name = 'poincare', dim = 60):
    embed_data = np.load('./results/ent_embed_' + name + str(dim) + '_' + dataset + '.npy') #加载文件
    embed_data = embed_data.item()
    print('Embedding Read Over')
    return embed_data

def layered_semantics(dataset):
    low_idx, mid_idx, high_idx = count_idx(dataset)
    ent_embed = Embedding_read(dataset = dataset)
    ent_embed_low = []
    ent_embed_mid = []
    ent_embed_high = []

    for i in range(len(low_idx)): ent_embed_low.append(ent_embed[low_idx[i]])
    for i in range(len(mid_idx)): ent_embed_mid.append(ent_embed[mid_idx[i]])
    for i in range(len(high_idx)): ent_embed_high.append(ent_embed[high_idx[i]])

    savemat('poincare_ent_embed_analysis_' + dataset + '.mat', {'low': ent_embed_low, 'mid': ent_embed_mid, 'high': ent_embed_high})

def diff_ways(dim):
    embed_transE_p1 = Embedding_read(name = 'transE_p1', dim = dim)
    embed_transE_p2 = Embedding_read(name='transE_p2', dim = dim)
    embed_distmult = Embedding_read(name='distmult', dim = dim)
    embed_poincare = Embedding_read(name='poincare', dim = dim)

    ent_embed_transE_p1 = list(embed_transE_p1.values())
    ent_embed_transE_p2 = list(embed_transE_p2.values())
    ent_embed_distmult = list(embed_distmult.values())
    ent_embed_poincare = list(embed_poincare.values())

    savemat('diff_ent_embed_analysis_WN18RR.mat', {'transE_p1': ent_embed_transE_p1, 'transE_p2': ent_embed_transE_p2, 'distmult': ent_embed_distmult, 'poincare': ent_embed_poincare})

def specify_label():

    ent_embed = Embedding_read(dataset = 'NELL-995-h100', name = 'poincare', dim = 40) # 主要是针对NELL数据集
    # 注意：训练的Trans没有归一化

    N = 4  # 总共看9个类别
    parts = ['university', 'chemical', 'book', 'musicalbum'] # 指定的标签

    # 创建列表，存储向量和标签
    embed_list = []
    label_name = []
    label_num = []

    for label in ent_embed: # ent_embed是一个字典
        concept, part, name = label.split('_', 2)
        for j in range(N):
            if part == parts[j]:
                embed_list.append(ent_embed[label])
                label_num.append(j)
                label_name.append(name)  # 保存city后面的标签名(分割两次并把剩下的保存下来）
            # break  # 找到对应类别就跳出循环
        else:
            continue

    '''
    # 一个归一化的步骤：（Trans系列的方法需要归一化，再进行PCA）
    for i in range(len(embed_list)):
        norm = np.linalg.norm(embed_list[i])
        embed_list[i] = embed_list[i] / norm
    '''

    embedding = pd.DataFrame(data=embed_list)
    embedding.to_csv('./embedding_list.csv', encoding='gbk', index = False, header=None) # header = None：不保存表头，index = False：不保存索引值
    label = pd.DataFrame(index=label_num, data=label_name)
    label.to_csv('./label_list.csv', encoding='gbk', header=None)

# 调用什么函数直接在这里写：

# layered_semantics(dataset = 'NELL-995-h100') # 生成中高低层语义实体向量
# diff_ways(dim = 20) # 生成不同方法的实体向量
# specify_label() # 抽取指定标签的实体向量
print('Over!')