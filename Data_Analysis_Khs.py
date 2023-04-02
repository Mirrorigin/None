import numpy as np

def load_data( data_dir, data_type="train"):
    with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:  # 一口气读取完所有数据
        # data形如 ['00260881\t_hypernym\t00260622', '01332730\t_derivationally_related_form\t03122748', ...,]
        data = f.read().strip().split("\n")  # 分割出一行（即一组三元组）
        # 对data中进行空格切分：
        # data 形如 [['00260881', '_hypernym', '00260622'], ['01332730', '_derivationally_related_form', '03122748'], ..., ]
        data = [i.split() for i in data]
    return data

def entities(data):
    entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
    return entities

dataset = "NELL-995-h75"
data_dir = "data/%s/" % dataset
train_data = load_data(data_dir, "train")
valid_data = load_data(data_dir, "valid")
test_data = load_data(data_dir, "test")
data = train_data + valid_data + test_data

relation_data = {}
for i in data:
    if i[1] not in relation_data:
        relation_data[i[1]] = []
    relation_data[i[1]].append(i) # 把包含该关系的三元组存储起来

for rel in relation_data.keys():
    entity_list = entities(relation_data[rel])
    length = len(entity_list)
    entity_idxs = {entity_list[i]: i for i in range(length)} # 实体名：索引

    R_matrix = np.zeros((length, length), dtype=int)

for i in relation_data:
    R_matrix[entity_idxs[i[0]]][entity_idxs[i[2]]] = 1

numerator = 0
denominator = 0
for i in range(0, length):
    for j in range(0, length):
        if(R_matrix[i][j] == 1):
            denominator += 1
            if(R_matrix[j][i] == 0):
                numerator += 1

khs = float(numerator) / denominator
print("khs：%f" % khs)
print('End')
