import numpy as np
import torch
import time
from collections import defaultdict
from load_data import Data
from model import *
from model_TransE_DistMult import *
from rsgd import *
import torch.optim as optim
import argparse
import os
from scipy.io import savemat

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Experiment:

    def __init__(self, p_norm = 1, learning_rate=50, dim=40, nneg=50, model="poincare",
                 num_iterations=500, batch_size=128, cuda=True): # 目前的设备没有cuda就直接默认设为False了
        self.model = model
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.p_norm = p_norm
        self.cuda = cuda

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].append(triple[idxs[2]])
        return er_vocab

    def save_list(self, loss_list, acclist_hit1, acclist_hit3, acclist_hit10, acclist_mr, acclist_mrr): # 存储列表list
        savemat('losslist_' + state_name + '.mat', {'array': loss_list})
        savemat('acclist_hit1_' + state_name + '.mat', {'array': acclist_hit1})
        savemat('acclist_hit3_' + state_name + '.mat', {'array': acclist_hit3})
        savemat('acclist_hit10_' + state_name + '.mat', {'array': acclist_hit10})
        savemat('acclist_mr_' + state_name + '.mat', {'array': acclist_mr})
        savemat('acclist_mrr_' + state_name + '.mat', {'array': acclist_mrr})

    def save_embedding(self, model): # 存储训练好的embedding
        # 先把gpu上的tensor转化为cpu的tensor
        ent_embed = model.Eh.weight.cpu()
        rel_embed = model.rvh.weight.cpu()
        # 再把cpu的tensor转化为numpy
        ent_embed = ent_embed.detach().numpy()
        rel_embed = rel_embed.detach().numpy()
        # 保存到字典
        entity_embed_dict = {d.entities[i]: ent_embed[i] for i in range(len(d.entities))}
        relation_embed_dict = {d.relations[i]: rel_embed[i] for i in range(len(d.relations))}
        # 存储
        np.save('ent_embed_' + state_name + '.npy', entity_embed_dict)
        np.save('rel_embed_' + state_name + '.npy', relation_embed_dict)


    def evaluate(self, model, data, acclist_hit1, acclist_hit3, acclist_hit10, acclist_mr, acclist_mrr):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs)):
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0])
            r_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions_s = model.forward(e1_idx.repeat(len(d.entities)),
                            r_idx.repeat(len(d.entities)), range(len(d.entities)))

            filt = sr_vocab[(data_point[0], data_point[1])]
            target_value = predictions_s[e2_idx].item()
            predictions_s[filt] = -np.Inf
            predictions_s[e1_idx] = -np.Inf
            predictions_s[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(predictions_s, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs==e2_idx.item())[0][0]
            ranks.append(rank+1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))

        acclist_hit1.append(np.mean(hits[0]))
        acclist_hit3.append(np.mean(hits[2]))
        acclist_hit10.append(np.mean(hits[9]))
        acclist_mr.append(np.mean(ranks))
        acclist_mrr.append(np.mean(1./np.array(ranks)))


    def train_and_eval(self):
        print("Training the %s model on %s dataset with %d dims..." % (self.model, dataset, self.dim))
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        # 模型选择
        if self.model == "poincare":
            model = MuRP(d, self.dim)
        else:
            model = MuRE(d, self.dim)

        print('RSGD Optimization...')
        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names) # RSGD优化

        if self.cuda:
            model.cuda()

        er_vocab = self.get_er_vocab(train_data_idxs)

        loss_list = []
        acclist_hit1 = []
        acclist_hit3 = []
        acclist_hit10 = []
        acclist_mr = []
        acclist_mrr = []
        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()

            losses = []

            np.random.shuffle(train_data_idxs)
            for j in range(0, len(train_data_idxs), self.batch_size):
                data_batch = np.array(train_data_idxs[j:j+self.batch_size])
                negsamples = np.random.choice(list(self.entity_idxs.values()),
                                              size=(data_batch.shape[0], self.nneg))

                # e1_idx是tensor张量，如果不加以处理会报错IndexError: tensors used as indices must be long, byte or bool tensors
                # 索引必须是 long, byte 或者 bool tensors
                # 处理一下.type(torch.long)
                e1_idx = torch.tensor(np.tile(np.array([data_batch[:, 0]]).T, (1, negsamples.shape[1]+1))).type(torch.long)
                r_idx = torch.tensor(np.tile(np.array([data_batch[:, 1]]).T, (1, negsamples.shape[1]+1))).type(torch.long)
                e2_idx = torch.tensor(np.concatenate((np.array([data_batch[:, 2]]).T, negsamples), axis=1)).type(torch.long)

                targets = np.zeros(e1_idx.shape)
                targets[:, 0] = 1
                targets = torch.DoubleTensor(targets)

                opt.zero_grad()
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()
                    targets = targets.cuda()

                predictions = model.forward(e1_idx, r_idx, e2_idx)
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(it)
            print(time.time()-start_train)
            print(np.mean(losses))
            loss_list.append(np.mean(losses))
            model.eval()
            with torch.no_grad():
                if not it%100:
                    print("Test:")
                    self.evaluate(model, d.test_data, acclist_hit1, acclist_hit3, acclist_hit10, acclist_mr, acclist_mrr)

        # 存储list
        self.save_list(loss_list, acclist_hit1, acclist_hit3, acclist_hit10, acclist_mr, acclist_mrr)

        # 存储训练好的embedding：
        self.save_embedding(model)

    # transE模型

    def get_adj_list(self, data, ent_idx, rel_idx):
        adj_list = {}
        for i in data:
            if rel_idx[i[1]] not in adj_list:
                adj_list[rel_idx[i[1]]] = np.zeros((len(ent_idx), len(ent_idx)))
            adj_list[rel_idx[i[1]]][ent_idx[i[0]], ent_idx[i[2]]] = 1
            adj_list[rel_idx[i[1]]][ent_idx[i[2]], ent_idx[i[0]]] = 1
        return adj_list


    def sample_neg(self, data_idx, adj_list, ent_idx):
        neg_data = []
        k = 0
        while len(neg_data) < len(data_idx):
            rel = data_idx[k][1]
            i, j = np.random.randint(0, len(ent_idx)-1), np.random.randint(0, len(ent_idx)-1)
            if i != j and adj_list[rel][i, j] == 0:
                neg_data.append((i, rel, j))
                k = k + 1
            # else add negative examples of no-link, if need be
            else:
                continue
        return neg_data

    def evaluate_transE_or_distmult(self, model, data, acclist_hit1, acclist_hit3, acclist_hit10, acclist_mr, acclist_mrr):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs)):
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0])
            r_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            predictions_s = model.forward(e1_idx.repeat(len(d.entities)), r_idx.repeat(len(d.entities)), range(len(d.entities)))

            filt = sr_vocab[(data_point[0], data_point[1])] # 过滤掉训练数据集中存在的
            target_value = predictions_s[e2_idx].item()
            predictions_s[filt] = -np.Inf
            predictions_s[e1_idx] = -np.Inf
            predictions_s[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(predictions_s, descending=False) # transE这里应该是升序排列而非降序！

            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs == e2_idx.item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        acclist_hit1.append(np.mean(hits[0]))
        acclist_hit3.append(np.mean(hits[2]))
        acclist_hit10.append(np.mean(hits[9]))
        acclist_mr.append(np.mean(ranks))
        acclist_mrr.append(np.mean(1. / np.array(ranks)))

    def train_and_eval_transE_or_distmult(self):
        print("Training the %s model on %s dataset with %d dims..." % (self.model, dataset, self.dim))
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        adj_list = self.get_adj_list(d.data, self.entity_idxs, self.relation_idxs)

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        # 模型选择
        if self.model == "transE":
            model = transE(d, self.dim, self.p_norm)
            print('Ready for transE model training...')
            print('p_norm: ', self.p_norm)
        else:
            model = distmult(d, self.dim)
            print('Ready for distmult model training...')

        print('Euclidean SGD Optimization...')
        # opt = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=params.momentum) # Adam优化
        opt = optim.Adam(model.parameters(), lr=self.learning_rate)

        if self.cuda:
            model.cuda()

        loss_list = []
        acclist_hit1 = []
        acclist_hit3 = []
        acclist_hit10 = []
        acclist_mr = []
        acclist_mrr = []
        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()

            losses = []

            np.random.shuffle(train_data_idxs)
            # 每一个大轮生成一次负样本
            negative_triples = self.sample_neg(train_data_idxs, adj_list, self.entity_idxs) # 抽取负样本不要变换关系（论文里是这样写的）
            for j in range(0, len(train_data_idxs), self.batch_size):
                data_batch = np.array(train_data_idxs[j:j + self.batch_size])
                neg_batch = np.array(negative_triples[j:j + self.batch_size])

                e1_idx = torch.cat((torch.tensor(data_batch[:, 0]).type(torch.long), torch.tensor(neg_batch[:, 0]).type(torch.long)))
                r_idx = torch.cat((torch.tensor(data_batch[:, 1]).type(torch.long), torch.tensor(neg_batch[:, 1]).type(torch.long)))
                e2_idx = torch.cat((torch.tensor(data_batch[:, 2]).type(torch.long), torch.tensor(neg_batch[:, 2]).type(torch.long)))

                opt.zero_grad()
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()

                predictions = model.forward(e1_idx, r_idx, e2_idx)

                pos_score = predictions[0: int(len(predictions) / 2)]
                neg_score = predictions[int(len(predictions) / 2): len(predictions)]

                loss = model.criterion(pos_score, neg_score, torch.tensor([-1], dtype=torch.long).cuda()).mean()
                loss.backward()
                opt.step()
                losses.append(float(loss.item() / self.batch_size))

            print(it)
            print(time.time() - start_train)
            print(np.mean(losses))
            loss_list.append(np.mean(losses))
            model.eval()
            with torch.no_grad():
                if not it % 100:
                    print("Test:")
                    self.evaluate_transE_or_distmult(model, d.test_data, acclist_hit1, acclist_hit3, acclist_hit10, acclist_mr, acclist_mrr)

        # 存储list
        self.save_list(loss_list, acclist_hit1, acclist_hit3, acclist_hit10, acclist_mr, acclist_mrr)

        # 存储训练好的embedding：
        self.save_embedding(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WN18RR", nargs="?",
                    help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="distmult", nargs="?",
                    help="Which model to use: poincare or transE or distmult or MuRE.")
    parser.add_argument("--num_iterations", type=int, default=100, nargs="?",
                    help="Number of iterations.")
    # 双曲空间：128 batch
    # TransE：400 batch
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--nneg", type=int, default=50, nargs="?",
                    help="Number of negative samples.")
    # RSGD学习率一般设为50，SGD学习率设为0.01
    parser.add_argument("--lr", type=float, default=0.01, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dim", type=int, default=50, nargs="?",
                    help="Embedding dimensionality.")
    parser.add_argument("--p_norm", type=int, default=1, nargs="?",
                    help="求和范数.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).") # 目前的设备没有cuda就直接默认设为False了

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_name = args.model + str(args.dim) + '_' + args.dataset
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir)
    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size,
                            num_iterations=args.num_iterations, dim=args.dim,
                            cuda=args.cuda, nneg=args.nneg, model=args.model, p_norm = args.p_norm)
    if args.model == 'transE' or args.model == 'distmult':
        if args.model =='transE':
            state_name = state_name + '_' + str(args.p_norm)
        experiment.train_and_eval_transE_or_distmult()
    else:
        experiment.train_and_eval()


