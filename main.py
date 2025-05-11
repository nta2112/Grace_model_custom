import argparse, os
import math
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from data_split import *
import time
from utils import  *
from models import *
import os
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# python main.py --use_cor

def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--dataset', default='dblp')
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_shot', type=int, help='k shot', default=3)
    parser.add_argument('--m_qry', type=int, help='m query', default=10)
    parser.add_argument('--test_num', type=int, help='test number', default=100)
    parser.add_argument('--loss_temperature', type=float, default=0.5)
    parser.add_argument('--loss_cls', type=float, default=1.0)
    parser.add_argument('--loss_orth', type=float, default=0.2)
    parser.add_argument('--model_temperature', type=float, default=2.0)
    parser.add_argument('--stepsize', type=int, default=10)
    parser.add_argument('--stepgamma', type=float, default=0.99)

    return parser

torch.cuda.set_device(0)
if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        exit()
    print(args)
    test_num = args.test_num
    n_way = args.n_way
    k_shot = args.k_shot
    m_qry = args.m_qry

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    bandwidth = 1.0
    dropout = dropt.get(args.dataset, 0)
    hidden_size = hidden.get(args.dataset, 32)
    patience = patience.get(args.dataset, 10)
    lr = lr.get(args.dataset, 0.001)
    gate = gate.get(args.dataset, 0.1)
    contrast = contrast.get(args.dataset, 1.)
    # Loading data
    data, train_idx, id_by_class, train_class, dev_class, test_class, degree_inv = split(args.dataset)
    features = data.x
    # features = (features - features.min(dim=0)[0]) / (features.max(dim=0)[0] - features.min(dim=0)[0] + 1e-5)
    # features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-5)

    data.edge_index = data.edge_index
    adj = torch.sparse.FloatTensor(
        data.edge_index,
        torch.ones(data.edge_index.size(1)),
        torch.Size([data.num_nodes, data.num_nodes])
    )

    identity = torch.sparse.FloatTensor(
        torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)]),
        torch.ones(data.num_nodes),
        torch.Size([data.num_nodes, data.num_nodes])
    )
    adj = adj + identity
    adj = adj.coalesce()
    row, col = adj.indices()
    deg = torch.sparse.sum(adj, dim=1).to_dense()  
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  
    D_inv_sqrt = torch.sparse.FloatTensor(
        torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)]),
        deg_inv_sqrt,
        torch.Size([data.num_nodes, data.num_nodes])
    )
    adj = torch.sparse.mm(torch.sparse.mm(D_inv_sqrt, adj), D_inv_sqrt)
    labels = data.y

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = adj.to(device)
    features = features.to(device)
    data.x = data.x.to(device)
    labels = labels.to(device)


    def train(model, optimizer):
        # Model training
        model.train()
        optimizer.zero_grad()

        args.unsup = False
        class_batch_n = n_way
        sampled_class = random.sample(range(len(train_class)), class_batch_n)
        sample_idx_n = []
        for c in sampled_class:
            sample_idx_n.extend(random.sample(id_by_class[c], k_shot))
        sample_idx_q = []
        for c in sampled_class:
            sample_idx_q.extend(random.sample(id_by_class[c], m_qry))
        id_support = sample_idx_n
        id_query = sample_idx_q
        embeddings, gate_weights, low_feature, high_feature, corrected_prototypes = model.forward_with_correction(
            features, adj, id_support, id_query, n_way, k_shot, bandwidth
        )
        prototype_embeddings = corrected_prototypes 
        query_embeddings = embeddings[id_query]
        dists = euclidean_dist(query_embeddings, prototype_embeddings)
        output = F.log_softmax(-dists, dim=1)
        labels_new = torch.LongTensor([sampled_class.index(i) for i in labels[id_query]]).to(device)
        labels_support = torch.LongTensor([sampled_class.index(int(i)) for i in labels[id_support]]).to(device)
        labels_query = torch.LongTensor([sampled_class.index(int(i)) for i in labels[id_query]]).to(device)
        loss_total = new_loss_function_with_contrast(
            output, labels_new, gate_weights, low_feature, high_feature,
            embeddings[id_support], labels_support, embeddings[id_query], labels_query,
            lambda_cls=args.loss_cls, lambda_gate=gate, lambda_orth=args.loss_orth, lambda_contrast=contrast, temperature=args.loss_temperature
        )
        loss_total.backward()
        optimizer.step()
        output_cpu = output.cpu().detach()
        labels_cpu = labels_new.cpu().detach()
        acc_train = accuracy(output_cpu, labels_cpu)
        f1_train = f1(output_cpu, labels_cpu)
        # print("acc train {}".format(acc_train))


        return loss_total.item(), acc_train, f1_train
    def test(model, eval_class, output_ari=False,draw=False):
        draw=False
        # Model testing
        model.eval()
        # sample downstream few-shot tasks
        test_acc_all = []
        purity_all = 0.
        nmi_all = 0.
        ari_all = 0.
        test_acc = 0.
        n_way = args.n_way
        k_shot = args.k_shot

        for i in range(test_num):
            test_id_support, test_id_query, test_class_selected = \
                test_task_generator(id_by_class, eval_class, n_way, k_shot, m_qry)
            embeddings, gate_weights, low_feature, high_feature, corrected_prototypes = model.forward_with_correction(
                features, adj, test_id_support, test_id_query, n_way, k_shot, bandwidth
            )
            prototype_embeddings = corrected_prototypes
            query_embeddings = embeddings[test_id_query]
            dists = euclidean_dist(query_embeddings, prototype_embeddings)
            output = F.log_softmax(-dists, dim=1)
            labels_new = torch.LongTensor([test_class_selected.index(i) for i in data.y[test_id_query]])
            labels_new = labels_new.to(device)
            output_cpu = output.cpu().detach()
            labels_cpu = labels_new.cpu().detach()
            acc_test = accuracy(output_cpu, labels_cpu)
            f1_test = f1(output_cpu, labels_cpu)
            test_acc_all.append(acc_test)
            # test_acc_all.append(lo_test)
        m, s = np.mean(test_acc_all), np.std(test_acc_all)
        interval = 1.96 * (s / np.sqrt(len(test_acc_all)))

        # print("="*40)
        print('test_acc = {}'.format(m))
        # print('test_interval = {}'.format(interval))
        return m, s, interval

    def train_eval():
        model = MOE(nfeat=features.shape[1], nhid=hidden_size, dropout=dropout,temperature=args.model_temperature)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.stepgamma)
        model.to(device)
        print('Start training !!!')
        best_test_acc = 0
        stop_cnt = 0
        best_epoch = 0
        time_begin = time.time()
        for epoch in range(10000):
            loss = train(model, optimizer)
            scheduler.step()
            if epoch % 100 == 0:
                print('epoch = {}, loss = {}'.format(epoch, loss))

            # validation
            if epoch % 10 == 0 and epoch != 0:
                test_acc, _, _ = test(model, dev_class)
                if test_acc >= best_test_acc:
                    if test_acc == 1.0 and test_acc == best_test_acc:
                        best_test_acc = test_acc
                        best_epoch = epoch
                        stop_cnt += 1
                    else:
                        best_test_acc = test_acc
                        best_epoch = epoch
                        stop_cnt = 0
                else:
                    stop_cnt += 1
                if stop_cnt > patience:
                    print('Time', time.time() - time_begin, 'Epoch: {}'.format(epoch))
                    break

        # final test
        acc, std, interval = test(model, test_class, output_ari=True,draw=True)

        print("Current acc mean: " + str(acc))
        print("Current acc std: " + str(std))
        print("Current interval: " + str(interval))
        return acc, std, interval


    acc_mean = []
    acc_std = []
    acc_interval = []
    for __ in range(5):
        m, s, interval = train_eval()
        acc_mean.append(m)
        acc_std.append(s)
        acc_interval.append(interval)
    print("****" * 20)
    print("Final acc: " + str(np.mean(acc_mean)))
    print("Final acc std: " + str(np.mean(acc_std)))
    print("Final acc interval: " + str(np.mean(acc_interval)))



