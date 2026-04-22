import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CoraFull, Reddit2, Coauthor, Planetoid, Amazon, DBLP
import random
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import json
import os

class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "Coauthor-CS": {"train": 5, 'dev': 5, 'test': 5},
    "Amazon-Computer": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
    "Reddit": {"train": 21, 'dev': 10, 'test': 10},
    'dblp': {"train": 77, 'dev': 30, 'test': 30},
    'tlu': {"train": 21, 'dev': 5, 'test': 5},
}

class dblp_data():
    def __init__(self):
        self.x = None
        self.edge_index = None
        self.num_nodes = None
        self.y = None
        self.num_edges = None
        self.num_features = 7202

class dblp_dataset():
    def __init__(self, data, num_classes, lb=None):
        self.data = data
        self.num_classes = num_classes
        self.lb = lb

def load_DBLP(root='/content/duno/', dataset_source='dblp'):
    dataset = dblp_data()
    n1s = []
    n2s = []
    network_path = os.path.join(root, "{}_network".format(dataset_source))
    for line in open(network_path):
        n1, n2 = line.strip().split('\t')
        if int(n1) > int(n2):
            n1s.append(int(n1))
            n2s.append(int(n2))

    num_nodes = max(max(n1s), max(n2s)) + 1
    print('nodes num', num_nodes)

    train_path = os.path.join(root, "{}_train.mat".format(dataset_source))
    test_path = os.path.join(root, "{}_test.mat".format(dataset_source))
    data_train = sio.loadmat(train_path)
    data_test = sio.loadmat(test_path)

    raw_labels = np.empty((num_nodes, 1), dtype=object)
    raw_labels[data_train['Index']] = data_train["Label"]
    raw_labels[data_test['Index']] = data_test["Label"]

    raw_labels_flat = raw_labels.flatten()
    
    lb = preprocessing.LabelBinarizer()
    lb.fit(raw_labels_flat)
    labels_encoded = lb.transform(raw_labels_flat)
    
    features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    features = torch.FloatTensor(features)
    if labels_encoded.shape[1] == 1:
        labels = torch.LongTensor(labels_encoded.flatten())
    else:
        labels = torch.LongTensor(np.where(labels_encoded)[1])

    dataset.edge_index = torch.tensor([n1s, n2s])
    dataset.y = labels
    dataset.x = features
    dataset.num_nodes = num_nodes
    dataset.num_edges = dataset.edge_index.shape[1]

    return dblp_dataset(dataset, num_classes=len(lb.classes_), lb=lb)

def split(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(root='./dataset/' + dataset_name, name="Cora")
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root='./dataset/' + dataset_name, name="CiteSeer")
    elif dataset_name == 'Amazon-Computer':
        dataset = Amazon(root='./dataset/' + dataset_name, name="Computers")
    elif dataset_name == 'Coauthor-CS':
        dataset = Coauthor(root='./dataset/' + dataset_name, name="CS")
    elif dataset_name == 'CoraFull':
        dataset = CoraFull(root='./dataset/' + dataset_name)
    elif dataset_name == 'Reddit':
        dataset = Reddit2(root='./dataset/' + dataset_name)
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=dataset_name, root='./dataset/' + dataset_name)
    elif dataset_name == 'dblp' or dataset_name == 'tlu':
        dataset = load_DBLP()
    else:
        print("Dataset not support!")
        exit(0)

    data = dataset.data
    num_nodes = data.num_nodes
    
    json_path = "/content/data/full_split.json"
    if os.path.exists(json_path):
        print(f"Loading custom split from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            custom_splits = json.load(f)
        
        train_names = custom_splits.get('train', [])
        dev_names = custom_splits.get('val', [])
        test_names = custom_splits.get('test', dev_names)
        
        if hasattr(dataset, 'lb') and dataset.lb is not None:
            class_to_id = {name: i for i, name in enumerate(dataset.lb.classes_)}
            train_class = [class_to_id[name] for name in train_names if name in class_to_id]
            dev_class = [class_to_id[name] for name in dev_names if name in class_to_id]
            test_class = [class_to_id[name] for name in test_names if name in class_to_id]
        else:
            train_num = class_split[dataset_name]["train"]
            dev_num = class_split[dataset_name]["dev"]
            test_num = class_split[dataset_name]["test"]
            class_list = list(range(dataset.num_classes))
            random.shuffle(class_list)
            train_class = class_list[:train_num]
            dev_class = class_list[train_num:train_num + dev_num]
            test_class = class_list[train_num + dev_num:]
    else:
        train_num = class_split[dataset_name]["train"]
        dev_num = class_split[dataset_name]["dev"]
        test_num = class_split[dataset_name]["test"]
        class_list = list(range(dataset.num_classes))
        random.shuffle(class_list)
        train_class = class_list[:train_num]
        dev_class = class_list[train_num:train_num + dev_num]
        test_class = class_list[train_num + dev_num:]

    print("train_num: {}; dev_num: {}; test_num: {}".format(len(train_class), len(dev_class), len(test_class)))

    id_by_class = {i: [] for i in range(dataset.num_classes)}
    for id, cla in enumerate(data.y.view(-1).tolist()):
        if cla in id_by_class:
            id_by_class[cla].append(id)

    train_idx = []
    for cla in train_class:
        train_idx.extend(id_by_class[cla])

    degree_inv = num_nodes / (data.edge_index.shape[1] * 2)

    return data, np.array(train_idx), id_by_class, train_class, dev_class, test_class, degree_inv

def test_task_generator(id_by_class, class_list, n_way, k_shot, m_query):
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected



