#!/usr/bin/env python
# coding: utf-8
# author: Yuanchen Bei


import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as metrics
import tqdm
from datasets import DatasetBuilder
from NRCGI import NRCGI
import argparse
import random
import os
import time


############################ feed in args ####################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='ml-10m')
parser.add_argument('--model_name', default='nrcgi')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--early_epoch', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--embed_dim', type=int, default=10)
parser.add_argument('--save_path', default='chkpt')
parser.add_argument('--fig_path', default='figure')
parser.add_argument('--record_path', default='record')
parser.add_argument('--use_gpu', default=True, help='Whether to use CUDA')
parser.add_argument('--cuda_id', type=int, default=0, help='CUDA id')
parser.add_argument('--seed', type=int, default=2021)

args = parser.parse_args()


def set_seed(seed, cuda):
    '''
        Set the random seed.
    '''
    print('Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# get parameters
embed_dim = args.embed_dim
learning_rate = args.learning_rate
weight_decay = args.weight_decay
epoch = args.epoch
trials = args.early_epoch
batch_size = args.batch_size
device = torch.device("cuda:%d" % (args.cuda_id) if (torch.cuda.is_available() and args.use_gpu) else "cpu")
save_path = args.save_path
fig_path = args.fig_path
record_path = args.record_path
model_name = args.model_name
dataset_name = args.dataset_name

set_seed(args.seed, args.use_gpu)

# load dataset
with open(f'../data/{dataset_name}.pkl', 'rb') as f:
    train_set = np.array(pickle.load(f, encoding='latin1'))
    val_set = np.array(pickle.load(f, encoding='latin1'))
    test_set = np.array(pickle.load(f, encoding='latin1'))
    cate_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)  # iid to cate list
    u_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device) # uid to cate list
    i_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device) # iid to cate list
    user_count, item_count, cate_count = pickle.load(f)  # (user_count, item_count, cate_count)

field_dims = [user_count + 1, item_count + 1, cate_count + 1]  # idx-0 used for padding.
sample_size = 2

train_data = DatasetBuilder(data=train_set, user_count=user_count, item_count=item_count)
val_data = DatasetBuilder(data=val_set, user_count=user_count, item_count=item_count)
test_data = DatasetBuilder(data=test_set, user_count=user_count, item_count=item_count)

train_data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)
valid_data_loader = DataLoader(val_data, batch_size=batch_size, num_workers=8)
test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_auc = 0.0
        self.best_logloss = 1000000
        self.save_path = save_path

    def is_continuable(self, model, auc, log_loss):
        if auc > self.best_auc:
            self.best_logloss = log_loss
            self.best_auc = auc
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=20):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts, user_id_list = list(), list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            user_id_list.extend(fields[:, 0].tolist())
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    targets = np.array(targets)
    predicts = np.array(predicts)
    return metrics.roc_auc_score(targets, predicts), metrics.log_loss(targets, predicts)


print("use dataset: " + dataset_name)
# dataset's cluster pre-computed by Bi-Louvain algorithm.
if dataset_name == 'ml-10m':
    dataset_cluster = 5
    seq_len = 100
elif dataset_name == 'electronics':
    dataset_cluster = 93
    seq_len = 100
else:
    raise RuntimeError('Dataset does not exist!')


print("now model: " + model_name)
if model_name == 'nrcgi':
    model = NRCGI(field_dims=field_dims, cate_list = cate_list, u_cluster_list = u_cluster_list, i_cluster_list = i_cluster_list, cluster_num = dataset_cluster, seq_len = seq_len, sample_size = sample_size, embed_dim=embed_dim, batch_size=batch_size, device=device).to(device)
else:
    raise RuntimeError("Model does not exist!")

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_stopper = EarlyStopper(num_trials=trials, save_path=f'{save_path}/{model_name}_{dataset_name}.pt')

val_auc = []
val_logloss = []
for epoch_i in range(epoch):
    train(model, optimizer, train_data_loader, criterion, device)
    auc, log_losses = test(model, valid_data_loader, device)
    print('epoch:', epoch_i, 'validation: auc:', auc, 'validation logloss:', log_losses)
    val_auc.append(auc)
    val_logloss.append(log_losses)
    if not early_stopper.is_continuable(model, auc, log_losses):
        print(f'validation: best auc: {early_stopper.best_auc}, best logloss: {early_stopper.best_logloss}')
        break

model = torch.load(f'{save_path}/{model_name}_{dataset_name}.pt').to(device)
auc, log_losses = test(model, test_data_loader, device)

print(f'test auc: {auc}, test logloss: {log_losses}')
