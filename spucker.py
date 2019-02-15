#!/usr/bin/env python

"""
    spucker.py
"""

import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",         type=str,   default="FB15k-237")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--lr",              type=float, default=0.0005)
    parser.add_argument("--dr",              type=float, default=1.0)
    parser.add_argument("--edim",            type=int,   default=200)
    parser.add_argument("--rdim",            type=int,   default=200)
    parser.add_argument("--input_dropout",   type=float, default=0.3)
    parser.add_argument("--hidden_dropout1", type=float, default=0.4)
    parser.add_argument("--hidden_dropout2", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    
    parser.add_argument("--seed", type=int, default=123)
    
    return parser.parse_args()

args = parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed + 111)
torch.cuda.manual_seed(args.seed + 222)

# --
# IO

def load_data(data_dir, data_type="train", reverse=False):
    with open("%s%s.txt" % (data_dir, data_type), "r") as f:
        data = f.read().strip().split("\n")
        data = [i.split() for i in data]
        if reverse:
            data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
    
    return data

data_dir = "data/FB15k-237/"

train_data = load_data(data_dir, "train", reverse=True)
valid_data = load_data(data_dir, "valid", reverse=True)
test_data  = load_data(data_dir, "test", reverse=True)

data = train_data + valid_data + test_data

sub, pred, obj = list(zip(*data))

entities  = sorted(list(set.union(set(sub), set(obj))))
relations = sorted(list(set(pred)))

entity_lookup   = dict(zip(entities, range(len(entities))))
relation_lookup = dict(zip(relations, range(len(relations))))

train_data_idxs = [(
    entity_lookup[x[0]], relation_lookup[x[1]], entity_lookup[x[2]],
) for x in train_data]

valid_data_idxs = [(
    entity_lookup[x[0]], relation_lookup[x[1]], entity_lookup[x[2]],
) for x in valid_data]

test_data_idxs = [(
    entity_lookup[x[0]], relation_lookup[x[1]], entity_lookup[x[2]],
) for x in test_data]

# --

class AdjList:
    def __init__(self, data_idxs):
        
        slice_dict = defaultdict(list)
        for s, p, o in data_idxs:
            slice_dict[(s, p)].append(o)
        
        self._slice_dict = dict(slice_dict)
        self._keys = list(self._slice_dict.keys())
    
    def __len__(self):
        return len(self._slice_dict)
    
    def get_batch_by_idx(self, idxs):
        keys = [self._keys[idx] for idx in idxs]
        return self.get_batch_by_keys(keys)
    
    def get_batch_by_keys(self, keys):
        xb = {"s" : [], "p" : []}
        yb = {"i" : [], "j" : []}
        
        s, p, o = [], [], []
        for offset, k in enumerate(keys):
            
            xb['s'].append(k[0])
            xb['p'].append(k[1])
            
            for oo in self._slice_dict[k]:
                yb['i'].append(offset)
                yb['j'].append(oo)
        
        return xb, yb


train_adjlist = AdjList(train_data_idxs)
valid_adjlist = AdjList(valid_data_idxs)
test_adjlist  = AdjList(test_data_idxs)
all_adjlist   = AdjList(train_data_idxs + valid_data_idxs + test_data_idxs)

# --
# Define model

def sparse_bce_with_logits(x, i, j):
    # !! Add support for label smoothing
    t1 = x.clamp(min=0).mean()
    t2 = - x[(i, j)].sum() / x.numel()
    t3 = torch.log(1 + torch.exp(-torch.abs(x))).mean()
    
    return t1 + t2 + t3


class Spucker(nn.Module):
    def __init__(self, num_entities, num_relations, ent_emb_dim, rel_emb_dim):
        super().__init__()
        
        self.ent_emb_dim = ent_emb_dim
        
        self.E = torch.nn.Embedding(num_entities, ent_emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(num_relations, rel_emb_dim, padding_idx=0)
        
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)
        
        self.W = torch.nn.Parameter(
            torch.tensor(
                np.random.uniform(-1, 1, (rel_emb_dim, ent_emb_dim, ent_emb_dim)), 
                dtype=torch.float,
                requires_grad=True
            )
        )
        
        input_dropout   = args.input_dropout
        hidden_dropout1 = args.hidden_dropout1
        hidden_dropout2 = args.hidden_dropout2
        
        self.input_dropout   = torch.nn.Dropout(input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(hidden_dropout2)
        
        self.bn0 = torch.nn.BatchNorm1d(ent_emb_dim)
        self.bn1 = torch.nn.BatchNorm1d(ent_emb_dim)
    
    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        r  = self.R(r_idx)
        
        print(e1.shape)
        
        x = self.input_dropout(self.bn0(e1))
        print(x.shape)
        x = x.unsqueeze(-2)
        print(x.shape)
        
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, self.ent_emb_dim, self.ent_emb_dim)
        W_mat = self.hidden_dropout1(W_mat)
        
        x = torch.bmm(x, W_mat)
        x = x.view(-1, self.ent_emb_dim)
        x = self.hidden_dropout2(self.bn1(x))
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        
        return x


num_entities  = len(entities)
num_relations = len(relations)
batch_size    = 128
# decay_rate    = 1.0

from torch.optim.lr_scheduler import ExponentialLR

model = Spucker(
    num_entities=num_entities,
    num_relations=num_relations,
    ent_emb_dim=args.edim,
    rel_emb_dim=args.rdim,
)
print(model)
model = model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)
# lr_scheduler = ExponentialLR(opt, decay_rate)

for epoch in range(args.epochs):
    
    # --
    # Train
    
    _ = model.train()
    
    idxs   = np.random.permutation(len(train_adjlist))
    chunks = np.array_split(idxs, idxs.shape[0] // batch_size)
    
    train_loss = []
    for chunk in tqdm(chunks):
        xb, yb = train_adjlist.get_batch_by_idx(chunk)
        x_s = torch.LongTensor(xb['s']).cuda()
        x_p = torch.LongTensor(xb['p']).cuda()
        y_i = torch.LongTensor(yb['i']).cuda()
        y_j = torch.LongTensor(yb['j']).cuda()
        
        opt.zero_grad()
        
        pred = model(x_s, x_p)
        loss = sparse_bce_with_logits(pred, y_i, y_j)
        
        loss.backward()
        opt.step()
        
        train_loss.append(loss.item())
    
    # lr_scheduler.step()
    
    # --
    # Eval
    
    _ = model.eval()
    
    idxs   = np.arange(len(valid_adjlist))
    chunks = np.array_split(idxs, idxs.shape[0] // batch_size)
    
    all_ranks = []
    for chunk in chunks:
        
        # Get validation batch
        xb, yb = valid_adjlist.get_batch_by_idx(chunk)
        x_s = torch.LongTensor(xb['s']).cuda()
        x_p = torch.LongTensor(xb['p']).cuda()
        y_i = torch.LongTensor(yb['i']).cuda()
        y_j = torch.LongTensor(yb['j']).cuda()
        
        pred = model(x_s, x_p)
        pred = torch.sigmoid(pred)
        target_pred = pred[(y_i, y_j)]
        
        # Get all true edges for keys
        _, ayb = all_adjlist.get_batch_by_keys(zip(xb['s'], xb['p']))
        ay_i   = torch.LongTensor(ayb['i']).cuda()
        ay_j   = torch.LongTensor(ayb['j']).cuda()
        
        pred[(ay_i, ay_j)] = 0
        
        ranks = (target_pred.view(-1, 1) < pred[y_i]).sum(dim=-1)
        all_ranks.append(ranks.cpu().numpy())
    
    all_ranks = np.hstack(all_ranks)
    
    print(json.dumps({
        "epoch"      : int(epoch),
        "train_loss" : float(np.mean(train_loss)),
        "h_at_10"    : float(np.mean(all_ranks < 10)),
        "h_at_03"    : float(np.mean(all_ranks < 3)),
        "h_at_01"    : float(np.mean(all_ranks < 1)),
    }))

