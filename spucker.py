import sys
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",         type=str,   default="FB15k-237")
    parser.add_argument("--num_iterations",  type=int,   default=500)
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

entities = set.union(set(sub), set(obj))

entities  = sorted(list(entities))
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

class SliceDict:
    def __init__(self, data_idxs):
        
        slice_dict = defaultdict(list)
        for s, p, o in data_idxs:
            slice_dict[(s, p)].append(o)
    
        self._slice_dict = dict(slice_dict)
        self._keys = list(self._slice_dict.keys())
    
    def __len__(self):
        return len(self._slice_dict)
    
    def get_batch(self, idxs):
        
        xb = {"s" : [], "p" : []}
        yb = {"i" : [], "j" : []}
        
        s, p, o = [], [], []
        for offset, idx in enumerate(idxs):
            k = self._keys[idx]
            
            xb['s'].append(k[0])
            xb['p'].append(k[1])
            
            for oo in self._slice_dict[k]:
                yb['i'].append(offset)
                yb['j'].append(oo)
        
        return xb, yb


train_data_slices = SliceDict(train_data_idxs)
valid_data_slices = SliceDict(valid_data_idxs)
test_data_slices  = SliceDict(test_data_idxs)


# --
# Define model


def sparse_bce_with_logits(x, i, j):
    # !! Add label smoothing
    
    t1 = x.clamp(min=0).mean()
    t2 = - x[(i, j)].sum() / x.numel()
    t3 = torch.log(1 + torch.exp(-torch.abs(x))).mean()
    
    return t1 + t2 + t3


class Spucker(nn.Module):
    def __init__(self, num_entities, num_relations, ent_emb_dim, rel_emb_dim):
        super().__init__()
        
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
        
        input_dropout   = 0.3
        hidden_dropout1 = 0.4
        hidden_dropout2 = 0.5
        
        self.input_dropout   = torch.nn.Dropout(input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(hidden_dropout2)
        
        self.bn0 = torch.nn.BatchNorm1d(ent_emb_dim)
        self.bn1 = torch.nn.BatchNorm1d(ent_emb_dim)
    
    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        r  = self.R(r_idx)
        
        x  = self.input_dropout(self.bn0(e1))
        x  = x.view(-1, 1, e1.size(1))
        
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)
        
        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.hidden_dropout2(self.bn1(x))
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        
        return x


num_entities  = len(entities)
num_relations = len(relations)
num_epochs    = 500
lr            = 0.005
batch_size    = 1024
emb_dim       = 200
decay_rate    = 0.995

from torch.optim.lr_scheduler import ExponentialLR

model = Spucker(
    num_entities=num_entities,
    num_relations=num_relations,
    ent_emb_dim=emb_dim,
    rel_emb_dim=emb_dim,
)
print(model)
model = model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = ExponentialLR(opt, decay_rate)

for _ in range(num_epochs):
    idxs   = np.random.permutation(len(train_data_slices))
    chunks = np.array_split(idxs[:-57], idxs.shape[0] // batch_size)
    
    all_loss = []
    for chunk in tqdm(chunks):
        xb, yb = train_data_slices.get_batch(chunk)
        
        x_s = torch.LongTensor(xb['s']).cuda()
        x_p = torch.LongTensor(xb['p']).cuda()
        y_i = torch.LongTensor(yb['i']).cuda()
        y_j = torch.LongTensor(yb['j']).cuda()
        
        opt.zero_grad()
        
        out  = model(x_s, x_p)
        loss = sparse_bce_with_logits(out, y_i, y_j)
        
        loss.backward()
        opt.step()
        
        all_loss.append(loss.item())
    
    print('loss=%f' % np.mean(all_loss), file=sys.stderr)
    
    lr_scheduler.step()


