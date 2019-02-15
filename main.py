from load_data import Data
import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import pickle
from tqdm import tqdm
torch.backends.cudnn.deterministic = True 

    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        
    def get_data_idxs(self, data):
        return [(
            self.entity_idxs[data[i][0]],
            self.relation_idxs[data[i][1]], \
            self.entity_idxs[data[i][2]]
        ) for i in range(len(data))]
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets).cuda()
        return np.array(batch), targets

    
    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx  = torch.tensor(data_batch[:,0]).cuda()
            r_idx   = torch.tensor(data_batch[:,1]).cuda()
            e2_idx  = torch.tensor(data_batch[:,2]).cuda()
            
            predictions = model.forward(e1_idx, r_idx)
            
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            e2_idx    = e2_idx.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j])[0][0]
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
    
    def train_and_eval(self):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model = model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)
            
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0]).cuda()
                r_idx  = torch.tensor(data_batch[:,1]).cuda()
                
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
            if self.decay_rate:
                scheduler.step()
            losses.append(loss.item())
            print('iter=%d' % it)
            print('time=%f' % (time.time() - start_train))
            print('loss=%f' % np.mean(losses))
            model.eval()
            
            if not it % 10:
                with torch.no_grad():
                    print("Validation:")
                    self.evaluate(model, d.valid_data)
                    
                    print("Test:")
                    self.evaluate(model, d.test_data)


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

d = Data(data_dir="data/%s/" % args.dataset, reverse=True)
_ = Experiment(
    num_iterations=args.num_iterations,
    batch_size=args.batch_size,
    learning_rate=args.lr, 
    decay_rate=args.dr,
    ent_vec_dim=args.edim,
    rel_vec_dim=args.rdim,
    input_dropout=args.input_dropout,
    hidden_dropout1=args.hidden_dropout1,
    hidden_dropout2=args.hidden_dropout2,
    label_smoothing=args.label_smoothing
).train_and_eval()
