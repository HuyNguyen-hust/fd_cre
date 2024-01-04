import os
from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder
from .utils import Moment, dot_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from .utils import osdist
from sklearn.decomposition import PCA
class Manager(object):
    def __init__(self, args):
        super().__init__()
        self.id2rel = None
        self.rel2id = None
    
    def make_dir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
    
    def get_svd(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens=torch.stack([x.to(args.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature, rp = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())
        
        features = np.concatenate(features)
        pca = PCA()
        pca.fit(features)
        
        return pca.singular_values_, pca.components_
    
    def train(self, args):
        def cal_cos(z1, z2):
            return np.abs(np.sum(z1 * z2, axis=1))
            
        for i in range(1):
            cur_round_path = os.path.join('spectral_visualization', args.dataname, args.mode)
            os.makedirs(cur_round_path, exist_ok=True)
            # set random seed
            random.seed(args.seed+i*100)

            # sampler setup
            sampler = data_sampler(args=args, seed=args.seed+i*100)
            self.id2rel = sampler.id2rel
            self.rel2id = sampler.rel2id
            
            # initialize memory and prototypes
            num_classes = len(sampler.id2rel)
            args.num_classes = num_classes
            
            # encoder setup
            encoder = Encoder(args=args).to(args.device)

            # load data and start computation
            # iterate through all tasks
            history_relation = []
            first_task = {}
            first_task_eigen_vec = {}
            for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
                # if steps > 1: break
                print(current_relations)
                # Initial
                train_data_for_initial = []
                for relation in current_relations:
                    history_relation.append(relation)
                    train_data_for_initial += training_data[relation]
                    if steps == 0:
                        first_task[relation] = training_data[relation]
                
                self.moment = Moment(args)
                self.moment.init_moment(args, encoder, train_data_for_initial, is_memory=False)
                self.train_simple_model(args, encoder, train_data_for_initial, args.step1_epochs, is_mem=False)
                
                if steps == 0:
                    for relation in current_relations:
                        eig_value, eig_vec = self.get_svd(args, encoder, training_data[relation])
                        with open(os.path.join(args.output_dir, f'eigval_{relation}'), 'w') as f:
                            f.write(str(eig_value.tolist()))
                        first_task_eigen_vec[relation] = eig_vec
                
                if steps == 9:
                    first_task_data = []
                    eig_cos = []
                    
                    for relation in first_task:
                        first_task_data += first_task[relation]
                        eig_value, eig_vec = self.get_svd(args, encoder, first_task[relation])
                        eig_cos.append(cal_cos(eig_vec, first_task_eigen_vec[relation]))
                        with open(os.path.join(args.output_dir, f'cos_{relation}'), 'w') as f:
                            f.write(str(eig_cos[-1].tolist()))
                
                    
    
    def get_optimizer(self, args, encoder):
        print('Use {} optim!'.format(args.optim))
        def set_param(module, lr, decay=0):
            parameters_to_optimize = list(module.named_parameters())
            no_decay = ['undecay']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return parameters_to_optimize
        params = set_param(encoder, args.learning_rate)

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer
    
    def train_simple_model(self, args, encoder, training_data, epochs, is_mem):

        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()

        optimizer = self.get_optimizer(args, encoder)
        def train_data(data_loader_, name = "", is_mem = False):
            losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                hidden, reps = encoder.bert_forward(tokens)
                loss = self.moment.loss(reps, labels, is_mem)
                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach())
                    
            print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_{}".format(epoch_i), is_mem=is_mem)
