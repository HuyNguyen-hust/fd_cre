import os
from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder, Classifier
from .utils import Moment, dot_dist
from .utils import DecorrelateLossClass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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
                feature = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())
        
        features = np.concatenate(features)
        pca = PCA()
        pca.fit(features)
        
        return pca.singular_values_, pca.components_
    
    def train(self, args):
        def cal_cos(z1, z2):
            return np.abs(np.sum(z1 * z2, axis=1))
        
        for i in range(1):
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
            classifier = Classifier(encoder.output_size, num_classes).to(args.device)

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
                # train model
                # no memory. first train with current task
                self.train_simple_model(args, encoder, classifier, train_data_for_initial, args.step1_epochs)
                
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
    
    def get_optimizer(self, args, encoder, classifier):
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
        params.append({'params': classifier.parameters(), 'lr': 0.001})

        if args.optim == 'adam':
            pytorch_optim = optim.AdamW
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer
    
    def train_simple_model(self, args, encoder, classifier, training_data, epochs):

        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()
        classifier.train()

        ce_loss = nn.CrossEntropyLoss()
        if args.mode == 'fd':
            fd_loss = DecorrelateLossClass()
        optimizer = self.get_optimizer(args, encoder, classifier)
        def train_data(data_loader_, name = ""):
            losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                hidden = encoder.bert_forward(tokens)
                
                output = classifier(hidden)
                loss = ce_loss(output, labels)
                
                hidden = F.normalize(hidden, p=2, dim=-1)
                if args.mode == 'fd':
                    hidden = F.normalize(hidden, p=2, dim=-1)
                    loss += fd_loss(hidden, labels) * args.mu
                
                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                    
            print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_{}".format(epoch_i))