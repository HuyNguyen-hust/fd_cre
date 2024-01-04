import os, sys, logging 

from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder, Classifier
from .utils import DecorrelateLossClass, osdist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class Manager(object):
    def __init__(self, args):
        super().__init__()
        self.id2rel = None
        self.rel2id = None
    
    def select_data(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens=torch.stack([x.to(args.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())
        
        mem_set = []
        features = np.concatenate(features)
        num_clusters = min(args.num_protos, len(sample_set))
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = sample_set[sel_index]
            mem_set.append(instance)
        return mem_set
    
    def tsne(self, args, encoder, sample_set):
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
        params.append({'params': [p for p in classifier.parameters() if p.requires_grad], 'lr': args.cls_lr})

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

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, classifier, test_data, seen_relations, mode="cur"):
        data_loader = get_data_loader(args, test_data, batch_size=256)
        n = len(test_data)
        # gold = []
        # pred = []
        correct = 0

        encoder.eval()
        classifier.eval()

        with torch.no_grad():
            for _, batch_data in enumerate(tqdm(data_loader, desc="Evaluate {}".format(mode))):
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

                # normal classifier
                hidden = encoder.bert_forward(tokens)
                logits = classifier(hidden)[:, :args.num_classes]
                
                predicts = logits.max(dim=-1)[1]
                labels = labels

                correct += (predicts == labels).sum().item()

        return correct / n

    def train(self, args):
        # set training batch
        all_test_total = []
        for i in range(args.total_round):
            test_cur = []
            test_total = []
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
            memorized_samples = {}

            # load data and start computation
            
            history_relation = []
            # iterate through all tasks
            for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):

                print(current_relations)
                # Initial
                train_data_for_initial = []
                for relation in current_relations:
                    history_relation.append(relation)
                    train_data_for_initial += training_data[relation]
                # train model
                # no memory. first train with current task
                self.train_simple_model(args, encoder, classifier, train_data_for_initial, args.step1_epochs)

                # repaly
                if len(memorized_samples)>0:
                    # select current task sample
                    for relation in current_relations:
                        memorized_samples[relation] = self.select_data(args, encoder, training_data[relation])
                    
                    train_data_for_memory = []
                    for relation in history_relation:
                        train_data_for_memory += memorized_samples[relation]
                    self.train_simple_model(args, encoder, classifier, train_data_for_memory, args.step2_epochs)

                for relation in current_relations:
                    memorized_samples[relation] = self.select_data(args, encoder, training_data[relation])
                    

                test_data_1 = []
                for relation in current_relations:
                    test_data_1 += test_data[relation]

                test_data_2 = []
                for relation in seen_relations:
                    test_data_2 += historic_test_data[relation]

                cur_acc = self.evaluate_strict_model(args, encoder, classifier, test_data_1, seen_relations, mode='cur')
                total_acc = self.evaluate_strict_model(args, encoder, classifier, test_data_2, seen_relations, mode='total')

                logger.info(f'Restart Num {i+1}')
                logger.info(f'task--{steps + 1}:')
                logger.info(f'current test acc:{cur_acc}')
                logger.info(f'history test acc:{total_acc}')
                test_cur.append(cur_acc)
                test_total.append(total_acc)
                
                logger.info(test_cur)
                logger.info(test_total)

            all_test_total.append(test_total)
        
        all_test_total = np.array(all_test_total)
        logger.info(f'average test acc: {np.mean(all_test_total, axis=0)}')
