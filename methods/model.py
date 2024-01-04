import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from .backbone import Bert_Encoder

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.encoder = Bert_Encoder(args)
        self.output_size = self.encoder.out_dim
        if args.mode == 'crl':
            dim_in = self.output_size
            self.head = nn.Sequential(
                    nn.Linear(dim_in, dim_in),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_in, args.feat_dim)
                )
    def bert_forward(self, x):
        out = self.encoder(x)
        if self.args.mode == 'crl':
            xx = self.head(out)
            xx = F.normalize(xx, p=2, dim=1)
            return out, xx
        else:
            return out
        
class Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_in, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
