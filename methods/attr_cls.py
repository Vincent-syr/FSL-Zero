
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import euclidean_dist

import copy




class AttrCls(nn.Module):
    def __init__(self, model_func,  n_way, n_support, n_query, n_attr, params):
        super(AttrCls, self).__init__()
        
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query
        self.feature = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.n_attr = n_attr
        self.cls = nn.Sequential(
            nn.Linear(self.feat_dim, self.n_attr),
            nn.ReLU(),
            nn.Dropout(params.mlp_dropout),
            nn.Linear(self.n_attr, self.n_attr),
            nn.Sigmoid(),
        )
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_attr = nn.BCELoss()   # sigmoid + softmax + CE, output a scalar
        self.loss_cls = nn.CrossEntropyLoss()

    def forward(self, x):
        z = self.feature.forward(x)
        logits = self.cls(z)
           
        return z, logits
        


    def compute_score(self, z_all):
        z_all       = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]  # (n_way, n_query, feat_dim)
        z_query     = z_query.contiguous().view(-1, z_query.shape[-1])
        
        img_proto   = z_support.mean(1)   # (n_way, feat_dim)
        dists = euclidean_dist(z_query, img_proto)
        scores = -dists
        return scores
        

    def correct_attr(self, logits, y_attr):
        """[summary]

        Args:
            logits ([type]): (n_way * (support + query), n_attr)
            y_attr ([type]): (n_way * (support + query), n_attr)
        """
        # predict = copy.deepcopy(logits)
        predict = logits.clone().detach()  # new memory, not in computation graph
        predict[predict>0.5] = 1
        predict[predict<=0.5] = 0
        # predict = predict
        correct = (predict==y_attr).sum(dim=0)  #(N, 1)
        count = y_attr.shape[0]                  # (N)
        return correct, count


    def focal_loss(self, logits, y_attr):
        """[summary]
            gamma = 2, alpha不考虑
        Args:
            logits ([type]): (n_way * (support + query), n_attr)
            y_attr ([type]): (n_way * (support + query), n_attr)
        """
        gamma = 2
        log_pred_pos = torch.pow(1 - logits, gamma) * y_attr * torch.log(logits)
        log_pred_neg = torch.pow(logits, gamma) * (1-y_attr) * torch.log(1-logits)
        loss = (-1) * (log_pred_neg + log_pred_pos).mean()  # 对总维度求均值
        return loss
    
        
    def correct_cls(self, scores):  
        # scores: (n_way*n_query , n_way) 每个query 到每个 prototype的距离
        y_query = np.repeat(range(self.n_way ), self.n_query )  # (n_way*n_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)




        
        


    
    