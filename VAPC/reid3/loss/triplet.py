from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import norm

import numpy as np

class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=0, use_semi=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        self.K = num_instances
        self.use_semi = use_semi

    def forward(self, inputs, targets, epoch):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec

