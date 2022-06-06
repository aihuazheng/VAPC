from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import pdb
import os

import torch
import numpy as np

from .evaluation_metrics import map_cmc
from .utils.meters import AverageMeter
from torch.autograd import Variable
from .utils import to_torch
from .utils import to_numpy
import os.path as osp
from PIL import Image
from torchvision.transforms import functional as F
import pdb


def extract_cnn_feature(model, inputs, output_feature=None):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = to_torch(inputs)
    inputs = inputs.to(device)
    with torch.no_grad():
        fcs,outputs = model(inputs, output_feature)
    outputs = outputs.data.cpu()
    fcs=fcs.data.cpu()
    return fcs,outputs


def extract_features(model, data_loader, print_freq=50, output_feature=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    fcs = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids,_) in enumerate(data_loader):
        #print("fnames, pids", fnames, pids,_) 
        data_time.update(time.time() - end)

        _fcs,outputs = extract_cnn_feature(model, imgs, output_feature)
        for fname,fc,output, pid in zip(fnames,_fcs, outputs, pids):
            features[fname] = output
            fcs[fname] = fc
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features,fcs


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    print("1")

    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    #np.savetxt("Botto_features.txt",y)
    #print("feature save done")
    m, n = x.size(0), y.size(0)
    print("query_features", x.shape)
    print("gallery_features", y.shape)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = np.dot(x, y.t())
    print("dist", dist.shape)
    print("2")
    return dist


def evaluate_all(self, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)
    #np.savetxt("gallery_ids.txt",gallery_ids)
    # Evaluation
    mAP, all_cmc = map_cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, all_cmc[k - 1]))
    return all_cmc[0],mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, output_feature=None):
        query_features, _ = extract_features(self.model, query_loader, 50, output_feature)
        gallery_features, _ = extract_features(self.model, gallery_loader, 50, output_feature)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return evaluate_all(self, distmat, query=query, gallery=gallery)
