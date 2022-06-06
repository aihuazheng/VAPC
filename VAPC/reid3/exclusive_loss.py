from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np
from torch import nn
from torch.nn import init
class Exclusive(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, V):
        ctx.V = V;
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.V.t())
        return outputs
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = grad_outputs.mm(ctx.V) if ctx.needs_input_grad[0] else None
        for x, y in zip(inputs, targets):
            ctx.V[y] = F.normalize( (ctx.V[y] + x)/2, p=2, dim=0)#(0.7*self.V[y] + 0.3*x) / 2
        return grad_inputs, None,None


class ExLoss(nn.Module):
    def __init__(self, num_features, num_classes,label_to_images, t=1.0,
                 weight=None):
        super(ExLoss, self).__init__()
        self.num_features = num_features
        self.num_classes=num_classes
        self.t = t
        self.weight = weight
        self.register_buffer('V', torch.zeros(num_classes, num_features))
        
        self.label_to_images=label_to_images
    
    def forward(self, inputs, targets):
        outputs = Exclusive.apply(inputs, targets,self.V) * self.t
        """
        #print("inputs",x1.shape)
        #loss, ks = self.target_loss(outputs, targets,common_kcrnn)
        
        #targets,ks1 = self.adaptive_selection(inputs.detach().clone(), targets.detach().clone())
        #print("ks1",ks1)
        outputs = F.log_softmax(outputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        """
        loss = F.cross_entropy(outputs, targets, weight=self.weight)
        """
        
        
        #loss2 = F.cross_entropy(r_out, rot, weight=self.weight)
        
         
        loss2 = self.triloss(x1,targets)
        x2 = self.feat(x1)
        x2 = self.feat_bn(x2)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x2 = self.classifier_x2(x2)
        loss1 = F.cross_entropy(x2, targets, weight=self.weight)
        #loss = self.smooth_loss(inputs, targets)
        #loss=self.DMI_loss(outputs,targets)
        loss=loss1+loss2
        """ 
        return loss, outputs
    
    
    def target_loss(self, inputs, targets,common_kcrnn):
        targets_line, ks= self.adaptive_selection2(inputs.detach().clone(), targets.detach().clone(),common_kcrnn)
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets_line * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss, ks

    def adaptive_selection2(self, inputs, targets,common_kcrnn):
        ###minimize distances between all person images
        #targets_onehot = (inputs > self.lambda0/self.tau).float()
        
        targets_onehot=torch.zeros(inputs.shape[0],inputs.shape[1])
        for i in range(len(targets)):
            for j in common_kcrnn[targets[i].item()]:
                targets_onehot[i,j]=1
        ks = (targets_onehot.sum(dim=1)).float()
        ks1 = ks.cpu()
        ks_mask = (ks > 1).float()
        ks = ks * ks_mask + (1 - ks_mask) * 2
        ks = 1 / (ks * torch.log(ks))
        ks = (ks * ks_mask).view(-1,1)
        targets_onehot = targets_onehot# * ks
        targets_onehot=targets_onehot.cuda()
        ###maximize distances between similar person images
        targets = torch.unsqueeze(targets, 1)

        targets_onehot.scatter_(1, targets, float(1))
         
        return targets_onehot, ks1
        
    
    def triloss(self,inputs,targets):
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
        return loss
     
    def DMI_loss(self,output, target):
        outputs = F.softmax(output, dim=1)
        #print("outputs",outputs)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = torch.matmul(y_onehot, outputs)
        #print("torch.det(mat.float()",torch.det(mat.float()))
        if torch.isnan(torch.det(mat.float())) or torch.det(mat.float())==0.:
            loss=-1.0 * torch.log(torch.tensor(0.001))
        else:
            loss = -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)
        return loss
    
    
    def smooth_loss(self, inputs, targets):
        targets,ks1 = self.adaptive_selection(inputs.detach().clone(), targets.detach().clone())
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss
       
    def adaptive_selection(self, inputs, targets):
        ###minimize distances between all person images
        #targets_onehot = (inputs > self.lambda0/self.tau).float()
        #print("inputs.size(0)",inputs.size(0))
        targets_onehot = torch.zeros(inputs.size(0),self.num_classes).cuda()
        targets = torch.unsqueeze(targets, 1)
        #print("inputs",inputs.shape)
        #print("targets",targets)
        targets_onehot.scatter_(1, targets, float(1))
       
        ks=[]  
        #print(self.label_to_images)
        for i in range(len(targets)):
            ks.append(len(self.label_to_images[targets[i].cpu().item()]))
        #ks = len(self.label_to_images[targets]).float()#(targets_onehot.sum(dim=1)).float()
        #print("ks",ks)
        ks = torch.from_numpy(np.array(ks)).view(-1,1).float().cuda()
        
        ks_mask = (ks > 1).float()
        #print("ks1",ks_mask)
        ks1 = torch.log(ks)/(ks*1.5) 
        ks1 = ks1 * ks_mask + (1 - ks_mask)
        #print("ks2",ks1)
        
        
        #print("ks3",ks1)
        ks1 = ks1.view(-1,1)
        
        targets_onehot = targets_onehot * ks1
        
        ###maximize distances between similar person images
        #targets = torch.unsqueeze(targets, 1)
        
         
        return targets_onehot, ks1
       