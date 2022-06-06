from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, fixed_layer=True):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.fixed_layer = fixed_layer
    def find_trans_mat(model_ce,num_classes,data_loader):
    # estimate each component of matrix T based on training with noisy labels
        print("estimating transition matrix...")
        output_ = torch.tensor([]).float().cuda()
    
        # collect all the outputs
        with torch.no_grad():
            for i, inputs in enumerate(data_loader):
                inputs, targets = self._parse_data(inputs)
                data = torch.autograd.Variable(inputs.cuda())
                outputs = self.model(data)
                outputs = torch.tensor(outputs).float().cuda()
                outputs = F.softmax(outputs, dim=1)
                output_ = torch.cat([output_, outputs], dim=0)
    
        # find argmax_{x^i} p(y = e^i | x^i) for i in C
        hard_instance_index = output_.argmax(dim=0)
    
        trans_mat_ = torch.tensor([]).float()
    
        # T_ij = p(y = e^j | x^i) for i in C j in C
        for i in range(num_classes):
            trans_mat_ = torch.cat([trans_mat_, output_[hard_instance_index[i]].cpu()], dim=0)
    
        trans_mat_ = trans_mat_.reshape(num_classes, num_classes)
    
        return trans_mat_
        
    def FW_loss(output, target, trans_mat):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
        outputs = F.softmax(output, dim=1)
        outputs = outputs @ trans_mat.cuda()
        outputs = torch.log(outputs)
        loss = CE_loss(outputs, target)
        return loss
        
    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        if self.fixed_layer:
            # The following code is used to keep the BN on the first three block fixed 
            fixed_bns = []
            for idx, (name, module) in enumerate(self.model.module.named_modules()):
                if name.find("layer3") != -1:
                    assert len(fixed_bns) == 22
                    break
                if name.find("bn") != -1:
                    fixed_bns.append(name)
                    module.eval() 

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)
    
            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            #torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.75)
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = Variable(imgs, requires_grad=False)
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs, _ = self.model(inputs)
        #print("outputs",outputs)
        loss, outputs = self.criterion(outputs, targets)
        #print("loss",loss.shape)
        #print("outputs",outputs.shape)
        prec, = accuracy(outputs.data, targets.data)
        prec = prec[0]

        return loss, prec
        

