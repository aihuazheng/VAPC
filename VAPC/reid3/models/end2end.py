from __future__ import absolute_import

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
import math

from .resnet import *
from .resnet_ibn_a import *

__all__ = ["End2End_AvgPooling"]


class AvgPooling(nn.Module):
    def __init__(self, input_feature_size, embeding_fea_size=1024, dropout=0.5):
        super(self.__class__, self).__init__()

        # embeding
        self.embeding_fea_size = embeding_fea_size
        self.embeding = nn.Linear(input_feature_size, embeding_fea_size)
        
        self.embeding_bn = nn.BatchNorm1d(embeding_fea_size)
        init.kaiming_normal_(self.embeding.weight, mode='fan_out')
        init.constant_(self.embeding.bias, 0)
        init.constant_(self.embeding_bn.weight, 1)
        init.constant_(self.embeding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        net = inputs.mean(dim = 1)    #32, 2048
        eval_feas = F.normalize(net, p=2, dim=1)  #32, 2048
       
        net = self.embeding(net)
        net = self.embeding_bn(net)
        net = F.normalize(net, p=2, dim=1)
        net = self.drop(net)        
        return net, eval_feas

class End2End_AvgPooling(nn.Module):

    def __init__(self, dropout=0,  embeding_fea_size=1024, fixed_layer=True):
        super(self.__class__, self).__init__()
        self.CNN = resnet50(dropout=dropout, fixed_layer=fixed_layer)
        self.avg_pooling = AvgPooling(input_feature_size=2048, embeding_fea_size = embeding_fea_size, dropout=dropout)

    def forward(self, x,output_feature=None):
        #print(x.shape)
        #print("x.data.shape",x.data.shape)
        assert len(x.data.shape) == 4
        # reshape (batch, samples, ...) ==> (batch * samples, ...)
        oriShape = x.data.shape
        #x = x.view(-1, oriShape[1], oriShape[2], oriShape[3])
        
        # resnet encoding
        resnet_feature = self.CNN(x)
        #print("resnet_feature1",resnet_feature.shape)
        # reshape back into (batch, samples, ...)
        resnet_feature = resnet_feature.view(oriShape[0], 1, -1)
        #print("resnet_feature2",resnet_feature.shape)
        # avg pooling
        output = self.avg_pooling(resnet_feature)
        #print("output",output.shape)
        return output
class thermal_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18',gm_pool = 'on'):
        super(thermal_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.thermal = model_ft
        self.dropout = nn.Dropout(p=0.5)
        self.gm_pool = gm_pool
    def forward(self, x):
        #pdb.set_trace()
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
       # x_s1 = x
        #x_s1_avg = self.thermal.avgpool(x_s1)
       # x_s1_f = x_s1_avg.view(x_s1_avg.size(0), x_s1_avg.size(1))
        x = self.thermal.layer2(x)
       # x_s2 = x
        #x_s2_avg = self.thermal.avgpool(x_s2)
        #x_s2_f = x_s2_avg.view(x_s2_avg.size(0), x_s2_avg.size(1))
        x = self.thermal.layer3(x)
        x_s3= x
        if self.gm_pool == 'on':
            b, c, h, w = x_s3.shape
            x_s3_f = x_s3.view(b, c, -1)
            p = 3.0
            x_s3_f = (torch.mean(x_s3_f**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_s3_f = self.visible.avgpool(x_s3)
            x_s3_f = x_s3_f.view(x_s3_f.size(0), x_s3_f.size(1))
        x = self.thermal.layer4(x)
        x_s4 = x
        if self.gm_pool  == 'on':
            b, c, h, w = x_s4.shape
            x_s4_f = x_s4.view(b, c, -1)
            p = 3.0
            x_s4_f  = (torch.mean(x_s4_f**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_s4_f  = self.visible.avgpool(x_s4)
            x_s4_f  = x_s4_f.view(x_s4_f .size(0),x_s4_f.size(1))
        '''
        num_part = 6 # number of part
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part-1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        #x = self.thermal.avgpool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        # x = self.dropout(x)
        '''
        return x_s3,x_s3_f,x_s4,x_s4_f
