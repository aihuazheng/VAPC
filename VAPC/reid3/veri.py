import torch
from torch import nn
from reid import models
from reid3.trainers import Trainer
from reid3.evaluators import extract_features, Evaluator
#from reid3.demo import extract_features, Evaluator
#from reid.dist_metric import DistanceMetric
import numpy as np
from collections import OrderedDict
import os.path as osp
import pickle
import copy, sys
from reid.utils.serialization import load_checkpoint
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
import random
import pickle as pkl
from reid3.exclusive_loss import ExLoss
import json
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid3.rerank import re_ranking
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
import time

class Bottom_up():
    def __init__(self, model_name, batch_size, num_classes,view_label,dataset, u_data, save_path, embeding_fea_size=1024,
                 dropout=0.5, max_frames=900, initial_steps=100, step_size=16):
        #32
        self.model_name = model_name
        self.num_classes = num_classes
        #self.data_dir = dataset.images_dir
        #self.is_video = dataset.is_video
        self.save_path = save_path
        self.view=[7637,9877,1458,13629,5128]#00#每个视角的图像数
        #self.view=[10000,10000,10000,10000,10000]#08#加载wild数据集时取消该注释

        self.index_dict = view_label
        self.dataset = dataset
        self.u_data = u_data
        self.u_label = np.array([label for _, label, _, _ in u_data])#加载真实标签计算聚类质量
        #self.u_label = np.loadtxt("/DATA/sunxia/gd_lable_wild.txt")#np.array([label for _, label, _, _ in u_data])#加载wild数据集时取消该注释
        #self.c_label = np.loadtxt("/DATA/sunxia/color.txt")
        self.lamda=0.98
        self.dataloader_params = {}
        self.dataloader_params['height'] = 384#224#加载wild数据集时取消该注释
        self.dataloader_params['width'] =384#224#加载wild数据集时取消该注释
        self.dataloader_params['batch_size'] = 128
        self.dataloader_params['workers'] = 6
 
        self.batch_size = 32
        self.data_height = 384
        self.data_width = 384
        self.data_workers = 6

        self.initial_steps = initial_steps
        self.step_size = step_size

        # batch size for eval mode. Default is 1.
        self.dropout = dropout
        self.max_frames = max_frames
        self.embeding_fea_size = embeding_fea_size
        self.FLAG=1
        self.eval_bs = 64
        self.fixed_layer = False
        self.frames_per_video = 1
        self.later_steps = 2
        self.krnn_set=OrderedDict()
        model = models.create(self.model_name, dropout=self.dropout,
                              embeding_fea_size=self.embeding_fea_size, fixed_layer=self.fixed_layer)
        #self.model = nn.DataParallel(model).cuda()
        #checkpoint = load_checkpoint("/home/sunxia/Bottom/logs/tiaoxuanzs.pth.tar")
        #checkpoint = load_checkpoint("/home/sunxia/Bottom/logs/checkpointbaseline.pth.tar")#"/DATA/sunxia/logs/ori_bottom_baseline.pth.tar"
        #checkpoint = load_checkpoint("/DATA/sunxia/t_snelog/tiaoxuanzs.pth.tar")
        checkpoint = load_checkpoint("/home/sunxia/Bottom-up-Clustering-Person-Re-identification/logs/bestcheckpoint038432.pth.tar")
        #这里为了省略训练时间，直接加载辨识阶段后的模型，如没有模型，先训练20个epoch辨识阶段，就是先注销这里再在run.py里的训练从(0,50)开始
        #checkpoint = load_checkpoint("/DATA/sunxia/logs/OIMLOSSall52.pth.tar") #224#加载wild数据集时取消该注释
        model.load_state_dict(checkpoint['state_dict'])
        self.model = nn.DataParallel(model).cuda()
        self.criterion = ExLoss(self.embeding_fea_size, self.num_classes,label_to_images=None, t=10).cuda() 
        self.rank1=0.
        self.bestrank1=0.
        self.mAP=0.
        self.bestmAP=0.
        self.hold=0.
    def get_dataloader(self, dataset, training=False):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        if training:
            transformer = T.Compose([
                T.RandomSizedRectCrop(self.data_height, self.data_width),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.batch_size
        else:
            transformer = T.Compose([
                T.RectScale(self.data_height, self.data_width),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.eval_bs
        
        #self.dataset.target_images_dir为目标数据集文件夹
        data_loader = DataLoader(
            Preprocessor(dataset, root=osp.join(self.dataset.target_images_dir,self.dataset.target_train_path),
                         transform=transformer),
            batch_size=batch_size, num_workers=self.data_workers,
            shuffle=training, pin_memory=True, drop_last=training)

        current_status = "Training" if training else "Clustring"
        print("Create dataloader for {} with batch_size {}".format(current_status, batch_size))
        return data_loader

    def train(self, train_data, step, loss, dropout=0.5):
        # adjust training epochs and learning rate
        epochs = self.initial_steps if step==0 else self.later_steps
        init_lr = 0.1 if step==0 else 0.001
        """
        if step<11:
             init_lr=0.001
        else:
             init_lr=0.0001
        """ 
        #init_lr=0.0001
        step_size = self.step_size if step==0 else sys.maxsize
        print("sys.maxsize",sys.maxsize)
        """ create model and dataloader """
        dataloader = self.get_dataloader(train_data,training=True)
        
        # the base parameters for the backbone (e.g. ResNet50)
        base_param_ids = set(map(id, self.model.module.CNN.base.parameters()))

        # we fixed the first three blocks to save GPU memory
        base_params_need_for_grad = filter(lambda p: p.requires_grad, self.model.module.CNN.base.parameters())

        # params of the new layers
        new_params = [p for p in self.model.parameters() if id(p) not in base_param_ids]

        # set the learning rate for backbone to be 0.1 times
        param_groups = [
            {'params': base_params_need_for_grad, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

        optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        # change the learning rate by step
        def adjust_lr(epoch, step_size):
            lr = init_lr / (10 ** (epoch // step_size))
            print("lr",lr)
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        """ main training process """
        trainer = Trainer(self.model, self.criterion, fixed_layer=self.fixed_layer)
        for epoch in range(epochs):#epochs
            adjust_lr(epoch, step_size)
            trainer.train(epoch, dataloader, optimizer, print_freq=max(5, len(dataloader) // 30 * 10))
            
            """
            if epoch %2==0:
                self.evaluate(self.dataset.query,self.dataset.gallery,'pool5')
                save_checkpoint({
                        'state_dict': self.model.module.state_dict(),
                        'step': step,
                        }, fpath=osp.join('logs', 'VEHILCEID256UDAP.pth.tar'))
            """
            #
            """
            if step==0: 
                #self.evaluate(self.dataset.query,self.dataset.gallery,'pool5')
                if epoch==32:
                    save_checkpoint({
                    'state_dict': self.model.module.state_dict(),
                    'step': step,
                    }, fpath=osp.join('logs', 'bestcheckpoint038432.pth.tar'))
            """
    def get_feature(self, dataset):
        dataloader = self.get_dataloader(dataset, training=False)
        features, fcs = extract_features(self.model, dataloader)
        #print("features_name",list(features.keys()))
        
        features = np.array([logit.numpy() for logit in features.values()])
        #print("load_features")
        
        #print("features",features)
        #fcs = np.array([logit.numpy() for logit in fcs.values()])
        #print("fcs",fcs)
        #np.savetxt("/DATA/sunxia/new/viewnonoise_step2.txt",list(features))
        return features#, fcs

    def update_memory(self, weight):
        self.criterion.weight = torch.from_numpy(weight).cuda()

    def evaluate(self,query, gallery, output_feature):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        test_transformer = T.Compose([
            T.Resize((self.data_height, self.data_width), interpolation=3),
            T.ToTensor(),
            normalizer,
        ])
        
        #osp.join(self.dataset.target_images_dir, self.dataset.query_path)
        query_loader = DataLoader(
            Preprocessor(query,root=osp.join(self.dataset.target_images_dir, self.dataset.query_path), transform=test_transformer),
            batch_size=64, num_workers=self.data_workers,
            shuffle=False, pin_memory=True)
        gallery_loader = DataLoader(
            Preprocessor(gallery,root=osp.join(self.dataset.target_images_dir, self.dataset.gallery_path), transform=test_transformer),
            batch_size=64, num_workers=self.data_workers,
            shuffle=False, pin_memory=True)
        evaluator = Evaluator(self.model)
        self.rank1,self.mAP=evaluator.evaluate(query_loader, gallery_loader, query, gallery, output_feature)
        #加载wild数据集时取消该注释
        """
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        test_transformer = T.Compose([
            T.Resize((self.data_height, self.data_width), interpolation=3),
            T.ToTensor(),
            normalizer,
        ])
        query_loader = DataLoader(
            Preprocessor(query,root=osp.join("/DATA/sunxia/VERI-WILD", self.dataset.query_path), transform=test_transformer),
            batch_size=64, num_workers=self.data_workers,
            shuffle=False, pin_memory=True)
        gallery_loader = DataLoader(
            Preprocessor(gallery,root=osp.join("/DATA/sunxia/VERI-WILD",self.dataset.gallery_path), transform=test_transformer),
            batch_size=128, num_workers=self.data_workers,
            shuffle=False, pin_memory=True)
        evaluator = Evaluator(self.model)
        self.rank1,self.mAP=evaluator.evaluate(query_loader, gallery_loader, query, gallery, output_feature)
        """
            
    def calculate_distance(self,u_feas):
        # calculate distance between features
        print("calculate_distance")
        x = torch.from_numpy(u_feas)
        y = x
        m = len(u_feas)
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())
        return dists

    # def select_merge_data(self, u_feas, nums_to_merge, label, label_to_images,  ratio_n,  dists):
    #     #calculate final distance (feature distance + diversity regularization)
    #     tri = np.tri(len(u_feas), dtype=np.float32)
    #     tri = tri * np.iinfo(np.int32).max
    #     tri = tri.astype('float32')
    #     tri = torch.from_numpy(tri)
    #     dists = dists + tri
    #     for idx in range(len(u_feas)):
    #         for j in range(idx + 1, len(u_feas)):
    #             if label[idx] == label[j]:
    #                 dists[idx, j] = np.iinfo(np.int32).max
    #             else:
    #                 dists[idx][j] =  dists[idx][j] + \
    #                                 + ratio_n * ((len(label_to_images[label[idx]])) + (len(label_to_images[label[j]])))
    #     dists = dists.numpy()
    #     ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
    #     idx1 = ind[0]
    #     idx2 = ind[1]
    #     return idx1, idx2


    def select_merge_data(self,dists):# u_feas, label, label_to_images,  ratio_n,  
        
        """
        print("select_merge_data")
        for idx in range(len(u_feas)):
            for j in range(idx + 1, len(u_feas)):
                if label[idx] == label[j]:
                    dists[idx, j] = 100000
        """
        #dists = dists.numpy()
        print(dists.dtype)
        #print("np.argsort(dists, axis=None).astype(np.int32)",np.argsort(dists, axis=None).astype(np.int32).shape)
        #print("np.argsort(dists, axis=None).astype(np.int32)",np.argsort(dists, axis=None).astype(np.int32)[0:715457428].shape)
        ind = np.unravel_index(np.argsort(dists, axis=None).astype(np.int32)[0:715457428], dists.shape)#[0:715457428]
        #print("ind",ind.shape)
        print("select_merge_data_done")
        idx1 = ind[0]
        idx2 = ind[1]
        
        return idx1, idx2

    def judgeview(self,num):
        l1=self.view[0]
        l2=l1+self.view[1]
        l3=l2+self.view[2]
        l4=l3+self.view[3]
        l5=l4+self.view[4]
        if num>=0 and num <l1:
            return 1
        if num>=l1 and num <l2:
            return 2
        if num>=l2 and num <l3:
            return 3
        if num>=l3 and num <l4:
            return 4
        if num>=l4 and num <l5:
            return 5

    def generate_new_train_label(self,step, idx1, idx2,u_feas,label,num_to_merge,dists,flag,view_to_images=None,rec_noise=None):
        print("type",type(label[0]))
        
        correct = 0
        error = 0
        cnt=0
        s_s=0
        s_d=0
        d_s=0
        d_d=0
        ori_label=label
        errorlable=[] 
        num_before_merge = len(np.unique(np.array(label)))
        # merge clusters with minimum dissimilarity  
        yuzhi=1200#设置第几对样本作为相似性阈值计算
        hold_set=[]
       
        #print(hold_set)        
        # merge clusters with minimum dissimilarity  
        print("vote")
        c = {}
        #hold=0.003
        if step==1:
            if dists[idx1[yuzhi],idx2[yuzhi]]!=100000:      
                self.hold=dists[idx1[yuzhi],idx2[yuzhi]]
            else:
                self.hold=dists[idx2[yuzhi],idx1[yuzhi]]
        ced=[]
        #self.hold=0.001
        print(self.hold,"self.hold")
        hit_1=0
        c_krnn=0
        b=[]
        d=[]
        for i in range(len(idx1)):
            label1 = label[idx1[i]]
            label2 = label[idx2[i]]
            if label1==-1 or label2 ==-1:
                continue
            if label1 in ced or label2 in ced:
                continue
            else:
                b.append((label1,label2))
                d.append((idx1[i],idx2[i]))
                """
                if (label1 in rec_noise and label2 not in rec_noise) or (label2 in rec_noise and label1 not in rec_noise):
                    if label1 in rec_noise:
                        ced.append(label1)
                    else:
                        ced.append(label2)
                    continue
                """    
                ced.append(label1)
                ced.append(label2)
            #if idx2[i] not in krnn_set[idx1[i]] or idx1[i] not in krnn_set[idx2[i]]:
            #    c_krnn+=1
            #    continue
           
                
            if (label1,label2) in c or (label2,label1) in c:
                
                if label1<label2:
                    c[(label1,label2)]+=1
                else:
                    c[(label2,label1)]+=1
            else:
                if label1<label2:
                    c[(label1,label2)]=1
                else:
                    c[(label2,label1)]=1
            #if c_krnn>=200:
            #        break
            #print("dists[idx1[i],idx2[i]]",dists[idx1[i],idx2[i]])
            if dists[idx1[i],idx2[i]]!=100000:      
                if dists[idx1[i],idx2[i]]>=self.hold:
                    break
            else:
                if dists[idx2[i],idx1[i]]>=self.hold:
                    break
            #if len(c)>=800:
            #    break
        print("hit_1",hit_1)    
        a = sorted(c.items(), key=lambda x: x[1], reverse=False)
        ca_min=[]
        ca_max=[]
        ca_margin=[]
        ea_min=[]
        ea_max=[]
        ea_margin=[]
        cnt_ced={}
        not_rb=[]
        sim_label=[]
        h=0.001
        print("h",h)
        for i in range(len(b)):
            label1 = b[i][0]
            label2 = b[i][1]
            idx1=d[i][0]
            idx2=d[i][1]
            if label1 < label2:
                label = [label1 if x == label2 else x for x in label]
            else:
                label = [label2 if x == label1 else x for x in label]
        
        #print("np.mean(ca_min)",np.mean(ca_min),"np.mean(ea_min)",np.mean(ea_min))
        #print(max(ca_margin),min(ca_margin),'-',max(ea_margin),min(ea_margin))
        #print("s_s",s_s,s_s/cnt,"s_d",s_d,s_d/cnt,"d_s",d_s,d_s/cnt,"d_d",d_d,d_d/cnt)
        #print("cnt",cnt,correct,"correc rate",correct/cnt,error,"error rate ",error/cnt)
           
        
       
        print("set new label to the new training data")
        #np.savetxt("compar1label.txt",label)
        
        #label=label-min(label)
        """
        label_to_images = {}
        #view_to_images = {} 
        for idx, l in enumerate(label):
            label_to_images[l] = label_to_images.get(l, []) + [idx]
        cont1=[]
        for i in label_to_images:
            if len(label_to_images[i])==1:
                cont1.append(label_to_images[i][0])
        """
        j=0
        unique_label = np.sort(np.unique(np.array(label)))
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            if label_now ==-1:
                continue
            label = [j if x == label_now else x for x in label]
            j+=1        
        #np.savetxt("compar1labe2.txt",label)
        #label=np.array(label)-min(label)
        num_after_merge = len(np.unique(np.array(label)))
        print("num of label before merge: ", num_before_merge, " after_merge: ", num_after_merge, " sub: ",
              num_before_merge - num_after_merge)
        return label
        
    def generate_new_train_data(self,label):
        label = [int(i) for i in label]
        new_train_data = []
        

        print("minlabel",min(label),max(label))
        for idx, data in enumerate(self.u_data):
            if label[idx]==-1:
                continue
            new_data = copy.deepcopy(data)
            new_data[1] = label[idx]
            new_train_data.append(new_data)
        label_to_images = {}
        print("new_train_data",len(new_train_data))
        for idx, l in enumerate(label):
            label_to_images[l] = label_to_images.get(l, []) + [idx]
        return new_train_data,label_to_images
        
    def generate_average_feature(self, labels):
        #extract feature/classifier
        u_feas = self.get_feature(self.u_data)
         
        #images of the same cluster
        
        #for key , value in label_to_images.items():
        #    view_to_images[key] = [self.index_dict[i] for i in value]
        
        """
        print("calculate average feature/classifier of a cluster")
        feature_avg = np.zeros((len(label_to_images), len(u_feas[0])))
        #fc_avg = np.zeros((len(label_to_images), len(fcs[0])))
        for l in label_to_images:
            feas = u_feas[label_to_images[l]]
            feature_avg[l] = np.mean(feas, axis=0)
            #fc_avg[l] = np.mean(fcs[label_to_images[l]], axis=0)
        """

        #print("feature_avg",feature_avg.shape)
        return u_feas
        
    def calculate_samv_distance(self,u_feas):
        # calculate distance between features
        x = u_feas
        y = x
        m = len(u_feas)
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())
        return dists
    
    def findsame(self,a):
        b=[]
        f=[]
        for i in a:
            if i ==-1:
                continue
            c=[]
            for item in enumerate(a):
                if item[1] ==i:
                    c.append(item[0])
            b.append(c)
        e=[] 
        for i in b:
            if i not in e:
                e.append(i)
        return e      
            
    def get_new_train_data(self, step, labels, nums_to_merge, size_penalty):
        
        timererank = time.strftime("%m_%d_%H:%M:%S")
        print("model traing end ",timererank)
        if self.rank1>self.bestrank1 and step != 1:
            print("save best checkpoint")
            self.bestrank1=self.rank1
            """
            save_checkpoint({
            'state_dict': self.model.module.state_dict(),
            'step': 1,
            }, fpath=osp.join('logs', '/DATA/sunxia/reid/temp_result/notianxiaozs'+str(step)+'.pth.tar'))
            #np.savetxt("notianxiaozs.txt",labels)   
            """
        
        print("self.bestrank1",self.bestrank1)
        
        print("self.bestrank1",self.bestrank1)
        flag = 1
        timererank = time.strftime("%m_%d_%H:%M:%S")
        print("feature exact ",timererank)
        num_before_merge = len(np.unique(np.array(labels)))
        u_feas = self.generate_average_feature(labels)
      
        u_feas = torch.from_numpy(u_feas)
        #np.savetxt("VAPC_feature.txt",u_feas)
        
        V = torch.split(u_feas, self.view, 0)
        timererank = time.strftime("%m_%d_%H:%M:%S")
        print("feature exact end",timererank)
        all=[]
        cluster={}
        num_before_merge = len(labels)
        labels=[]
        new_view=[0,0,0,0,0]
        m=0
        print("no k-rec ecoding")
        ###########################################################################
        #第一阶段聚类
        ##加载wild数据集时相同视角聚类为4次
        #第一阶段聚类
        ##加载wild数据集时相同视角聚类为4次
        #第一阶段聚类
        ##加载wild数据集时相同视角聚类为4次
        ###########################################################################
        for v in range(0,5):
            time1 = time.strftime("%m_%d_%H:%M:%S")
            print("dbscan begin time",time1)
            print("rerank begin time",time1)
            cnt1=0
            rec_noise=[]
            rec_noise_index=[]
            s_noise=[]
            s_cluster=[]
            #rerank_dist,oridist = re_ranking(0, V[v], step = 2,flag=0)
            
            print("no re rank")
            timererank = time.strftime("%m_%d_%H:%M:%S")
            print("timererank end ",timererank)
            ###########################################################################
            #k互近邻编码
            oridist = re_ranking(0, V[v], step = 2,flag=1)
            #oridist = np.abs(oridist)

            """
            clusterer = hdbscan.HDBSCAN(min_samples=10, metric='precomputed',n_jobs=8)  # min_cluster_size=2,
            label = clusterer.fit_predict(rerank_dist.astype(np.double))
            """ 
            print("rerank_dist",oridist.shape)
            tri_mat = np.triu(oridist, 1) # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            top_num = np.round(0.5e-3*tri_mat.size).astype(int) #1.6
            print("top_num",top_num)
            eps = tri_mat[:top_num].mean()

            print('eps ',eps)
            cluster = DBSCAN(eps=eps,min_samples=4,metric='precomputed', n_jobs=8)  #8s2 4
            print('Clustering and labeling...')
            label = cluster.fit_predict(oridist)
            timedbscan = time.strftime("%m_%d_%H:%M:%S")
            print("dbscan end ",timedbscan)
            #print("label",label)
            un_c=0
            for j in label:
                if j==-1:
                    un_c+=1
            print("number of -1",un_c)
            new_view[v]=len(set(label))-1
            
            #np.savetxt("dblabel.txt",label)
            """
            if min(label)<0:
                tmp_max=max(label)
                print("tmp_max",tmp_max)
                for i in range(len(label)):
                    if label[i]<0:
                        cnt1+=1
                        label[i]+=tmp_max+1+m+cnt1
                        rec_noise.append(label[i])
                        rec_noise_index.append(i)
                    else:
                        label[i]+=m
            else:
                label=np.array(label)
                label+=m  
            m+=len(np.unique(np.array(label)))
            """
            """
            #np.savetxt("dblabel2.txt",label)
            if min(label)<0:
                tmp_max=max(label)
                print("tmp_max",tmp_max)
                for i in range(len(label)):
                    if label[i]>=0:
                        label[i]+=m
            else:
                label=np.array(label)
                label+=m
            if min(label)<0: 
                m+=len(np.unique(np.array(label)))
                m-=1
            else:
                m+=len(np.unique(np.array(label)))
            
            """
            ###########################################################################
            #噪声选择
            print("noise selection")
            if v !=-2:
                timenoise0 = time.strftime("%m_%d_%H:%M:%S")
                print("noise calucate begin",timenoise0)
                initial_rank = np.argsort(oridist).astype(np.int32)
                #oridist.add_(torch.tril(100000 * torch.ones(len(V[v]), len(V[v]))))
                #oridist=oridist.numpy()
                #ind = np.unravel_index(initial_rank, oridist.shape)#[0:715457428]
                #print("ind",ind.shape)
                #print("select_merge_data_done")
                #idx1 = ind[0]
                #idx2 = ind[1]
                #print("rec_noise_index",rec_noise_index,"rec_noise",rec_noise)
                tmp_rec=[]
                tmp=0
                rec_noi=[]
                tmp_dist=[]
                clus=[]    
                for k in rec_noise_index:
                    #print("label[k]",label[k],k)
                    #print("initial_rank",initial_rank[k]) 
                    tmp_dist.append(oridist[k,initial_rank[k,1]])
                    clus.append((k,initial_rank[k,1]))
                so_index=np.argsort(tmp_dist)
                #print("clus",clus)
                for i in so_index:
                    k=clus[i][0]
                    #print("clus[i][0",clus[i][0],clus[i][1])   
                    label1=label[k]
                    if k in tmp_rec:
                        continue
                    #s_noise.append((label1,label[initial_rank[rec_noise_index[k],1]])) 
                    #print("k",k,initial_rank[k,1:10])
                    tmp=initial_rank[k,1]
                    #print(initial_rank[tmp,1:2])
                    if tmp in rec_noise_index:
                        if k in initial_rank[tmp,1:2]:
                            s_noise.append((k,tmp))
                            rec_noi.append(k)
                            rec_noi.append(tmp)
                        else:
                            tmp_rec.append(k)        
                            
                    else:
                        if k in rec_noi:
                            continue
                        if k in initial_rank[tmp,1:2]:
                            s_cluster.append((tmp,k))
                            #rec_noise_index.remove(k)

                        tmp_rec.append(k)
                    
                    #print("s_noise",s_noise)
                    #print("s_cluster",s_cluster)
                print("s_noise",len(s_noise))
                print("s_cluster",len(s_cluster))
                  
                
                
                for i in  range(len(s_cluster)):
                    label1 = label[s_cluster[i][0]]
                    label2 = label[s_cluster[i][1]]
                    label = [label1 if x == label2 else x for x in label]
                for i in  range(len(s_noise)):
                    label1 = label[s_noise[i][0]]
                    label2 = label[s_noise[i][1]]
                    if label1 < label2:
                        label = [label1 if x == label2 else x for x in label]
                    else:
                        label = [label2 if x == label1 else x for x in label]
            timenoise1 = time.strftime("%m_%d_%H:%M:%S")
            print("noise calucate end ",timenoise1)      
            labels=np.concatenate((labels,np.array(label)))#handel differnnt case
            #np.savetxt("/DATA/sunxia/reid/temp_result/labelswons"+str(step)+".txt",labels)
            print("label",np.array(label),len(np.unique(np.array(label))))
    
         ###########################################################
         #第二阶段不同视角聚类
         #第二阶段不同视角聚类
         #第二阶段不同视角聚类
         #第二阶段不同视角聚类
         ###########################################################
        if flag!=-1:
            timedists0 = time.strftime("%m_%d_%H:%M:%S")
            print("global dists calucate",timedists0)
            print("clustring in the different view")     
            flag = 1
            print("different view re_rank")
            dists= re_ranking(0, u_feas, step=2,flag=1)
            timedists1 = time.strftime("%m_%d_%H:%M:%S")
            print("global dists calucate end",timedists1)
            #tmp_dists=copy.deepcopy(dists)
            #dists = self.calculate_samv_distance(u_feas)
            #np.savetxt("seconde_dists.txt",dists)
            #wild训练注释该部分进行替换
            """
            l1=self.view[0]
            l2=l1+self.view[1]
            l3=l2+self.view[2]
            l4=l3+self.view[3]
            l5=l4+self.view[4]
            dists.add_(torch.tril(100000 * torch.ones(len(u_feas), len(u_feas))))
            #dists += 0.005 * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))
            print("dists.shape",dists.shape)
            l=[l1,l2,l3,l4,l5]             
    
            dists[0:l1,0:l1]=1000000
            dists[l1:l2,l1:l2]=1000000
            dists[l2:l3,l2:l3]=1000000
            dists[l3:l4,l3:l4]=1000000
            dists[l4:l5,l4:l5]=1000000
            """
            l1=self.view[0]
            l2=l1+self.view[1]
            l3=l2+self.view[2]
            l4=l3+self.view[3]
            l5=l4+self.view[4]
            print(l1,l2,l3,l4,l5)
            dists[0:l5,0:l1]=100000
            #dists[0:7637,9877+7637:37729]=100000
            
            dists[l1:l5,l1:l2]=100000
            #dists[7637:7637+9877,7637+9877+1458:37729]=100000
            
            #dists[7637+9877:37729,7637+9877:7637+9877+1458]=100000
            dists[l2:l5,l2:l3]=100000
            #dists[7637+9877:7637+9877+1458,7637+9877+1458+13629:37729]=100000
            
            dists[l3:l5,l3:l4]=100000
            dists[l4:l5,l4:l5]=100000
            
            """
            #dists.add_(torch.tril(100000 * torch.ones(m, n)))
            for idx in range(len(u_feas)):
                for j in range(idx + 1, len(u_feas)):
                    if labels[idx] == labels[j]:
                        dists[idx, j] = 100000
            """
            #cnt = torch.FloatTensor([len(view_to_images[labels[idx]]) for idx in range(len(u_feas))])
            #np.savetxt("cnt.txt",cnt.numpy())
            #dists += 0.0003 * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))
            #del cnt
          
            nums_to_merge =  len(np.unique(np.array(labels)))* 0.3#merge_perce[i]
            idx1, idx2 = self.select_merge_data(dists)
            #del tmp_dists
            labels = self.generate_new_train_label(step,idx1, idx2,u_feas,labels,nums_to_merge,dists,flag)
        
        
        #print("metrics.adjusted_rand_score(labels_true, labels_pred)",metrics.adjusted_rand_score(self.u_label, labels))
        #print("metrics.adjusted_mutual_info_score(labels_true, labels_pred)",metrics.adjusted_mutual_info_score(self.u_label, labels))
        #print("v_measure_score",metrics.v_measure_score(self.u_label, labels)) 
        """
        unique_label = np.sort(np.unique(np.array(labels)))
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            #if label_now ==-1:
            #    continue
            labels = [i if x == label_now else x for x in labels]
        """
        timererank = time.strftime("%m_%d_%H:%M:%S")
        print("model traing begin ",timererank)
        new_train_data, label_to_images = self.generate_new_train_data(labels)
        
       
        #un_c=0
        maxc=0
        j=0
        for i in label_to_images:
            if len(label_to_images[i])>maxc:
                maxc=len(label_to_images[i])
                j=i
            
        print("maxc",maxc)
        
        #np.savetxt("compar1labe2.txt",label)
        #print("new_view",new_view)
        num_after_merge=len(np.unique((np.array(labels))))
        #num_after_merge = len(unique_label)
        print("num of label before merge: ", num_before_merge, " after_merge: ", num_after_merge, " sub: ",num_before_merge - num_after_merge)
        #np.savetxt("dblabels.txt",labels)
        #if min(labels)<0:
        #    num_after_merge-=1
        self.criterion = ExLoss(self.embeding_fea_size, num_after_merge,label_to_images,t=10).cuda()
        #print(labels)
        #np.savetxt("u_feas1",u_feas)
        
        
        return labels, new_train_data


def change_to_unlabel(dataset):
    #trimmed_dataset = []
    index_labels = []
    index_dict = OrderedDict()
    #with open("/home/sunxia/Bottom-up-Clustering-Person-Re-identification/image_view.json",'r', encoding='UTF-8') as f:
    #    img_dic = json.load(f)
    for (imgs, pid, camid) in dataset:
        #trimmed_dataset.append([imgs, pid, camid, pid])
        #print("imgs",imgs)
        index_labels.append(pid)  # index
        #index_dict[pid] = img_dic[str(imgs)]
        #return index_dict
    return index_labels,index_dict
