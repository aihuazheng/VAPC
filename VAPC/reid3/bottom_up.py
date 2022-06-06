import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
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
from reid.exclusive_loss import ExLoss
import json
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.rerank import re_ranking

class Bottom_up():
    def __init__(self, model_name, batch_size, num_classes,view_label,dataset, u_data, save_path, embeding_fea_size=1024,
                 dropout=0.5, max_frames=900, initial_steps=32, step_size=16):

        self.model_name = model_name
        self.num_classes = num_classes
        #self.data_dir = dataset.images_dir
        #self.is_video = dataset.is_video
        self.save_path = save_path
        self.view=[7637,9877,1458,13629,5128]
        self.index_dict = view_label
        self.dataset = dataset
        self.u_data = u_data
        self.u_label = np.loadtxt("/home/sunxia/Bottom/gd.txt")#np.array([label for _, label, _, _ in u_data])
        #self.c_label = np.loadtxt("/DATA/sunxia/color.txt")
        self.lamda=0.98
        self.dataloader_params = {}
        self.dataloader_params['height'] = 384
        self.dataloader_params['width'] =384
        self.dataloader_params['batch_size'] = batch_size
        self.dataloader_params['workers'] = 6

        self.batch_size = batch_size
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

        model = models.create(self.model_name, dropout=self.dropout,
                              embeding_fea_size=self.embeding_fea_size, fixed_layer=self.fixed_layer)
        #self.model = nn.DataParallel(model).cuda()
        #checkpoint = load_checkpoint('/DATA/sunxia/logs/checkpoint320384.pth.tar')
        checkpoint = load_checkpoint('/home/sunxia/Bottom/logs/bestcheckpoint038432.pth.tar')
        #checkpoint = load_checkpoint("/DATA/sunxia/logs/checkpoint219384+loss.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        self.model = nn.DataParallel(model).cuda()
        #self.criterion = ExLoss(self.embeding_fea_size, self.num_classes, t=10).cuda() 
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
        #data_dir = self.data_dir

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
        init_lr = 0.1 if step==0 else 0.01 
        step_size = self.step_size if step==0 else sys.maxsize

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
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        """ main training process """
        trainer = Trainer(self.model, self.criterion, fixed_layer=self.fixed_layer)
        for epoch in range(epochs):#epochs
            adjust_lr(epoch, step_size)
            trainer.train(epoch, dataloader, optimizer, print_freq=max(5, len(dataloader) // 30 * 10))
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
        #np.savetxt("features_name.txt",list(features.keys()))
        features = np.array([logit.numpy() for logit in features.values()])
        #print("load_features")
        #np.savetxt("features.txt",features)
        #print("features",features)
        #fcs = np.array([logit.numpy() for logit in fcs.values()])
        #print("fcs",fcs)
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
        query_loader = DataLoader(
            Preprocessor(query,root="/home/sunxia/VeRi", transform=test_transformer),
            batch_size=64, num_workers=self.data_workers,
            shuffle=False, pin_memory=True)
        gallery_loader = DataLoader(
            Preprocessor(gallery,root="/home/sunxia/VeRi", transform=test_transformer),
            batch_size=64, num_workers=self.data_workers,
            shuffle=False, pin_memory=True)
        evaluator = Evaluator(self.model)
        evaluator.evaluate(query_loader, gallery_loader, query, gallery, output_feature)

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
        dists = dists.numpy()
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

    def generate_new_train_label(self, idx1, idx2,u_feas,label,num_to_merge,dists,flag,view_to_images):
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
        print("generate_new_train_data")
        ced = []
        #label11=[]
        #label22=[]
        """
        print("view1",len(label[0:7637]),min(label[0:7637]),max(label[0:7637]))
        print("view2",len(label[7637:7637+9877]),min(label[7637:7637+9877]),max(label[7637:7637+9877]))
        print("view3",len(label[7637+9877:7637+9877+1458]),min(label[7637+9877:7637+9877+1458]),max(label[7637+9877:7637+9877+1458]))
        print("view4",len(label[7637+9877+1458:7637+9877+1458+13629]),min(label[7637+9877+1458:7637+9877+1458+13629]),max(label[7637+9877+1458:7637+9877+1458+13629]))
        print("view5",len(label[7637+9877+1458+13629:7637+9877+1458+13629+5128]),min(label[7637+9877+1458+13629:7637+9877+1458+13629+5128]),max(label[7637+9877+1458+13629:7637+9877+1458+13629+5128]))
        """
        #merge_clustr=[]
        #index_clustr=[]
        ca_min=[]
        ca_max=[]
        ca_margin=[]
        ea_min=[]
        ea_max=[]
        ea_margin=[]
        for i in range(len(idx1)):
            label1 = label[idx1[i]]
            label2 = label[idx2[i]]
            #label11.append(idx1[i])
            #label22.append(idx2[i])
            #print("index11",idx1[i])
            #print("index22",idx2[i])
            """
            if flag==1:
                if len(set(ced))>=1000:
                    break
            
            
            if i % 100==0:
                print("label1,label2",label1,label2)
                print("dists[label1,label2]",dists[idx1[i],idx2[i]])
                print("index1,indx2",idx1[i],idx2[i])
                print("viewa,viewb",self.judgeview(idx1[i]),self.judgeview(idx2[i]))
            """
            
            if label1 in ced or label2 in ced or label1==label2:
                continue
            else:
                ced.append(label1)
                ced.append(label2)
            """
            sim=[]
            for i in range(len(view_to_images[label1])): 
                for j in range(i+1,len(view_to_images[label2])):
                    if dists[view_to_images[label1][i],view_to_images[label2][j]]<100000:
                        sim.append(dists[view_to_images[label1][i],view_to_images[label2][j]])
                    #print("sim",sim)
            """
            #if len(sim)>0:        
            #    if max(sim)-min(sim)>0.0057:#0025
            #        print("min(sim)",min(sim),"max(sim)",max(sim))
            #        continue
            #if self.c_label[idx1[i]]!=self.c_label[idx2[i]]:
            #    continue
            if label1 < label2:
                label = [label1 if x == label2 else x for x in label]
            else:
                label = [label2 if x == label1 else x for x in label]
            
            if self.u_label[idx1[i]] == self.u_label[idx2[i]]:
                correct += 1
                #if len(sim)>0:
                #    ca_min.append(min(sim))
                #    ca_max.append(max(sim))
                #    ca_margin.append(max(sim)-min(sim))
                if self.judgeview(idx1[i])==self.judgeview(idx2[i]):
                    s_s+=1
                else:
                    s_d+=1
            else:
                error+=1
                if self.judgeview(idx1[i])==self.judgeview(idx2[i]):
                    d_s+=1
                else:
                    d_d+=1
                #if len(sim)>0:
                #    ea_min.append(min(sim))
                #    ea_max.append(max(sim))
                #    ea_margin.append(max(sim)-min(sim))
            num_merged =  num_before_merge - len(np.sort(np.unique(np.array(label))))
            if num_merged >= num_to_merge or i>=715457428:
                break
            #if dists[idx1[i],idx2[i]]>0.0012:
            #    print("dists[idx1[i],idx2[i]]",dists[idx1[i],idx2[i]])
            #    break   
            cnt+=1
        #print(max(ca_max),min(ca_max),max(ca_margin),min(ca_margin),'-',max(ea_max),min(ea_max),max(ea_margin),min(ea_margin))
        print("s_s",s_s,s_s/cnt,"s_d",s_d,s_d/cnt,"d_s",d_s,d_s/cnt,"d_d",d_d,d_d/cnt)
        print("cnt",cnt,correct,"correc rate",correct/cnt,error,"error rate ",error/cnt)
           
        if flag==-1:
            """
            unique_label = np.sort(np.unique(np.array(label)))
            for i in range(len(unique_label)):
                label_now = unique_label[i]
                label = [i if x == label_now else x for x in label]
            """    
            print("adaptiva selected samples")
            """
            list_price_positoin_address = []
            for i in label:
                address_index = [x for x in range(len(label)) if label[x] == i]
                list_price_positoin_address.append([i, address_index])
            sl_label_to_images = dict(list_price_positoin_address)
            """
            #print("sl_label_to_images",len(sl_label_to_images))
            sl_label_to_images = {}
            for idx, l in enumerate(label):
                sl_label_to_images[l] = sl_label_to_images.get(l, []) + [idx]
            #feature_avg = np.zeros((max(label)+1, len(u_feas[0])))
            print("max(label)",max(label))
            #print("sl_label_to_images",sl_label_to_images)
            for l in sl_label_to_images:
                #print("l",l)
                sim=[]
                if len(sl_label_to_images[l])>2:
                    #feas = u_feas[sl_label_to_images[l]]
                    #print("feas",feas.shape)
                    #feature_avg[l] = np.mean(feas, axis=0)
                    #print("feature_avg[l]",feature_avg[l].shape)
                    #sim = torch.cosine_similarity(torch.from_numpy(np.reshape(feature_avg[l],(1,2048))).double(),torch.from_numpy(feas).double(), dim=1)
                    for i in range(len(sl_label_to_images[l])): 
                        for j in range(i+1,len(sl_label_to_images[l])):
                            if dists[sl_label_to_images[l][i],sl_label_to_images[l][j]]<100000:
                                sim.append(dists[sl_label_to_images[l][i],sl_label_to_images[l][j]])
                    #print("sim",sim)
                    if len(sim)>0:        
                        if max(sim)-min(sim)>0.002:#0025
                            #print("current cluster",label[sl_label_to_images[l][0]])
                            #print("original cluster",ori_label[sl_label_to_images[l][0]],ori_label[sl_label_to_images[l][-1]])
                            print("min(sim)",min(sim),"max(sim)",max(sim))
                            #print("sl_label_to_images[l]",sl_label_to_images[l])
                            errorlable=np.concatenate((errorlable,sl_label_to_images[l])).astype(np.int32)
                    
                    if l == 0:
                        print("sim",sim)
                    #print("sim",sim)
                    """
                    sle=(sim<0.97).nonzero()
                    #print("sle",sle)
                    tem_label=[]
                    if len(sle)>0:
                        if len(sle)>10:
                            print("sle",sle)
                        for i in sle:
                            tem_label.append(ori_label[sl_label_to_images[l][i]])
                        for j in sl_label_to_images[l]:
                            if ori_label[j] in tem_label:
                                errorlable.append(j)
                    """
            cont=0
            for i in range(len(label)):
                if label[i]==ori_label[i]:
                    cont+=1
            print("cont",cont)
            #print("len(set(array1) & set(array2))",len(set(array1) & set(array2)))
            if len(errorlable)>0:
                for i in errorlable:
                    label[i]=ori_label[i]
            cont=0
            for i in range(len(label)):
                if label[i]==ori_label[i]:
                    cont+=1
            print("cont",cont)
            print("errorlable",errorlable)
       
        print("set new label to the new training data")
        #np.savetxt("compar1label.txt",label)
        unique_label = np.sort(np.unique(np.array(label)))
        #label=label-min(label)
        
        
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            label = [i if x == label_now else x for x in label]
        
        #np.savetxt("compar1labe2.txt",label)

        num_after_merge = len(np.unique(np.array(label)))
        print("num of label before merge: ", num_before_merge, " after_merge: ", num_after_merge, " sub: ",
              num_before_merge - num_after_merge)
        return label
    def generate_new_train_diff_label(self,step, idx1, idx2,u_feas,label,num_to_merge,dists,flag,view_to_images):
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
        yuzhi=1000
        hold_set=[]
        print("yuzhi w balance",yuzhi)
        """
        for i in range(yuzhi):
            #print("dists[idx1[i],idx2[i]]",dists[idx1[i],idx2[i]])
            if dists[idx1[i],idx2[i]]!=100000:      
                hold_set.append(dists[idx1[i],idx2[i]])
            else:
                hold_set.append(dists[idx2[i],idx1[i]])
        """   
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
        hold=0.003
        print(self.hold,"self.hold")
        hit_1=0
        c_krnn=0
        b=[]
        d=[]
        for i in range(len(idx1)):
            label1 = label[idx1[i]]
            label2 = label[idx2[i]]
            if label1 in ced or label2 in ced:
                continue
            else:
                ced.append(label1)
                ced.append(label2)
            #if idx2[i] not in krnn_set[idx1[i]] or idx1[i] not in krnn_set[idx2[i]]:
            #    c_krnn+=1
            #    continue
            b.append((label1,label2))
            d.append((idx1[i],idx2[i]))    
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
        #print("a",a)
        #label11=[]
        #label22=[]
        """
        print("view1",len(label[0:7637]),min(label[0:7637]),max(label[0:7637]))
        print("view2",len(label[7637:7637+9877]),min(label[7637:7637+9877]),max(label[7637:7637+9877]))
        print("view3",len(label[7637+9877:7637+9877+1458]),min(label[7637+9877:7637+9877+1458]),max(label[7637+9877:7637+9877+1458]))
        print("view4",len(label[7637+9877+1458:7637+9877+1458+13629]),min(label[7637+9877+1458:7637+9877+1458+13629]),max(label[7637+9877+1458:7637+9877+1458+13629]))
        print("view5",len(label[7637+9877+1458+13629:7637+9877+1458+13629+5128]),min(label[7637+9877+1458+13629:7637+9877+1458+13629+5128]),max(label[7637+9877+1458+13629:7637+9877+1458+13629+5128]))
        """ 
        #merge_clustr=[]
        #index_clustr=[]
        ca_min=[]
        ca_max=[]
        ca_margin=[]
        ea_min=[]
        ea_max=[]
        ea_margin=[]
        cnt_ced={}
        for i in range(len(b)):
            label1 = b[i][0]
            label2 = b[i][1]
            idx1=d[i][0]
            idx2=d[i][1]
            #if label1==-1 or label2==-1:
            #    continue
            #label11.append(idx1[i])
            #label22.append(idx2[i])
            #print("index11",idx1[i])
            #print("index22",idx2[i])
            """
            if flag==1:
                if len(set(ced))>=1000:
                    break
            
            
            if i % 100==0:
                print("label1,label2",label1,label2)
                print("dists[label1,label2]",dists[idx1[i],idx2[i]])
                print("index1,indx2",idx1[i],idx2[i])
                print("viewa,viewb",self.judgeview(idx1[i]),self.judgeview(idx2[i]))
            """
            
            #if label1 in ced or label2 in ced or label1==label2:
            #    continue
            #else:
            #    ced.append(label1)
            #    ced.append(label2)
            """
            if label1==label2:
                continue
                
            if label1 in cnt_ced:
                if cnt_ced[label1]>2:
                    continue
            else:
                cnt_ced[label1]=0
            if label2 in cnt_ced: 
                if cnt_ced[label2]>2:
                    continue
            else:  
                cnt_ced[label2]=0
            
            cnt_ced[label1]+=1
            cnt_ced[label2]+=1
            """
         
                    #if dists[view_to_images[label1][i],view_to_images[label2][j]]<100000:
                        #sim.append(dists[view_to_images[label1][i],view_to_images[label2][j]])
                    #print("sim",sim)
            #if len(sim)>0:        
            #    if max(sim)-min(sim)>0.0057:#0025
            #        print("min(sim)",min(sim),"max(sim)",max(sim))
            #        continue
            #if self.c_label[idx1[i]]!=self.c_label[idx2[i]]:
            #    continue
            if label1 < label2:
                label = [label1 if x == label2 else x for x in label]
            else:
                label = [label2 if x == label1 else x for x in label]
          
            if self.u_label[idx1] == self.u_label[idx2]:
                correct += 1
                
                if self.judgeview(idx1)==self.judgeview(idx2):
                    s_s+=1
                else:
                    s_d+=1
            else:
                error+=1
                if self.judgeview(idx1)==self.judgeview(idx2):
                    d_s+=1
                else:
                    d_d+=1
               
          
            #num_merged =  num_before_merge - len(np.sort(np.unique(np.array(label))))
            #if num_merged >= num_to_merge or i>=715457428:
            #    break
            #if dists[idx1[i],idx2[i]]>0.0012:
            #    print("dists[idx1[i],idx2[i]]",dists[idx1[i],idx2[i]])
            #    break
               
            cnt+=1
        
        #print("np.mean(ca_min)",np.mean(ca_min),"np.mean(ea_min)",np.mean(ea_min))
        #print(max(ca_margin),min(ca_margin),'-',max(ea_margin),min(ea_margin))
        #print("s_s",s_s,s_s/cnt,"s_d",s_d,s_d/cnt,"d_s",d_s,d_s/cnt,"d_d",d_d,d_d/cnt)
        #print("cnt",cnt,correct,"correc rate",correct/cnt,error,"error rate ",error/cnt)
        
        unique_label = np.sort(np.unique(np.array(label)))
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            label = [i if x == label_now else x for x in label]
        
        #np.savetxt("compar1labe2.txt",label)

        num_after_merge = len(np.unique(np.array(label)))
        print("num of label before merge: ", num_before_merge, " after_merge: ", num_after_merge, " sub: ",
              num_before_merge - num_after_merge)
        return label
    def generate_new_train_data(self,label):
        label = [int(i) for i in label]
        new_train_data = []
        for idx, data in enumerate(self.u_data):
            new_data = copy.deepcopy(data)
            new_data[1] = label[idx]
            new_train_data.append(new_data)
        label_to_images = {}
        #view_to_images = {}
        for idx, l in enumerate(label):
            label_to_images[l] = label_to_images.get(l, []) + [idx]
        return new_train_data,label_to_images
        
    def generate_average_feature(self, labels):
        #extract feature/classifier
        u_feas = self.get_feature(self.u_data)
        
        #images of the same cluster
        label_to_images = {}
        #view_to_images = {}
        for idx, l in enumerate(labels):
            label_to_images[l] = label_to_images.get(l, []) + [idx]
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
        return u_feas, label_to_images
        
    def calculate_samv_distance(self,u_feas):
        # calculate distance between features
        x = u_feas
        y = x
        m = len(u_feas)
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())
        return dists
        
    def get_new_train_data(self, step, labels, nums_to_merge, size_penalty):
        flag = 1
        u_feas,view_to_images = self.generate_average_feature(labels)
        #np.savetxt("u_feas1",u_feas)
        """
        if step == 0:
            save_checkpoint({
            'state_dict': self.model.module.state_dict(),
            'step': step,
            }, fpath=osp.join('logs', 'checkpoint0224.pth.tar')) 
        
        if step == 19:
            save_checkpoint({
            'state_dict': self.model.module.state_dict(),
            'step': step,
            }, fpath=osp.join('logs', 'checkpoint219384+rerank006.pth.tar'))
            np.savetxt("label219384+rerank006.txt", np.array(labels,dtype=np.int16))
        
        if step == 20:
            save_checkpoint({
            'state_dict': self.model.module.state_dict(),
            'step': step,
            }, fpath=osp.join('logs', 'checkpoint320384+rerank006.pth.tar'))
            np.savetxt("label320384+rerank006.txt", np.array(labels,dtype=np.int16))
        """
        
        merge_perce=[0.057,0.058,0.05,0.059,0.056]

        print("clustring same view")
        u_feas = torch.from_numpy(u_feas)
        v = torch.split(u_feas, self.view, 0)
        for i in range(0,5):
            cnt_view = 0
            #dists = re_ranking(0, v[i].numpy(), lambda_value=0)
            dists = self.calculate_samv_distance(v[i])

            #idx1, idx2 = self.select_merge_data(v[i],v[i],labels, label_to_images, size_penalty,dists,state)
            nums_to_merge =  len(v[i]) * 0.05#merge_perce[i]
            if i > 0:
                for j in range(0, i):
                    cnt_view += self.view[j]
                    print("i",i,"cnt_view",cnt_view)
                #idx1 += cnt_view
                #idx2 += cnt_view
            
            m = len(v[i])
            n = len(v[i])

            dists.add_(torch.tril(100000 * torch.ones(m, n)))
            for idx in range(m):
                for j in range(idx + 1, n):
                    if labels[cnt_view + idx] == labels[cnt_view + j]:
                        dists[idx, j] = 100000
            idx1, idx2 = self.select_merge_data(dists)
            idx1 += cnt_view
            idx2 += cnt_view                
            labels = self.generate_new_train_label(idx1, idx2,u_feas,labels,nums_to_merge,dists,flag,view_to_images)
            
        #global dists 
        #dists = self.calculate_distance(u_feas)


        print("clustring in the different view")
        flag = 1
        
        dists = self.calculate_samv_distance(u_feas)
        #np.savetxt("seconde_dists.txt",dists)
        print("save_dist_done")
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
        dists[0:37729,0:7637]=100000
        #dists[0:7637,9877+7637:37729]=100000
        
        dists[7637:37729,7637:7637+9877]=100000
        #dists[7637:7637+9877,7637+9877+1458:37729]=100000
        
        ##dists[7637+9877:37729,7637+9877:7637+9877+1458]=100000
        dists[0:37729,7637+9877:7637+9877+1458]=100000
        #dists[7637+9877:7637+9877+1458,7637+9877+1458+13629:37729]=100000
        
        dists[7637+9877+1458:37729,7637+9877+1458:13629+7637+9877+1458]=100000
        dists[7637+9877+1458+13629:37729,13629+7637+9877+1458:37729]=100000
        
        
        #dists.add_(torch.tril(100000 * torch.ones(m, n)))
        for idx in range(len(u_feas)):
            for j in range(idx + 1, len(u_feas)):
                if labels[idx] == labels[j]:
                    dists[idx, j] = 100000
        
        #cnt = torch.FloatTensor([len(view_to_images[labels[idx]]) for idx in range(len(u_feas))])
        #np.savetxt("cnt.txt",cnt.numpy())
        #dists += 0.0003 * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))
        #del cnt
        nums_to_merge =  len(np.unique(np.array(labels)))* 0.3#merge_perce[i]
        idx1, idx2 = self.select_merge_data(dists)
        labels = self.generate_new_train_diff_label(step, idx1, idx2,u_feas,labels,nums_to_merge,dists,flag,view_to_images)
            #new_train_data, labels = self.generate_new_train_data(idx1, idx2,u_feas,labels,nums_to_merge,flag)
        new_train_data, label_to_images = self.generate_new_train_data(labels)
        num_train_ids = len(np.unique(np.array(labels)))

        # change the criterion classifer
        self.criterion = ExLoss(self.embeding_fea_size, num_train_ids,label_to_images,t=10).cuda()
        #new_classifier = fc_avg.astype(np.float32)
        #self.criterion.V = torch.from_numpy(new_classifier).cuda()
       
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
