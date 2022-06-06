from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re
from functools import cmp_to_key

class Data(object):

    def __init__(self, data_dir, target):
        
        # source / target image root
        #self.source_images_dir = osp.join(data_dir, source)
        self.target_images_dir = osp.join(data_dir, target)
        # training image dir
        self.source_train_path = 'bounding_box_train'
        self.target_train_path = 'VERI-WILD-view-aware'
        # self.target_train_camstyle_path = 'bounding_box_train_camstyle'
        self.gallery_path = "images"
        self.query_path = 'images'

        self.target_train, self.query, self.gallery =[], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.cam_dict = self.set_cam_dict()
        #self.target_num_cam = self.cam_dict[target]
        #self.source_num_cam = self.cam_dict[source]

        self.load()
    
    def set_cam_dict(self):
        cam_dict = {}
        cam_dict['market'] = 6
        cam_dict['duke'] = 8
        cam_dict['msmt17'] = 15
        cam_dict['VeRi'] = 20
        cam_dict['sunxia'] = 1
        return cam_dict
    

    def preprocess(self, images_dir, path, relabel=True, fakelabel=False):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        ret = []
        if isinstance(path, (tuple, list)):
            paths = path
        else:
            paths = [path]
        for path_here in paths:
            if 'cuhk03' in images_dir:
                fpaths = sorted(glob(osp.join(images_dir, path_here, '*.png')))
            elif 'VeriTrain' in images_dir:
                fpaths1 = sorted(glob(osp.join('/home/sunxia/data/VeRi/VeRiTrain', '1', '*.jpg')))                
                fpaths2 = sorted(glob(osp.join('/home/sunxia/data/VeRi/VeRiTrain', '2', '*.jpg')))
                fpaths3 = sorted(glob(osp.join('/home/sunxia/data/VeRi/VeRiTrain', '3', '*.jpg')))
                fpaths4 = sorted(glob(osp.join('/home/sunxia/data/VeRi/VeRiTrain', '4', '*.jpg')))
                fpaths5 = sorted(glob(osp.join('/home/sunxia/data/VeRi/VeRiTrain', '5', '*.jpg')))
                fpaths = fpaths1 + fpaths2 + fpaths3 + fpaths4 + fpaths5
                # print('fpaths',fpaths)
                # print("len",len(fpaths)),key = lambda i:int(i.split("/")[5].split("_")[0])
            elif 'VERI-WILD-view-aware' in images_dir:
                fpaths1 = sorted(glob(osp.join("/DATA/sunxia/VERI-WILD-view-aware/", '1', '*.jpg')))
                
                #fpaths1=sorted(fpaths1,key = lambda i:int(i.split("/")[5].split("_")[0])))
                fpaths2 = sorted(glob(osp.join("/DATA/sunxia/VERI-WILD-view-aware/", '2', '*.jpg')))
                #fpaths3 = sorted(glob(osp.join('/DATA/sunxia/VERI-pview', '3', '*.jpg')))
                fpaths4 = sorted(glob(osp.join("/DATA/sunxia/VERI-WILD-view-aware/", '4', '*.jpg')))
                fpaths5 = sorted(glob(osp.join("/DATA/sunxia/VERI-WILD-view-aware/", '5', '*.jpg')))
                #print(fpaths1[0:10000])
                fpaths = fpaths1[0:10000] + fpaths2[0:10000]  + fpaths4[0:10000] + fpaths5[0:10000] #+ fpaths3
                
                """
                gd_lable_wild=[]
                for k in fpaths:
                    #print(k)
                    tmp=k.split("/")[5].split("_")[0]
                    gd_lable_wild.append(int(tmp))
                
                #print(gd_lable_wild)
                print(len(set(gd_lable_wild[0:10000]) & set(gd_lable_wild[10000:10000+10000])))
                print(len(set(gd_lable_wild[10000+10000:10000+10000+10000]) & set(gd_lable_wild[10000+10000+10000:10000+10000+10000+10000])))
                
                np.savetxt("gd_lable_wild.txt",gd_lable_wild)
                """
            elif 'VERI-test' in images_dir:
                fpaths=[]
                with open("/DATA/sunxia/VERI-WILD/train_test_split/test_3000.txt", "r") as f:
                    data = f.read().splitlines()
                for i in data:
                    fpaths.append("/DATA/sunxia/VERI-WILD/"+i+'.jpg')
            elif 'VERI-query' in images_dir:
                fpaths=[]
                with open("/DATA/sunxia/VERI-WILD/train_test_split/test_3000_query.txt", "r") as f:
                    data = f.read().splitlines()
                for i in data:
                    fpaths.append("/DATA/sunxia/VERI-WILD/"+i+'.jpg')
            else:
                fpaths = sorted(glob(osp.join(images_dir, path_here, '*.jpg')))

            for fpath in fpaths:
                #fname = osp.basename(fpath)
                fname = fpath.split('/')
                fname = fname[-2]+'/'+fname[-1]
                #print(fname)
                if 'cuhk03' in images_dir:
                    name = osp.splitext(fname)[0]
                    pid, cam = map(int, pattern.search(fname).groups())
                else:
                    if fakelabel:
                        pid = fname.split('.')[0]
                        _, cam = map(int, pattern.search(fname).groups())
                        # print("pid", pid,"_", _, "cam", cam)
                    else:
                        pid = int(fname.split('/')[0])
                        cam=int(fname.split('/')[1].split('.')[0])
                        #pid, cam = map(int, pattern.search(fname).groups())
                        #print("pid",pid,"cam",cam)
                if pid == -1: continue  # junk images are just ignored
                if relabel:
                    if pid not in all_pids:
                        all_pids[pid] = len(all_pids)
                else:
                    if pid not in all_pids:
                        all_pids[pid] = pid
                pid = all_pids[pid]
                #cam -= 1
                ret.append([fname, pid, cam])
            
            #if 'VERI-priew' in images_dir:
            #    print(images_dir,"VERI-priew",ret)
            #if 'VERI-test' in images_dir:
            #    print(images_dir,"VERI-test",ret)
            #if 'VERI-query' in images_dir:
            #    print(images_dir, "VERI-query", ret)
            
        return ret, int(len(all_pids))

    def load(self):
        #self.source_train, self.num_train_ids = self.preprocess(self.source_images_dir, self.source_train_path)
        print("For target train set, indexes are treated as identities of images")
        #self.target_train, self.num_target_ids = self.preprocess(self.target_images_dir, self.target_train_path, fakelabel=True)
        #self.target_train, self.num_target_ids = self.preprocess('/home/sunxia/data/VeRi/VeriTrain', 'VeriTrain',fakelabel=True)
        self.target_train, self.num_target_ids = self.preprocess('/DATA/sunxia/VERI-WILD-view-aware', 'VERI-WILD-view-aware',fakelabel=True)
        
        #print("train",self.target_train)
        self.gallery, self.num_gallery_ids = self.preprocess('/VERI-test', 'VERI-test',fakelabel=False)
        #print("gallery",self.gallery) 
        self.query, self.num_query_ids = self.preprocess('/VERI-query', 'VERI-query',fakelabel=False)
        #print("query",self.query)                                                
        #self.gallery, self.num_gallery_ids = self.preprocess(self.target_images_dir, self.gallery_path, False)
        #self.query, self.num_query_ids = self.preprocess(self.target_images_dir, self.query_path, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  target train    | {:5d} | {:8d}"
              .format(self.num_target_ids, len(self.target_train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))

