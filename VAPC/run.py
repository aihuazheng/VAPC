from __future__ import print_function, absolute_import
#from reid.new3bottom import *
#from reid.ori_bottom import *
#from reid.votto_bottom import *
from reid3.veri import *
from reid import datasets
from reid import models
import numpy as np
import argparse
import os, sys, time
from reid.utils.logging import Logger
import os.path as osp
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.datasets.data import Data
from reid.datasets.data3 import VehicleID
#from reid.datasets.data2wild import Data#加载wild数据集时取消该注释
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint, save_checkpoint

def set_seed(seed):
    if seed < 0:
        seed = random.randint(0, 10000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    return seed
    
def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True    
    save_path = args.logs_dir

    
    sys.stdout = Logger(osp.join('./', 'xlsy-noall'+ str(args.merge_percent)+ time.strftime(".%m_%d_%H:%M:%S") + '.txt'))
    seed = set_seed(args.seed)
    print('Random seed of this run: %d\n' % seed)
    print("args",args)
    #dataset = Data('/DATA/', 'sunxia')
    dataset = Data('/home/sunxia/data/VeRi776/', 'VeRi')
    #dataset = VehicleID(root='/DATA/sunxia/', verbose=True)
    # get all unlabeled data for training
    #dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset)) 
    cluster_id_labels,view_label = change_to_unlabel(dataset.target_train)
    #np.savetxt("testlabel.txt",cluster_id_labels)
    #cluster_id_labels = np.loadtxt('/DATA/sunxia/label320384.txt').astype(np.int32)
    #cluster_id_labels = np.loadtxt('/DATA/sunxia/label219384+loss.txt').astype(np.int32)
    
    #print("cluster_id_labels",cluster_id_labels)
    new_train_data= dataset.target_train
    #print("cluster_id_labels",cluster_id_labels)
    num_train_ids = len(np.unique(np.array(cluster_id_labels)))

    #num_train_ids2 = dataset.num_train_ids
    #assert num_train_ids!=num_train_ids2, 'id error'
    #num_train_ids=13164
    nums_to_merge = int(num_train_ids * args.merge_percent)
    BuMain = Bottom_up(model_name=args.arch, batch_size=args.batch_size, 
            num_classes=num_train_ids,
            view_label=[],
            dataset=dataset,
            u_data=new_train_data, save_path=args.logs_dir, max_frames=args.max_frames,
            embeding_fea_size=args.fea)

    
   
    for step in range(1,50):
        print('step: ',step)
        #BuMain.train(new_train_data, step, loss=args.loss)
        #如果step从1开始那么是加载辨识阶段后的预训练模型，先聚类再聚类
        #如果step从0开始，那么先进行辨识阶段训练，再聚类，需要把BuMain.train(new_train_data, step, loss=args.loss)放到BuMain.get_new_train_data前面
        if step > 0:
            BuMain.evaluate(dataset.query,dataset.gallery, args.output_feature)
        # get new train data for the next iteration
        print('----------------------------------------bottom-up clustering------------------------------------------------')
        cluster_id_labels, new_train_data = BuMain.get_new_train_data(step,cluster_id_labels, nums_to_merge, size_penalty=args.size_penalty)
        BuMain.train(new_train_data, step, loss=args.loss)
        print('\n\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bottom-up clustering')
    parser.add_argument('-se', '--seed', type=int, default=2856, help="random seed for training, default: -1 (automatic)")
    parser.add_argument('-b', '--batch-size', type=int, default=32)  
    parser.add_argument('-f', '--fea', type=int, default=2048)
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',choices=models.names())
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    parser.add_argument('--max_frames', type=int, default=900)
    parser.add_argument('--loss', type=str, default='ExLoss')
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    parser.add_argument('-s', '--step_size', type=int, default=55)
    parser.add_argument('--size_penalty',type=float, default=0.005)
    parser.add_argument('-mp', '--merge_percent',type=float, default=0.05)
    parser.add_argument('--output_feature', type=str, default='pool5')
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    main(parser.parse_args())

