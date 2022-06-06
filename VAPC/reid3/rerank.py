#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
Modified by L.Song and C.Wang
"""
import torch
from collections import OrderedDict
import numpy as np
from scipy.spatial.distance import cdist
import torch.nn.functional as F

#
def re_ranking(input_feature_source, input_feature, step=0,flag=0,k1=20, k2=6):

    all_num = input_feature.shape[0]    
    feat = input_feature
    krnn_set=OrderedDict()
    """
    if lambda_value != 0:
        print('Computing source distance...')
        all_num_source  = input_feature_source.shape[0]
        sour_tar_dist = np.power(
            cdist(input_feature, input_feature_source), 2).astype(np.float16)
        sour_tar_dist = 1-np.exp(-sour_tar_dist)
        source_dist_vec = np.min(sour_tar_dist, axis = 1)
        source_dist_vec = source_dist_vec / np.max(source_dist_vec)
        source_dist = np.zeros([all_num, all_num])
        for i in range(all_num):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]
        del sour_tar_dist
        del source_dist_vec
    """

    print('Computing original distance...')
    """
    original_dist = cdist(feat,feat).astype(np.float16)  
    original_dist = np.power(original_dist,2).astype(np.float16)
    del feat    
    euclidean_dist = original_dist
    """
    print("calculate_distance")
    x = feat
    y = x
    m = len(feat)
    dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
    dists.addmm_(1, -2, x, y.t())
    original_dist= dists.numpy()
    del feat
    euclidean_dist = original_dist
   
    if step!=1 and flag ==1:
        print("differnet view dont re rank")
        #euclidean_dist = np.transpose(euclidean_dist/np.max(euclidean_dist,axis = 0))
        return euclidean_dist
    gallery_num = euclidean_dist.shape[0] #gallery_num=all_num
    print("1")
    euclidean_dist = np.transpose(euclidean_dist/np.max(euclidean_dist,axis = 0))
    V = np.zeros_like(euclidean_dist).astype(np.float16)
    if flag==1:
        print("dont compute same view")
        euclidean_dist[0:7637,0:7637]=100000
            #dists[0:7637,9877+7637:37729]=100000
            
        euclidean_dist[7637:7637+9877,7637:7637+9877]=100000
        #dists[7637:7637+9877,7637+9877+1458:37729]=100000
        
        #dists[7637+9877:37729,7637+9877:7637+9877+1458]=100000
        euclidean_dist[7637+9877:7637+9877+1458,7637+9877:7637+9877+1458]=100000
        #dists[7637+9877:7637+9877+1458,7637+9877+1458+13629:37729]=100000
        
        euclidean_dist[7637+9877+1458:13629+7637+9877+1458,7637+9877+1458:13629+7637+9877+1458]=100000
        euclidean_dist[7637+9877+1458+13629:37729,13629+7637+9877+1458:37729]=100000
    initial_rank = np.argsort(euclidean_dist).astype(np.int32)  ## default axis=-1.  
    
    print('Starting re_ranking...')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
        fi = np.where(backward_k_neigh_index==i)[0]  
        k_reciprocal_index = forward_k_neigh_index[fi]   ## get R(p,k) in the paper
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
       
        krnn_set[i]= k_reciprocal_expansion_index
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])  
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    """
    cnt_c=0
    clusring=OrderedDict()
    done=[]
    print("krnn_set",krnn_set)
    for i in krnn_set:
        if i in done:
            continue
        tmp=krnn_set[i]
        for j in range(1,len(krnn_set[i])):
            if len(np.intersect1d(krnn_set[i],krnn_set[krnn_set[i][j]]))> 0.9*len(krnn_set[i]):
                tmp=np.concatenate((tmp,krnn_set[krnn_set[i][j]]))
            done.append(krnn_set[i][j])
        clusring[cnt_c]=np.unique(tmp)
        cnt_c+=1
    print("clusring",clusring) 
    """
       
    #original_dist = original_dist[:query_num,]    
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float16)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = [] 
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float16)


    for i in range(all_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float16)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    if flag == 0:
        print("same view with re rank")
        return jaccard_dist,original_dist
    if flag == 1:
        print("different view with re rank")
        return original_dist,original_dist
    
    #else:
    #    final_dist = jaccard_dist*(1-lambda_value) + source_dist*lambda_value
    #    return final_dist
    
