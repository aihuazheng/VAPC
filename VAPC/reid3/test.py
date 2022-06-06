from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F
import torch


#a=torch.FloatTensor(2, 3)#np.arange(5, 20).dtype='float32'
#b=torch.FloatTensor(2, 3)#np.arange(6, 20).dtype='float32'
a=torch.tensor([[1.0,1.0,2.0,1.0,1.0,1.0,0.,0.,0.]])
a.reshape(1,-1)
b=torch.tensor([[1.0,1.0,1.0,0.,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,0.,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,0.,1.0,1.0,1.0,1.0,1.0]])
b.reshape(3,-1)
#a = torch.from_numpy(a)
#b = torch.from_numpy(b)





feature1 = F.normalize(a)
feature2 = F.normalize(b,p=2,dim=0)
print(feature1)
print(feature2)
#c = feature1.mm(feature2.t())
#c=cosine_similarity(a,b)

c=torch.cosine_similarity(a, b, dim=1)
print("c",c)

