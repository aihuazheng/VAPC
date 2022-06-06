import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os.path as osp
import matplotlib
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils
import cv2
from scipy import misc
import scipy
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)
matplotlib.use('Agg')
class_names  = ['1', '2', '3', '4','5']
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

num_classes = 5

num_epoch = 50
batch_size = 64
learning_rate = 0.001
ft_learning_rate=0.001
"""
train_dataset = torchvision.datasets.ImageFolder(root="/home/sunxia/data/VeRi/VeRiTrain/",transform=transforms.Compose([
                                                       transforms.Resize(size=(224,224)),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       normalize,
                                                      ]))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                                            shuffle=True,num_workers=4)
"""                                 
test_dataset = torchvision.datasets.ImageFolder(root='/home/sunxia/viewtest',transform=transforms.Compose([
                                                           transforms.Resize(size=(224,224)),
                                                           transforms.ToTensor(),
                                                           normalize,
                                                  ]))

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=20,
                                          shuffle=False,num_workers=4)



ft_dataset = torchvision.datasets.ImageFolder(root="/home/sunxia/WILD-VIEW/",transform=transforms.Compose([
                                                           transforms.Resize(size=(224,224)),
                                                           transforms.RandomHorizontalFlip(p=1.0),
                                                           transforms.ToTensor(),
                                                           normalize,
                                                  ]))
ft_loader = torch.utils.data.DataLoader(ft_dataset,batch_size=32,
                                          shuffle=True,num_workers=4)        
print("ft_loader",len(ft_dataset))                             


prec_dataset = torchvision.datasets.ImageFolder(root="/DATA/sunxia/VERI-train/",transform=transforms.Compose([
                                                           transforms.Resize(size=(224,224)),
                                                           transforms.ToTensor(),
                                                           normalize,
                                                  ]))
print("2")                                                  
prec_loader = torch.utils.data.DataLoader(prec_dataset,batch_size=32,
                                          shuffle=False,num_workers=4)   

"""
prec_dataset = torchvision.datasets.ImageFolder(root="/home/sunxia/data/VeRi/VeRiTrain/",transform=transforms.Compose([
                                                           transforms.Resize(size=(224,224)),
                                                           transforms.ToTensor(),
                                                           normalize,
                                                  ]))
prec_loader = torch.utils.data.DataLoader(prec_dataset,batch_size=32,
                                          shuffle=False,num_workers=4) 
"""                                            
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = ft_learning_rate * (0.1 ** (epoch // 8))
    print("lr",lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
           

def train(net,optimizer,loss_fn,num_epoch,data_loader,device):
    net.train()
    Loss_list = []
    Accuracy_list = []
    best_test=0.
    print("data_loader",len(data_loader))
    for epoch in range(num_epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        adjust_learning_rate(optimizer,epoch)
        for i,data in enumerate(data_loader):
            inputs,labels = data[0].to(device),data[1].to(device)           
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

            print('Epoch = %1d, loss = %.3f' % (epoch+1,running_loss/32))
                #running_loss = 0.0
        print('Accuracy of the network on the training images2: %d %%' % (100 * correct / total))
        test_acc, test_matrix = evaluate(net=net,
                                           data_loader=data_loader,
                                           device=device, state='train')
        save_checkpoint({
                        'state_dict': net.state_dict(),
                        'step': epoch,
                        }, fpath='WILD-new-2.pth.tar')
        
        """
        if 100 * correct / total>best_test:
            if 100 *correct / total>70:
                print("save")
                save_checkpoint({
                    'state_dict': net.state_dict(),
                    'step': epoch,
                    }, fpath='WILD-new'+str(100 * correct / total)+'.pth.tar')
            best_test=100 * correct / total
        """
        
        #print("train_matrix")
        #print(train_matrix)
        print("test_matrix")
        print(test_matrix)
        print('Test Accuracy: %.2f %%' % (100*test_acc))
        Loss_list.append(loss)
        Accuracy_list.append(test_acc)
    x1 = range(0, num_epoch)
    x2 = range(0, num_epoch)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('training loss vs. epoches')
    plt.ylabel('training loss')
    #plt.show()
    plt.savefig("accuracy_loss.jpg")

def evaluate(net,data_loader,device,state):
    was_training = net.training
    net.eval()
    correct = 0.0
    total = 0
    images_so_far = 0
    matrix = np.zeros((5, 5))
    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))
    num_images = 6
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                matrix[labels[i]][predicted[i]] += 1
    acc = correct / total
    return acc,matrix


def main():
    
    net = models.resnet50(pretrained=True)
    featureSize = net.fc.in_features
    net.fc = nn.Linear(featureSize,num_classes)
    checkpoint = load_checkpoint("/home/sunxia/checkpointbestviewprec.pth.tar")
    net.load_state_dict(checkpoint['state_dict'])
    
    
    #state_dict = torch.load('resnet50.pth')
    #net.load_state_dict(state_dict)
    #net = models.googlenet(pretrained=True)
    #print("net",net)
   
    exclude_layers = ['layer1','layer2','layer3']
    for name, param in net.named_parameters():
        for layer in exclude_layers:
            if name.startswith(layer):
                param.requires_grad = False
                break
  
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = learning_rate, momentum = 0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #optimizer = torch.optim.RMSprop(net.parameters(),lr=learning_rate)  
    net.to(device)
    print("test")
    np.set_printoptions(suppress=True)
    test_acc,test_matrix = evaluate(net=net,
                         data_loader = test_loader,
                         device=device,state='train')
    print(test_acc,test_matrix)
    print("test")
    train(net=net,
          optimizer=optimizer,
          loss_fn =  criterion,
          num_epoch = num_epoch,
          data_loader = ft_loader,
          device = device)  
    
    test_acc,test_matrix = evaluate(net=net,
                         data_loader = test_loader,
                         device=device,state='train')
    
    print("test_matrix")
    print(test_matrix)
    #print("test_matrix")
    #print(test_matrix)
    print('Test Accuracy: %.2f %%' % (100*test_acc))
    
    #print('Test Accuracy: %.2f %%' % (100 * test_acc))
    
def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    
    torch.save(state, fpath)
    #if is_best:
    #    shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model    

def imshow(inp,pre, lable,j,title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    #cv2.imwrite("precview/"+pre+'/'+pre+'_'+str(j)+'.jpg',cv2.normalize(inp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),[int(cv2.IMWRITE_JPEG_QUALITY),95])
    #if not os.path.exists("/DATA/sunxia/VERI-preview/"+pre+'/'):
    #    os.makedirs("/DATA/sunxia/VERI-preview/"+pre+'/')
    print("/DATA/sunxia/VERI-WILD-view-aware/"+pre+'/'+str(lable.item())+'_c'+str(j)+'.jpg')
    scipy.misc.imsave("/DATA/sunxia/VERI-WILD-view-aware/"+pre+'/'+str(lable.item())+'_c'+str(j)+'.jpg',inp)
    #cv2.destroyAllWindows()
    """Imshow for Tensor."""
    
    #plt.axis('off')

    #folder = os.path.exists("precview/"+pre+'/')
    #if not folder:
    #    os.makedirs(label)

    """
    plt.savefig("precview/"+pre+'/'+pre+'_'+str(j)+'.jpg')
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    """
def prec():
    net = models.resnet50()
    featureSize = net.fc.in_features
    net.fc = nn.Linear(featureSize,num_classes)
    checkpoint = load_checkpoint("WILD-new-2.pth.tar")
    net.load_state_dict(checkpoint['state_dict'])
    net=net.to(device)
    net.eval()
    correct = 0.0
    total = 0
    images_so_far = 0
    matrix = np.zeros((5, 5))
    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))
    num_images = 6
    #fig = plt.figure()
    with torch.no_grad():
        cnt=0
        for data in prec_loader:
            images, labels = data[0].to(device), data[1].to(device)
            #print("labels",labels)
            #print("images",images)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            #print("predicted",predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            """
            imgs_t = outputs / 2 #+ 0.5
            imgs_t = vutils.make_grid(imgs_t.data, nrow=1, normalize=False, scale_each=True).cpu().numpy()
            imgs_t = np.clip(imgs_t * 224, 0, 224).astype(np.uint8)
            imgs_t = imgs_t.transpose(1, 2, 0)
            imgs_t = Image.fromarray(imgs_t)
            """
            #print(predicted,labels)
            
            for j in range(labels.size()[0]):
                #ax = plt.subplot()
                #ax.axis('off')
                #ax.set_title('{}:predict: {}'.format(classes[labels[j]],classes[preds[j]]))
                imshow(inp=images.cpu().data[j],pre=class_names[predicted[j]],lable=labels[j],j=j+cnt*32)
                print(j+cnt*32)
                #print("{} pred label:{}, true label:{}".format(len(predicted), class_names[predicted[j]], class_names[labels[j]]))
            cnt+=1
            
            
            #filename = str(class_names[predicted[j]])+'.jpg'
            #imgs_t.save(os.path.join("precview", filename))
            
   
    
    #print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(2)))

if __name__ == '__main__':
    #main()
    prec()
    #net = models.resnet50(pretrained=True)
    #print(net)


