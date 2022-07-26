import os
import torch
import h5py
import pickle
import time
import numpy as np
from utils.args import *
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio
from data import get_train_loader, get_eval_loader
from tools import Array,mAP
from model import *

def train():
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    best_optimizer_pth_path = file_path+'/best_optimizer.pth'
    optimizer_pth_path = file_path+'/optimizer.pth'
    infos = {}
    histories = {}
    if use_checkpoint is True and os.path.isfile(os.path.join(file_path, 'infos.pkl')):
        with open(os.path.join(file_path, 'infos.pkl'),'rb') as f:
            infos = pickle.load(f)
        if os.path.isfile(os.path.join(file_path, 'histories.pkl')):
            with open(os.path.join(file_path, 'histories.pkl'),'rb') as f:
                histories = pickle.load(f)
    
    print('Learning rate: %.4f' % lr)
    epoch = 0
    
    if use_checkpoint:
        model.load_state_dict(torch.load(models_path+'_epoch_'+'xxx'+'.pth'))
        itera = infos.get('iter', 0)
        epoch = infos.get('epoch', 0)

    optimizer = Adam(model.parameters(), lr=lr)

    if os.path.exists(best_optimizer_pth_path) and use_checkpoint:
        optimizer.load_state_dict(torch.load(optimizer_pth_path))

    def set_lr(optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr

    train_loader= get_train_loader(train_feat_path, batch_size,shuffle = True)
    total_len = len(train_loader)
    
    while True: 
        itera = 0
        decay_factor = 0.9  ** ((epoch)//lr_decay_rate)
        model.set_alpha(epoch)
        current_lr = max(lr * decay_factor,1e-4)
        set_lr(optimizer, current_lr) # set the decayed rate
        for i, data in  enumerate(train_loader, start=1):
            optimizer.zero_grad()
            batchsize = data["double_video_feat"].size(0)
            data = {key: value.cuda() for key, value in data.items()}
            b1,h1,e1 = model.forward(data["double_video_feat"][:,:max_frames,:])
            b2,h2,e2 = model.forward(data["double_video_feat"][:,max_frames:,:])
            _,similar1,_ = model.forward(data["triple_video_feat"][:,:max_frames,:])
            _,dissimilar1,_ = model.forward(data["triple_video_feat"][:,max_frames:max_frames*2,:])
            _,dissimilar2,_ = model.forward(data["triple_video_feat"][:,max_frames*2:max_frames*3,:])
            
            cluter_loss = (torch.sum((torch.mean(e1,1)-data['n1'])**2)\
                            + torch.sum((torch.mean(e2,1)-data['n2'])**2))/(hidden_size*batchsize)

            sim = h1.mul(h2)
            sim = torch.sum(sim,1)/nbits
            sim_loss = torch.sum((1*data["is_similar"].float()-sim)**2)/batchsize
            q_loss=(torch.sum((h1-b1)**2)+torch.sum((h2-b2)**2))/(batchsize*nbits)

            pos = F.pairwise_distance(h1,similar1,p=2)**2
            neg1 = F.pairwise_distance(h1,dissimilar1,p=2)**2
            neg2 = F.pairwise_distance(dissimilar1,dissimilar2,p=2)**2
            quad_loss=torch.sum(F.relu(nbits/4.0+pos-neg1)+F.relu(nbits/64.0+pos-neg2))/batchsize
            
            loss=0.8*cluter_loss+0.1*sim_loss+0.01*q_loss+0.01*quad_loss
            
            loss.backward()
            optimizer.step()
            itera += 1
            infos['iter'] = itera
            infos['epoch'] = epoch
            if itera%10 == 0 or batchsize<batch_size:  
                print('Epoch:%d Step:[%d/%d] cluter_loss: %.2f sim_loss: %.2f q_loss: %.2f quad_loss: %.2f'\
                % (epoch, i, total_len, cluter_loss.data.cpu().numpy(),sim_loss.data.cpu().numpy(),\
                    q_loss.data.cpu().numpy(),quad_loss.data.cpu().numpy()))

        torch.save(model.state_dict(), models_path+'_epoch_'+str(epoch)+'.pth')
        torch.save(optimizer.state_dict(), optimizer_pth_path)
       
        with open(os.path.join(file_path, 'infos.pkl'), 'wb') as f:
            pickle.dump(infos, f)
        with open(os.path.join(file_path, 'histories.pkl'), 'wb') as f:
            pickle.dump(histories, f)
        epoch += 1
        if epoch>num_epochs:
            break
        model.train()

def eval():

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    h5_file = h5py.File(test_feat_path, 'r')
    video_feats = h5_file['feats']
    num_sample = len(video_feats)
    hashcode = np.zeros((num_sample,nbits),dtype = np.float32)
    rem = num_sample%test_batch_size

    label_array = Array()
    labels = sio.loadmat(label_path)['labels']
    label_array.setmatrcs(labels)
    labels = label_array.getmatrics()
    sim_labels = np.dot(labels, labels.transpose())

    eval_loader = get_eval_loader(test_feat_path,batch_size=test_batch_size)
    batch_num = len(eval_loader)

    for epoch in range(26,num_epochs+1):
        print('epoch:',epoch)
        print(models_name+'_epoch_'+str(epoch)+'.pth')
        model.load_state_dict(torch.load(models_path+'_epoch_'+str(epoch)+'.pth'))
        model.eval()

        time0 = time.time()
        for i, data in enumerate(eval_loader): 
            data = {key: value.to(device) for key, value in data.items()}
            _,h,_= model.forward(data["video_feat"])
            BinaryCode = torch.sign(h)
            if i == batch_num-1: 
                hashcode[i*test_batch_size:,:] = BinaryCode[:rem,:].data.cpu().numpy()
            else:
                hashcode[i*test_batch_size:(i+1)*test_batch_size,:] = BinaryCode.data.cpu().numpy()
        test_hashcode = np.matrix(hashcode)
        time1 = time.time()
        print ('retrieval costs: ',time1-time0)
        Hamming_distance = 0.5*(-np.dot(test_hashcode,test_hashcode.transpose())+nbits)
        time2 = time.time()
        print ('hamming distance computation costs: ',time2-time1)
        HammingRank = np.argsort(Hamming_distance, axis=0)
        time3 = time.time()
        print ('hamming ranking costs: ',time3-time2)
        records = open(results_path+'_epoch_'+str(epoch)+'.txt','w+')
        maps = []
        map_list = [5,10,20,40,60,80,100]
        for i in map_list:
            map,_,_ = mAP(sim_labels, HammingRank,i)
            maps.append(map)
            records.write('topK: '+str(i)+'\tmap: '+str(map)+'\n')
            print ('i: ',i,' map: ', map)
        records.close()


if __name__ == '__main__':
    pid = os.getpid()
    print(pid)
    model = MC_MLP(feature_size).to(device)
    models_name = '/MC_MLP'
    models_path = file_path + models_name
    results_path = result_path + models_name
    train()  
    eval()  

    