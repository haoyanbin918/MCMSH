import os
import torch
import h5py
import time
import numpy as np
from utils.args import *
import scipy.io as sio
from tools import Array,mAP
from data import get_eval_loader
from model import MC_MLP

pid = os.getpid()
print(pid)
models_name = '/MC_MLP'
models_path = file_path + models_name
results_path = result_path + models_name

if not os.path.exists(result_path):
    os.makedirs(result_path)

model = MC_MLP(feature_size).to(device)

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

model.load_state_dict(torch.load(models_path+'.pth'))
model.eval()

time0 = time.time()
for i, data in enumerate(eval_loader): 
    data = {key: value.to(device) for key, value in data.items()}
    _,h,_ = model.forward(data["video_feat"])
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

records = open(results_path+'.txt','w+')
maps = []
map_list = [5,10,20,40,60,80,100]
for i in map_list:
    map,_,_ = mAP(sim_labels, HammingRank,i)
    maps.append(map)
    records.write('topK: '+str(i)+'\tmap: '+str(map)+'\n')
    print ('i: ',i,' map: ', map)
records.close()



