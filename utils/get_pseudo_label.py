import h5py
from calculate_neighbors import *
from args import *

'''
For each video, search several neighbors from the anchor set and save them in a file.
To save space, we only save the index of them.
The nearest anchor is a pseudo label of the video.
'''
with h5py.File(train_feat_path,'r') as h5_file:
    video_feats = h5_file['feats'][:] 
video_feats=np.mean(video_feats,1)
with h5py.File(home_root+'/data/cluster.h5','r') as h5_file:
    clusters = h5_file['feats'][:]   

Z1,_,pos1 = ZZ(video_feats, clusters, 3, None)
h5 = h5py.File(home_root+'/data/pseudo_label.h5', 'w')
h5.create_dataset('pos', data = pos1)
h5.close()
print("get pseudo label done")