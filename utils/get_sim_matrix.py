import h5py
from calculate_neighbors import *
from args import *

'''
We set tag=1 for closest pairs(similar),
tag=2 for pairs with middle distances(dissimilar),
tag = 0 for other cases (we don't care)
'''
with h5py.File(train_feat_path,'r') as h5_file:
    video_feats = h5_file['feats'][:] 
video_feats=np.mean(video_feats,1)
with h5py.File(home_root+'/data/cluster.h5','r') as h5_file:
    clusters = h5_file['feats'][:]   

Z,_,pos1 = ZZ(video_feats, clusters, 3, None)
s = np.asarray(Z.sum(0)).ravel()
isrl = np.diag(np.power(s, -1)) 
# isrl = inverse square root of lambda
Adj = np.dot(np.dot(Z,isrl),Z.T)
SS1 = (Adj>0.00001).astype('float32')

Z,_,pos1 = ZZ(video_feats, clusters, 4, None)
s = np.asarray(Z.sum(0)).ravel()
isrl = np.diag(np.power(s, -1)) 
Adj = np.dot(np.dot(Z,isrl),Z.T)
SS2 = (Adj>0.00001).astype('float32')

Z,_,pos1 = ZZ(video_feats, clusters, 5, None)
s = np.asarray(Z.sum(0)).ravel()
isrl = np.diag(np.power(s, -1))
Adj = np.dot(np.dot(Z,isrl),Z.T)
SS3 = (Adj>0.00001).astype('float32')

SS4 = SS3-SS2
SS5 = 2*SS4+SS1

hh5 = h5py.File(data_root+'/sim_matrix.h5', 'w')
hh5.create_dataset('adj', data = SS5)
hh5.close()
print("get sim_matrix done")
