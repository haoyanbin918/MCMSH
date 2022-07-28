import h5py
import torch
import copy
import torch.utils.data as data
import numpy as np
import random
from utils.args import *


class TrainDataset(data.Dataset):

    def __init__(self, feature_path):
        
        with h5py.File(feature_path,'r') as h5_file:
          self.video_feats = h5_file['feats'][:]   
        with h5py.File(data_root+'/sim_matrix.h5','r') as h5_file:
          self.sim_matrix = h5_file['adj'][:]   
        with h5py.File(home_root+'/data/pseudo_label.h5','r') as h5_file:
          self.label = h5_file['pos'][:]    
        with h5py.File(home_root+'/data/cluster_pca_256.h5','r') as h5_file:
          self.cluster = h5_file['feats'][:]    
    
    def random_frame(self, video):
        videos=video
        for i,_ in enumerate(video):
            prob = random.uniform(0,1)
            if prob < 0.15:#15% mask
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    videos[i] = video[i]*0

                # 10% randomly change token to random token
                elif prob < 0.9:
                    xx = np.random.randint(0,self.video_feats.shape[0],size=1)
                    yy = np.random.randint(0,self.video_feats.shape[1],size=1)
                    videos[i] = self.video_feats[xx[0],yy[0],:]
        return videos
        
    def random_video(self, index):
        if random.random() > 0.5:
            # get a similar video
            index2 = self.get_neighbor_video(index)
            return index2, 1.0
        else:
            # get a dissimilar video
            index2 = self.get_random_video(index)
            return index2, -1.0

    def return_video(self, index):
        # get a similar video
        similar1 = self.get_neighbor_video(index)
        # get a dissimilar video
        dissimilar1 = self.get_random_video(index)
        dissimilar2 = self.get_random_video(index)
        if self.sim_matrix[dissimilar2,dissimilar1]==1.:
            dissimilar2 = self.get_random_video(index)
        return similar1,dissimilar1,dissimilar2

    def get_neighbor_video(self, index):
        neighbors = np.where(self.sim_matrix[index]==1)
        neighbors=neighbors[0]
        ranind = np.random.randint(0,len(neighbors),size=1)
        return neighbors[ranind[0]]

    def get_random_video(self, index):
        # dissimilar videos have middle distance from the query video (tag is 2)
        others = np.where(self.sim_matrix[index]==2)#2
        others=others[0]
        # If there is no tag 2 video, we get dissimilar videos with tag 0
        if len(others)==0:
            others = np.where(self.sim_matrix[index]==0)
            others=others[0]
        ranind = np.random.randint(0,len(others),size=1)
        return others[ranind[0]]

    
    def __getitem__(self, item):
        item2, is_similar = self.random_video(item)
        similar1,dissimilar1,dissimilar2 = self.return_video(item)
        t1 = self.video_feats[item]
        t2 = self.video_feats[item2]
        t3 = self.video_feats[similar1]
        t4 = self.video_feats[dissimilar1]
        t5 = self.video_feats[dissimilar2]
        t1 = self.random_frame(t1)
        t2 = self.random_frame(t2)
        double_video_feat = np.concatenate((t1,t2),0)
        triple_video_feat = np.concatenate((t3,t4,t5),0)
        neighbor1 = torch.from_numpy(self.cluster[self.label[item][0]])
        neighbor2 = torch.from_numpy(self.cluster[self.label[item2][0]])
        
        output = {"double_video_feat": double_video_feat,
                  "triple_video_feat": triple_video_feat,
                  "is_similar": is_similar,
                  "n1":neighbor1,
                  "n2":neighbor2,
                  }

        return {key: torch.as_tensor(value) for key, value in output.items()}

    def __len__(self):
        return len(self.video_feats)

class TestDataset(data.Dataset):

    def __init__(self, feature_path):
        with h5py.File(feature_path,'r') as h5_file:
          self.video_feats = h5_file['feats'][:]   


    def __getitem__(self, item):

        video_feat = self.video_feats[item]

        output = {"video_feat": video_feat}

        return {key: torch.tensor(value) for key, value in output.items()}

    def __len__(self):
        return len(self.video_feats)


def get_train_loader(feature_path,batch_size=10, shuffle=True, num_workers=1, pin_memory=True):
    v = TrainDataset(feature_path)
    data_loader = torch.utils.data.DataLoader(dataset=v,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              #collate_fn=train_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


def get_eval_loader(feature_path, batch_size=256, shuffle=False, num_workers=1, pin_memory=False):
    vd = TestDataset(feature_path)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              #collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader

