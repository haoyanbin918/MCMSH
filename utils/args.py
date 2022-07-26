import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = 'fcv'  # type=str, 'yfcc | fcv |
workers = 0 # type=int, number of data loading workers, default=0
batch_size = 256 
num_epochs = 55 
test_batch_size = 256
use_cuda = True
use_checkpoint = False
lr = 3e-4 #type=float, default=0.0001
lr_decay_rate = 20 #type=float, default=30
single_lr_decay_rate = 20 #type=float, default=30
weight_decay = 1e-4 #type=float, default=1e-4
nbits = 64 #hash code lengths
feature_size = 4096
max_frames = 25
hidden_size = 256
r_max=16 #reduction ratio in ChannelMixer
r_min=4 #reduction ratio in TokenMixer

data_root = '/data/fcv'  #to save large data files
home_root = '/home/MCMSH'
file_path = home_root + '/models/fcv_bits_' + str(nbits)
result_path = home_root + '/results/fcv_bits_' + str(nbits)

train_feat_path = data_root+'/fcv_train_feats.h5'
test_feat_path = data_root+'/fcv_test_feats.h5'
label_path = data_root+'/fcv_test_labels.mat'






