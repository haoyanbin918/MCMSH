import torch
import torch.nn as nn
from torch.autograd import Function
from .MC_MLP import MixerLayer
from utils.args import *
import math

class LBSign(Function):
    
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

class MC_MLP(nn.Module):

    def __init__(self, frame_size):
        super(MC_MLP,self).__init__()
        self.mc_mlp = MixerLayer(num_features=hidden_size, num_patches=max_frames, expansion_factor=2, dropout=0.5)
        self.binary = nn.Linear(hidden_size, nbits)
        self.activation = nn.Tanh()  
        self.alpha=1.0
        self.sign=self.sign_function
        

    def sign_function(self,x):
       return LBSign.apply(x)
    
    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

    def forward(self, x):
        
        e = self.mc_mlp(x)
        y = self.binary(e)
        h = self.activation(self.alpha*y)
        h = torch.mean(h,1)
        b=self.sign(h)
        return b,h,e

    

