from torch import nn
from torch.nn import functional as F
from utils.args import *

class LRD(nn.Module):

    def __init__(self, inplanes, planes, cdiv=16):
        super(LRD, self).__init__()
        self.fc1=nn.Conv1d(inplanes,inplanes,kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Linear(planes, planes//cdiv, bias=False)
        self.bn1 = nn.BatchNorm1d(planes//cdiv)
        self.relu = nn.ReLU()
        self.deconv = nn.Linear(planes//cdiv, planes, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.sigmoid = nn.Sigmoid()
        self.fc2=nn.Conv1d(inplanes,inplanes,kernel_size=1)

        nn.init.normal_(self.conv.weight, 0, 0.001)
        nn.init.normal_(self.deconv.weight, 0, 0.001)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        bn, n, c = x.size()
        x = self.fc1(x)
        x = x.permute(0, 2, 1).contiguous()
        y = self.avg_pool(x).view(bn, c)
        y = self.conv(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y).view(bn, c, 1)
        x = x*y.expand_as(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc2(x)
        return x

class MRD(nn.Module):
    def __init__(self, inplanes, planes, cdiv=16, k=3, use_group=True):
        super(MRD, self).__init__()
        if use_group:
            group = planes//cdiv
        else:
            group = 1
        self.fc1=nn.Conv1d(inplanes,inplanes,kernel_size=1)
        self.avg_pool1 = nn.AdaptiveAvgPool1d((inplanes//k))
        self.conv = nn.Conv1d(planes, planes//cdiv, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
        self.bn1 = nn.BatchNorm1d(planes//cdiv)
        self.relu = nn.ReLU()
        self.deconv = nn.Conv1d(planes//cdiv, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool2 = nn.AdaptiveAvgPool1d((inplanes))
        self.fc2=nn.Conv1d(inplanes,inplanes,kernel_size=1)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.deconv.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
    
    def forward(self, x):
        bn, n, c = x.size()
        x = self.fc1(x)
        x = x.permute(0, 2, 1).contiguous()
        y = self.avg_pool1(x)
        y = self.conv(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y)
        y = self.avg_pool2(y)
        x = x*y
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc2(x)
        return x

class SRD(nn.Module):
    def __init__(self, inplanes, planes, cdiv=16, use_group=True):
        super(SRD, self).__init__()
        if use_group:
            group = planes//cdiv
        else:
            group = 1
        self.fc1=nn.Conv1d(inplanes,inplanes,kernel_size=1)
        self.conv = nn.Conv1d(planes, planes//cdiv, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
        self.bn1 = nn.BatchNorm1d(planes//cdiv)
        self.relu = nn.ReLU()
        self.deconv = nn.Conv1d(planes//cdiv, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.sigmoid = nn.Sigmoid()
        self.fc2=nn.Conv1d(inplanes,inplanes,kernel_size=1)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.deconv.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
    
    def forward(self, x):
        bn, n, c = x.size()
        x = self.fc1(x)
        x = x.permute(0, 2, 1).contiguous()
        y = self.conv(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y)
        x = x*y
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc2(x)
        return x



class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

class TokenMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.mlp = MLP(num_patches, expansion_factor, dropout)
        self.lrd = LRD(num_features, num_patches, r_min)
        self.mrd = MRD(num_features, num_patches, r_min,use_group=False)
        self.srd = SRD(num_features, num_patches, r_min,use_group=False)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x1 = self.lrd(x)
        x2 = self.mrd(x)
        x3 = self.srd(x)
        x = x1+x2+x3
        x = x.transpose(1,2)
        out = x + residual
        return out

class ChannelMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.mlp = MLP(num_features, expansion_factor, dropout)
        self.lrd = LRD(num_patches, num_features, r_max)
        self.mrd = MRD(num_patches, num_features, r_max)
        self.srd = SRD(num_patches, num_features, r_max)     

    def forward(self, x):
        residual = x
        x = self.mlp(x)
        x1 = self.lrd(x)
        x2 = self.mrd(x)
        x3 = self.srd(x)
        x = x1+x2+x3
        out = x + residual
        return out

class MixerLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.preproj = nn.Linear(feature_size, hidden_size)
        self.token_mixer = TokenMixer(
            num_features, num_patches, expansion_factor, dropout)
        self.channel_mixer = ChannelMixer(
            num_features, num_patches, expansion_factor, dropout)
        self.lrd = LRD(num_patches, num_features, r_max)
        self.mrd = MRD(num_patches, num_features, r_max)
        self.srd = SRD(num_patches, num_features, r_max)
        
    def forward(self, x):
        x = self.preproj(x)
        x = F.relu(x)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        x1 = self.lrd(x)
        x2 = self.mrd(x)
        x3 = self.srd(x)
        x = x1+x2+x3
        return x


