import torch
import torch.nn as nn
from collections import namedtuple
from .vgg import VGG19
from .transformation import GeometricTnf

vgg_multiplier = namedtuple("VggOutputs", ['relu1_2', 'relu2_2',
                                           'relu3_4', 'relu4_4', 'relu5_4'])
mutipliers = vgg_multiplier(1, 2, 4, 8, 16)


class WarpNet(nn.Module):
    def __init__(self,
                 vgg_layer='relu5_4',
                 corr_normalize=False, 
                 matching_type='correlation',
                 size=512,
                 output_theta=6,
                 fr_channels=[225, 128, 64, 32],
                 reg_normalization=False):
        
        super(WarpNet, self).__init__()
        
        # set corr out size connector 
        m = getattr(mutipliers, vgg_layer)
        self.corr_out_size = int((size/m)**2)

        # set conv layers for each steps
        self.ext = Extraction(vgg_layer=vgg_layer)
        self.cor = Correlation(normalization=corr_normalize, matching_type=matching_type)        
        self.reg = Regression(input_size=self.corr_out_size,
                              output_dim=output_theta,
                              channels=fr_channels,
                              normalization=reg_normalization)
        

    def forward(self, x, y):
        x, y = self.ext(x, y)
        corr = self.cor(x, y)
        theta = self.reg(corr)
        
        return theta


class Extraction(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """
    def __init__(self,
                 use_vgg: bool = True,
                 vgg_layer: str = 'relu4_4'):

        super(Extraction, self).__init__()

        if use_vgg:
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            # TODO 어떻게 gray scale 이미지로 할 수 있는 방법 없나
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 chennel images.'

            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)

        return x, y


class Correlation(torch.nn.Module):
    def __init__(self, normalization=True, matching_type='correlation'):
        super(Correlation, self).__init__()
        self.normalization = normalization
        self.matching_type = matching_type
        self.ReLU = nn.ReLU()
    
    def forward(self, x, y):
        b,c,h,w = x.size()
        if self.matching_type=='correlation':

            # reshape features for matrix multiplication
            x = x.transpose(2,3).contiguous().view(b,c,h*w)
            y = y.view(b,c,h*w).transpose(1,2)
            # perform matrix mult.
            mul = torch.bmm(y,x)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            corr = mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
            
            if self.normalization:
                corr = self.ReLU(corr)
                norm = (((corr ** 2).sum(1) + 1e-6) ** 0.5).unsqueeze(1).expand_as(corr)
                corr = corr / norm
        
            return corr

        if self.matching_type=='subtraction':
            return x - y
        
        if self.matching_type=='concatenation':
            return torch.cat((x,y),1)


class Regression(nn.Module):
    def __init__(self, input_size, output_dim=6, normalization=True, channels=[225,128,64,32]):
        super(Regression, self).__init__()
        num_layers = len(channels)
        # to make adaptive to input size change
        self.connector = nn.Sequential(
            nn.Conv2d(input_size, 225, kernel_size=3, padding=1),
            nn.BatchNorm2d(225, track_running_stats=False),
            nn.ReLU())

        nn_modules = list()
        for i in range(num_layers-1): # last layer is linear 
            ch_in = channels[i]
            ch_out = channels[i+1]            
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1))
            if normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out, track_running_stats=False))
            nn_modules.append(nn.ReLU())
        self.body = nn.Sequential(*nn_modules)

        # lienar map to theata output
        self.linear1 = nn.Linear(ch_out * input_size, 1024)
        self.linear2 = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = self.connector(x)
        x = self.body(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x