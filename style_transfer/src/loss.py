import torch
from torch import nn
import torch.nn.functional as F
from .utils import gram_matrix

class StyleTransferLoss(nn.Module):
    def __init__(self,content_weight=1.0,style_weight=1e6,tv_weight=1e-6):
        super(StyleTransferLoss,self).__init__()
        self.content_weight=content_weight
        self.style_weight=style_weight
        self.tv_weight=tv_weight

    def forward(self,target_c_feats,target_s_feats,content_feats,style_feats):
        c_loss=F.mse_loss(target_c_feats['conv4_2'],content_feats['conv4_2'].detach())
        s_loss=0
        for layer in style_feats.keys():
            target_grm=gram_matrix(target_s_feats[layer])
            style_grm=gram_matrix(style_feats[layer].detach())
            s_loss+=F.mse_loss(target_grm,style_grm)

        total_loss=self.content_weight*c_loss+self.style_weight*s_loss

        return total_loss,c_loss,s_loss

def tv_loss(img,tv_weight=1e-6):
    w_variance=torch.sum(torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]))
    h_variance=torch.sum(torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]))

    loss=tv_weight*(h_variance+w_variance)
    return loss