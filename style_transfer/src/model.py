import torch
from torch import nn
from torchvision.models import vgg19,VGG19_Weights

class VGG19Features(nn.Module):
    def __init__(self):
        super(VGG19Features,self).__init__()

        vgg=vgg19(weights=VGG19_Weights.DEFAULT).features

        for param in vgg.parameters():
            param.requires_grad_(False)

        vgg.eval()

        self.features=nn.Sequential()
        for idx,layer in enumerate(vgg):
            self.features.add_module(str(idx),layer)
            if idx==28:
                break

        self.content_layers=[21]
        self.style_layers=[0,5,10,19,28]

    def forward(self,x):
        content_feats={}
        style_feats={}

        for name,layer in self.features.named_children():
            x=layer(x)
            idx=int(name)

            if idx in self.content_layers:
                content_feats['conv4_2']=x
            if idx in self.style_layers:
                style_names={0:'conv1_1',5:'conv2_1',10:'conv3_1',19:'conv4_1',28:'conv5_1'}
                style_feats[style_names[idx]]=x

        return content_feats,style_feats
