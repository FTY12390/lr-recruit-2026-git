import torch
import torch.optim as optim
from .loss import StyleTransferLoss,tv_loss
import torchvision.utils as vutils
import os
from .utils import unnormalize

class StyleTransferTrainer:
    def __init__(self,model,content_img,style_img,device='cuda',
                 content_weight=1.0,style_weight=8e5,tv_weight=1e-6,lr=0.01):
        self.model=model
        self.content_img=content_img
        self.style_img=style_img
        self.deevice=device

        self.criterion=StyleTransferLoss(content_weight,style_weight)
        self.tv_weight=tv_weight

        with torch.no_grad():
            self.content_feats,_=self.model(self.content_img)
            _,self.style_feats=self.model(self.style_img)

        self.target_img=self.content_img.clone().requires_grad_(True)

        self.optimizer=optim.Adam([self.target_img],lr=lr)

    def train(self,steps=1000,print_every=50,save_dir='data/outputs'):
        print(f"开始训练，总步数：{steps}")
        os.makedirs(save_dir,exist_ok=True)

        for step in range(1,steps+1):
            self.optimizer.zero_grad()

            target_comtent_feats,target_style_feats=self.model(self.target_img)

            _,c_loss,s_loss=self.criterion(target_comtent_feats,target_style_feats,self.content_feats,self.style_feats)

            t_loss=tv_loss(self.target_img,self.tv_weight)

            total_loss=self.criterion.content_weight*c_loss+self.criterion.style_weight*s_loss+t_loss

            total_loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.target_img.clamp_(-3.0,3.0)

            if step%print_every==0 or step==1:
                print(f"Step[{step}/{steps}] | Total Loss:{total_loss.item():.4f}| "
                      f"Content:{c_loss.item():.4f} | Style:{s_loss.item():.4f} | "
                      f"TV:{t_loss.item():.4f}")

                save_img=unnormalize(self.target_img)
                vutils.save_image(save_img,os.path.join(save_dir,f"step_{step}.jpg"))
