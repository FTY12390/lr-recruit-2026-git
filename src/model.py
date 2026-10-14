# src/model.py

import torch.nn as nn

class TitanicModel(nn.Module):

    def __init__(self,input_dim):
        super(TitanicModel,self).__init__()

        self.net=nn.Sequential(
            nn.Linear(input_dim,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x)