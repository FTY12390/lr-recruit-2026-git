import torch
from torch import nn

class MyModel(nn.Module):

    def __init__(self,num_classes=5):
        super(MyModel,self).__init__()

        self.conv1=nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2=nn.Conv2d(96,256,kernel_size=5,padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3=nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4=nn.Conv2d(384,384,kernel_size=3,padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5=nn.Conv2d(384,256,kernel_size=3,padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool=nn.MaxPool2d(kernel_size=3,stride=2)

        self.relu=nn.ReLU()

        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,num_classes)
        )

    def forward(self,x):
        x=self.pool(self.relu(self.bn1(self.conv1(x))))
        x=self.pool(self.relu(self.bn2(self.conv2(x))))
        x=self.relu(self.bn3(self.conv3(x)))
        x=self.relu(self.bn4(self.conv4(x)))
        x=self.pool(self.relu(self.bn5(self.conv5(x))))
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x


if __name__=='__main__':
    model=MyModel(num_classes=5)
    x=torch.randn(32,3,224,224)
    print(model(x.shape))