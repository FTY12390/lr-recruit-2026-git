import torch
from torch import nn
from torch import optim
from torchvision import models
from dataset import get_dataloaders
import time

def train_pretrained():
    device=torch.device("cuda" if torch.cuda.is_available() else  "cpu")

    DATA_ROOT=r"D:\A\lr2\flower_cnn\data\flower_copy"
    train_loader,val_loader,_,num_classes,_=get_dataloaders(DATA_ROOT,batch_size=32)

    model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad=False

    num_ftrs=model.fc.in_features
    model.fc=nn.Linear(num_ftrs,num_classes)
    model=model.to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.fc.parameters(),lr=0.001)

    num_epochs=5
    for epoch in range(num_epochs):
        model.train()
        start_time=time.time()
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct,total=0,0
        with torch.no_grad():
            for images,labels in val_loader:
                images,labels=images.to(device),labels.to(device)
                _,pred=torch.max(model(images),1)
                total+=labels.size(0)
                correct+=(pred==labels).sum().item()

        print(f"Epoch: {epoch+1}/{num_epochs} | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"ResNet18 Val Acc: {correct/total:.4f}")

if __name__=='__main__':
    train_pretrained()