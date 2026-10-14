# src/dataset.py

import torch
from torch.utils.data import Dataset,DataLoader

class TitanicDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X.values,dtype=torch.float32)
        self.y=torch.tensor(y.values,dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]


def get_dataloaders(X_train,y_train,X_test,y_test,batch_size=32):
    train_dataset=TitanicDataset(X_train,y_train)
    test_dataset=TitanicDataset(X_test,y_test)

    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader=DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader,test_loader
