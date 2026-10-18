import torch
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import datasets
from data_process import train_transform,val_test_transform

class SubsetWithTransform(Dataset):
    def __init__(self,subset,transform):
        self.subset=subset
        self.transform=transform

    def __getitem__(self, index):
        x,y=self.subset[index]
        if self.transform:
            x=self.transform(x)
        return x,y

    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_root,batch_size=32):
    full_train_dataset=datasets.ImageFolder(root=f'{data_root}/train',transform=None)
    num_classes=len(full_train_dataset.classes)
    class_names=full_train_dataset.classes

    train_size=int(0.8*len(full_train_dataset))
    val_size=len(full_train_dataset)-train_size
    train_subset,val_subset=random_split(
        full_train_dataset,[train_size,val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset=SubsetWithTransform(train_subset,transform=train_transform)
    val_dataset=SubsetWithTransform(val_subset,transform=val_test_transform)
    test_dataset=datasets.ImageFolder(root=f'{data_root}/test',transform=val_test_transform)

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)

    print(f"类别数：{num_classes}")
    print(f"训练集：{len(train_dataset)},验证集：{len(val_dataset)},测试集：{len(test_dataset)}")
    return train_loader,val_loader,test_loader,num_classes,class_names

if __name__=='__main__':
    DATA_ROOT=r'D:\A\lr2\flower_cnn\data\flower_copy'

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'当前使用的计算设备是：{device}')

    train_loader,val_loader,test_loader,num_classes,class_names=get_dataloaders(DATA_ROOT)

    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        print(f'图像已成功移至GPU:{images.device}')
        print(f'图像形状：{images.shape}')
        print(f'标签形状：{labels.shape}')
        break
































