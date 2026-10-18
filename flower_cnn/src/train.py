import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import os
import time
from dataset import get_dataloaders
from model import MyModel

def train_model():
    DATA_ROOT=r"D:\A\lr2\flower_cnn\data\flower_copy"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备使用的计算设备是：{device}")

    batch_size=32
    num_epochs=20
    learning_rate=0.0001

    train_loader,val_loader,test_loader,num_classes,class_names=get_dataloaders(DATA_ROOT,batch_size)

    model=MyModel(num_classes=num_classes).to(device)

    criterion=nn.CrossEntropyLoss()

    optimizer=optim.Adam(model.parameters(),lr=learning_rate)

    train_losses=[]
    val_losses=[]
    train_accs=[]
    val_accs=[]

    best_val_acc=0.0

    for epoch in range(num_epochs):
        epoch_start_time=time.time()
        model.train()
        running_loss=0.0
        correct_train=0
        total_train=0

        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)

            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            _,predicted=torch.max(outputs.data,1)
            total_train+=labels.size(0)
            correct_train+=(predicted==labels).sum().item()

        train_loss=running_loss/len(train_loader)
        train_acc=correct_train/total_train

        model.eval()
        running_val_loss=0.0
        correct_val=0
        total_val=0

        with torch.no_grad():
            for images,labels in val_loader:
                images,labels=images.to(device),labels.to(device)
                outputs=model(images)
                loss=criterion(outputs,labels)

                running_val_loss+=loss.item()
                _,predicted=torch.max(outputs.data,1)
                total_val+=labels.size(0)
                correct_val+=(predicted==labels).sum().item()

        val_loss=running_val_loss/len(val_loader)
        val_acc=correct_val/total_val

        epoch_end_time=time.time()
        epoch_time=epoch_end_time-epoch_start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | " 
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc>best_val_acc:
            best_val_acc=val_acc
            torch.save(model.state_dict(),'src/best_model.pth')
            print(f"--> 保存最佳模型，准确率提升至{best_val_acc:.4f}")
    
    print("训练完成")

    print("\n==========开始测试集评估==========")
    model.load_state_dict(torch.load('src/best_model.pth'))
    model.eval()

    correct_test=0
    total_test=0

    with torch.no_grad():
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            total_test+=labels.size(0)
            correct_test+=(predicted==labels).sum().item()

    test_acc=correct_test/total_test
    print(f"最终测试集准确率：{test_acc*100:.2f}%")
    print("====================================")

    epochs_range=range(1,num_epochs+1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs_range,train_losses,label='Train Loss')
    plt.plot(epochs_range,val_losses,label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range,train_accs,label='Train Acc')
    plt.plot(epochs_range,val_accs,label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    os.makedirs('docs',exist_ok=True)
    plt.savefig('docs/traing_curves.png')
    print("图标已保存至 docs/traing_curves.png")

if __name__=='__main__':
    train_model()