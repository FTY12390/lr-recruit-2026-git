# src/train.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def train_model(model,train_loader,test_loader,epochs=50,lr=0.001):
    os.makedirs('docs',exist_ok=True)

    criterion=nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)

    train_losses=[]
    train_accs=[]
    test_accs=[]

    print("开始训练")
    for epoch in range(epochs):
        model.train()
        running_loss=0.0
        correct_train=0
        total_train=0

        for batch_X,batch_y in train_loader:
            optimizer.zero_grad()
            outputs=model(batch_X)
            loss=criterion(outputs,batch_y)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()*batch_X.size(0)

            predicted=(outputs>=0.5).float()
            correct_train+=(predicted==batch_y).sum().item()
            total_train+=batch_y.size(0)

        epoch_train_loss=running_loss/total_train
        epoch_train_acc=correct_train/total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        correct_test=0
        total_test=0

        with torch.no_grad():
            for batch_X,batch_y in test_loader:
                outputs=model(batch_X)
                predicted=(outputs>=0.5).float()
                correct_test+=(predicted==batch_y).sum().item()
                total_test+=batch_y.size(0)

        epoch_test_acc=correct_test/total_test
        test_accs.append(epoch_test_acc)

        if (epoch+1)%10==0 or epoch==0:
            print(f"Epoch[{epoch+1}/{epochs}] |" 
                  f"Train Loss:{epoch_train_loss:.4f} | "
                  f"Train Acc: {epoch_train_acc:.4f} | "
                  f"Test Acc: {epoch_test_acc:.4f}")

    print("训练完成")

    torch.save(model.state_dict(),'src/titanic_model.pth')
    
    print("正在生成可视化图表")

    plt.figure(figsize=(8,5))
    plt.plot(range(1,epochs+1),train_losses,label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/loss_curve.png')
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(range(1,epochs+1),train_accs,label='Train Accuracy')
    plt.plot(range(1,epochs+1),test_accs,label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/accuracy_curve.png')
    plt.close()

    print("突变已成功保存至 doce/loss_curve.png和docs/accuracy_curve.png")

    return model,train_losses,train_accs,test_accs
