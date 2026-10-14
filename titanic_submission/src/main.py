from data_process import process_data
from dataset import get_dataloaders
from model import TitanicModel
from train import train_model
from predict import predict_new_passenger
import torch
import json

torch.manual_seed(42)

if __name__=='__main__':

    X_train,X_test,y_train,y_test,params=process_data('data/train.csv')

    train_loader,test_loader=get_dataloaders(X_train,y_train,X_test,y_test,batch_size=32)

    input_dim = X_train.shape[1]
    model = TitanicModel(input_dim=input_dim)

    print("\n===开始训练===")
    trained_model,train_losses,train_accs,test_accs=train_model(
        model,train_loader,test_loader,epochs=50,lr=0.001
    )

    print("\n===测试新样本预测===")

    new_passenger={
        'Pclass':3,
        'Sex':'male',
        'Age':22,
        'SibSp':1,
        'Parch':0,
        'Fare':7.25,
        'Embarked':'S'
    }

    result=predict_new_passenger(trained_model,new_passenger,params)
    print(f"新乘客信息：{new_passenger}")
    print(f"预测结果：{result}(0=遇难,1幸存)")
    
    print("\n===项目运行完毕！===")