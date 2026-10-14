# src/data_process.py

import pandas as pd
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 读取并处理数据
def process_data(file_path='data/train.csv'):
    print("正在读取数据")

    df=pd.read_csv(file_path).copy()

    drop_cols=['PassengerId','Name', 'Ticket', 'Cabin']
    df=df.drop(drop_cols,axis=1)

    X=df.drop('Survived',axis=1).copy()
    y=df['Survived'].copy()

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )
    print(f"划分完成->训练集X:{X_train.shape},测试集X:{X_test.shape}")

    preprocess_params={}

    age_mean=X_train['Age'].mean()
    embarked_mode=X_train['Embarked'].mode()[0]

    preprocess_params['age_mean']=float(age_mean)
    preprocess_params['embarked_mode']=str(embarked_mode)

    for dataset in [X_train,X_test]:
        dataset['Age']=dataset['Age'].fillna(age_mean)
        dataset['Embarked']=dataset['Embarked'].fillna(embarked_mode)

    X_train['FamilySize']=X_train['SibSp']+X_train['Parch']+1
    X_test['FamilySize']=X_test['SibSp']+X_test['Parch']+1

    X_train=X_train.drop(['SibSp','Parch'],axis=1)
    X_test=X_test.drop(['SibSp','Parch'],axis=1)

    sex_map={'male':0,'female':1}
    embarked_map={'C':0,'Q':1,'S':2}

    for dataset in [X_train,X_test]:
        dataset['Sex']=dataset['Sex'].map(sex_map)
        dataset['Embarked']=dataset['Embarked'].map(embarked_map)

    preprocess_params['sex_map']=sex_map
    preprocess_params['embarked_map']=embarked_map

    age_mean_norm=X_train['Age'].mean()
    age_std_norm=X_train['Age'].std()
    fare_mean_norm=X_train['Fare'].mean()
    fare_std_norm=X_train['Fare'].std()

    preprocess_params['age_mean_norm']=float(age_mean_norm)
    preprocess_params['age_std_norm']=float(age_std_norm)
    preprocess_params['fare_mean_norm']=float(fare_mean_norm)
    preprocess_params['fare_std_norm']=float(fare_std_norm)

    for dataset in [X_train,X_test]:
        dataset['Age']=(dataset['Age']-age_mean_norm)/age_std_norm
        dataset['Fare']=(dataset['Fare']-fare_mean_norm)/fare_std_norm

    preprocess_params['feature_cols']=list(X_train.columns)

    with open('data/preprocess_params.json','w') as f:
        json.dump(preprocess_params,f,indent=4)

    print("缺失值处理、特征工程、类别转换、数值标准化全部完成！")
    print("预处理参数已成功保存至data/preprocess_params.json")
    print(f"最终训练集形状：{X_train.shape}")

    return X_train,X_test,y_train,y_test,preprocess_params

# 测试
if __name__ == '__main__':
    # 假设在项目根目录 titanic_submission 下运行
    X_train,X_test,y_train,y_test,params=process_data('data/train.csv')
    
    print("\n=== 训练集前5行数据预览 ===")
    print(X_train.head())
    print("\n=== 预处理参数字典 ===")
    print(params)
