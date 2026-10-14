# src/predict.py

import torch
import pandas as pd
import json
from model import TitanicModel

def predict_new_passenger(model,passenger_info,preprocess_params):
    df=pd.DataFrame([passenger_info])

    if 'Age' not in df.columns or pd.isnull(df['Age']).any():
        df['Age']=df["Age"].fillna(preprocess_params['age_mean'])
    if 'Embarked' not in df.columns or pd.isnull(df['Embarked']).any():
        df['Embarked']=df["Embarked"].fillna(preprocess_params['embarked_mode'])
    if 'Fare' not in df.columns or pd.isnull(df['Fare']).any():
        df['Fare']=df['Fare'].fillna(0)

    df['FamilySize']=df['SibSp']+df['Parch']+1

    df['Sex']=df['Sex'].map(preprocess_params['sex_map'])
    df['Embarked']=df['Embarked'].map(preprocess_params['embarked_map'])

    df['Age']=(df['Age']-preprocess_params['age_mean_norm'])/preprocess_params['age_std_norm']
    df['Fare']=(df['Fare']-preprocess_params['fare_mean_norm'])/preprocess_params['fare_std_norm']

    df=df.drop(['SibSp','Parch'],axis=1)
    df=df[preprocess_params['feature_cols']]
    input_tensor=torch.tensor(df.values,dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output=model(input_tensor)
        prediction=(output>=0.5).float().item()

    return int(prediction)














































