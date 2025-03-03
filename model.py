import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingRegressor 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cleaning_data import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

def loading_training():
    df = cleaning_df()
    
    #Selecting input
    x_features = df.drop(columns= ['Primary energy consumption per capita (kWh/person)', 'Value_co2_emissions_kt_by_country','Entity','Year'] , axis=1)
    
    #Selecting target (output) for Energy consumption model
    df['Primary energy consumption per capita (kWh/person)'] = df['Primary energy consumption per capita (kWh/person)'].astype('int')
    y_energy = df[['Primary energy consumption per capita (kWh/person)']]
    
    X_train,X_test ,y_train, y_test = train_test_split(x_features, y_energy ,test_size=0.3, random_state=0)

    
    print('starting training')

    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Train the model
    #lr = LogisticRegression(**params)
    #lr.fit(X_train, y_train)
    regr = BaggingRegressor(n_estimators=10, random_state=0).fit(X_train, y_train)

    # Predict on the test set
    y_pred = regr.predict(X_test)
    

    # Calculate metrics
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return rmse, params, X_train, y_train, regr, X_test, y_test,r2
