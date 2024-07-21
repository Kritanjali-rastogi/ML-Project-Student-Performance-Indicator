import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
            logging.info("Error occured in saving object")
            raise CustomException(e,sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    
    try:
         
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]                  # Selecting model
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            model.fit(X_train, y_train)         # Fitting the model

            y_pred = model.predict(X_test)      # Predict using model

            model_accuracy_score = r2_score(y_test, y_pred)  # Finding model accuracy score

            report[list(models.keys())[i]] = model_accuracy_score

        return report
    
    except Exception as e:
            logging.info("Error occured in model evaluation")
            raise CustomException(e,sys)