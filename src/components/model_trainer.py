import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_model, save_object
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error   


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            logging.info("Splitting train and test input data into dependent and independent features")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
             
            logging.info("Data segreggated into independent and target features")

            models = {'Linear Regression': LinearRegression(),
                      'Lasso': Lasso(),
                      'Ridge': Ridge(),
                      'ElasticNet': ElasticNet(),
                      'Decision_Tree': DecisionTreeRegressor(),
                      'SVM': SVR(),
                      'KNeighborsRegressor': KNeighborsRegressor(),
                      'RandomForest': RandomForestRegressor(),
                      'AdaBoost': AdaBoostRegressor(),
                      'XGB': XGBRegressor(),
                      'CatBoost': CatBoostRegressor()}
            
            logging.info("Model evaluation started")
            
            params = {
                'Linear Regression': {}, 
                'Lasso': {}, 
                'Ridge': {}, 
                'ElasticNet': {},
                'Decision_Tree': {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                  'splitter':['best','random'],'max_features':['sqrt','log2']},
                'SVM': {},
                'KNeighborsRegressor': {'weights': ['uniform', 'distance'],
                                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                "RandomForest":{'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                'max_features':['sqrt','log2',None],
                                'n_estimators': [8,16,32,64,128,256]},
                'AdaBoost':{'learning_rate':[.1,.01,0.5,.001],
                            'loss':['linear','square','exponential'],
                            'n_estimators': [8,16,32,64,128,256]},
                "XGB":{'learning_rate':[.1,.01,.05,.001],
                       'n_estimators': [8,16,32,64,128,256]},
                'CatBoost':{'depth': [6,8,10],
                                         'learning_rate': [0.01, 0.05, 0.1],
                                         'iterations': [30, 50, 100]}}

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models = models, params=params)

            # Best Score

            logging.info("Model evaluation completed")

            best_model_score = max(sorted(model_report.values()))

            # Best Model Name

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # Best Modal

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")

            logging.info("Best model Found")
            print(f'The best model is {best_model_name} with a r2 acore of {best_model_score}')
            logging.info(f'The best model is {best_model_name} with a r2 acore of {best_model_score}')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model)

            logging.info("Best model saved as pickle file")

            y_pred = best_model.predict(X_test)

            accuracy_score = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            logging.info(f'Accuracy: {accuracy_score:.4f}, Mean Abosulte Error: {mae:.4f}, Mean Squared Error: {mse:.4f}')

            return accuracy_score

        except Exception as e:
            logging.info("Error occured in preprocessor creation")
            raise CustomException(e,sys)


        
