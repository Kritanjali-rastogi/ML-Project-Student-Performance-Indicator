import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pickle
from src.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger import logging
from src.exception import CustomException

class ModelEvaluation:

    def __init__(self):
        logging.info('Evaluation started')

    def eval_metrics(self, actual, predicted):

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        logging.info('Evaluation metrics captured')
        return (rmse, mae, r2)
    
    def initiate_model_evaluation(self, train_arr, test_arr):
        try:
            X_train = train_arr[:,:-1]
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            model_path = os.path.join('artifacts', 'model.pkl')
            model = load_object(model_path)

            #mlflow.set_registry_uri("")

            logging.info('Model registered')

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                prediction = model.predict(X_test)
                signature = infer_signature(X_train, prediction)

                (rmse, mae, r2) = self.eval_metrics(y_test,prediction)

                mlflow.log_metric('rmse', rmse)
                mlflow.log_metric('mae', mae)
                mlflow.log_metric('r2', r2)

                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(model, "model", registered_model_name = 'Ridge', signature= signature)
                else:
                    mlflow.sklearn.log_model(model, 'model', signature=signature)

        except Exception as e:
            logging.info("Error occured in model evaluation")
            raise CustomException(e,sys)