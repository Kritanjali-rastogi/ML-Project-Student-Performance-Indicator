import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

# Data Transformation Configuration

@dataclass
class DataTransformationConfiguration:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# Data Transformation Class

class DataTransformation:

    # Constructor

    def __init__(self):
        self.data_transformation_config = DataTransformationConfiguration()

    # Function to get the preprocessor file object

    def get_preprocessor_obj(self):
        
        logging.info("Pipeline cration started")
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ['gender', 
                            'race_ethnicity', 
                            'parental_level_of_education', 
                            'lunch', 
                            'test_preparation_course']


            num_pipeline = Pipeline(steps=
                        [('imputer', SimpleImputer(strategy= 'median')),
                         ('Scaler', StandardScaler(with_mean=False))])

            cat_pipeline = Pipeline(steps= 
                        [
                            ('imputer', SimpleImputer(strategy= 'most_frequent')),
                            ('Encoder', OneHotEncoder()),
                            ('Scaler', StandardScaler(with_mean=False))

                        ])
            logging.info("Numerical and Categorical pipelines created")

            preprocessor = ColumnTransformer([('numerical_pipeline', num_pipeline, num_features),
                                              ('categorical_pipeline', cat_pipeline, cat_features)])
            
            logging.info("Preprocessor object created successfully")
            
            return preprocessor
        
        except Exception as e:
            logging.info("Error occured in preprocessor creation")
            raise CustomException(e,sys)


    # Function to initiate preprocessing

    def initiate_data_transformation(self, train_data_path, test_data_path):

        logging.info("Data transformation using preprocessor initiated")

        try:

            # Reading train and test data from artifacts folder

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Train and test data read from artifacts folder as dataframe")

            logging.info(f'Train Data Frame: {train_df.head().to_string()}')
            logging.info(f'Train Data Frame: {test_df.head().to_string()}')

            # Segregating independent and target features

            logging.info("Segregating independent and target features started")

            target_column = 'math_score'

            train_df_independent_features = train_df.drop(columns= [target_column], axis =1)
            train_df_target_feature = train_df[target_column]

            test_df_independent_features = test_df.drop(columns= [target_column], axis =1)
            test_df_target_feature = test_df[target_column]

            logging.info("Segregating independent and target features completed")

            # Calling preprocessor object and applyting required transformations on train and test data

            logging.info("Calling preprocessor onject")

            preprocesor_obj = self.get_preprocessor_obj()

            train_df_independent_features_arr = preprocesor_obj.fit_transform(train_df_independent_features)
            test_df_independent_features_arr = preprocesor_obj.transform(test_df_independent_features)

            logging.info("Preprocessing transformation applied on train and test independent features")

             # Getting back train and test array

            logging.info("Getting back the train and test array")

            train_arr = np.c_[train_df_independent_features_arr, np.array(train_df_target_feature)]
            test_arr = np.c_[test_df_independent_features_arr, np.array(test_df_target_feature)]

            logging.info("Data Transformation completed successfully")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocesor_obj 
            )
            
            logging.info("Preprocessor object saved")

            return (train_arr, 
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)    
            
        except Exception as e:
            logging.info("Error occured in data ingestion")
            raise CustomException(e,sys)