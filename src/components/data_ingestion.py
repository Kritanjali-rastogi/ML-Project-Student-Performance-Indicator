import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


# Data Ingestion Configuration

@dataclass
class DataIngestionConfiguration:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


# Data Ingestion Class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfiguration()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            df = pd.read_csv('notebooks/Data/stud.csv')
            logging.info("Data Read as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)

            df.to_csv(self.ingestion_config.raw_data_path, header= True, index= False)

            # Applying train-test split

            logging.info("Train test split began")

            train_set, test_set = train_test_split(df,test_size= 0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, header = True, index = True)
            test_set.to_csv(self.ingestion_config.test_data_path, header = True, index = True)

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        
            logging.info("Data ingestion completed successfully")

        except Exception as e:
            logging.info("Error occured in data tarnsformation")
            raise CustomException(e,sys)