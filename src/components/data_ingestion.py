import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_Path: str =os.path.join("artifact","train.csv")
    test_data_Path: str =os.path.join("artifact","test.csv")
    raw_data_Path: str =os.path.join("artifact","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df =pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_Path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_Path, index=False, header=True)
            logging.info("Train test split initiated")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_Path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_Path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_Path,
                self.ingestion_config.test_data_Path,
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    logging.info("Data ingestion completed successfully")

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path, test_data_path)