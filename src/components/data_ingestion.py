import os 
import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
import pandas as pd

from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_train import ModelTrainerConfig
from src.components.model_train import ModelTrainer

@dataclass
class DataIngestionCongfig:
    train_data_set: str=os.path.join("artifacts", "train.csv")
    test_data_set: str=os.path.join("artifacts", "test.csv")
    raw_data_set: str=os.path.join("artifacts", "data.csv")

class DataIngest:
    def __init__(self):
        self.ingestion_config = DataIngestionCongfig()

    def initiate_ingest_data(self):
        logging.info("Entered the data ingestion method or component")
        try:
            data = pd.read_csv(r"C:\Users\aethe\Desktop\work\MLOPS\notebook\Data\stud.csv")
            # data = pd.DataFrame(df)
            logging.info("Read the data as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_set), exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_set, index=False, header=True)
            logging.info("Raw data is inisiated")
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_set, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_set, index=False, header=True)
            logging.info("Train_set and test_set are insiated in artifacts")

            return(
                self.ingestion_config.train_data_set,
                self.ingestion_config.test_data_set
            )
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngest()
    train_data, test_data = obj.initiate_ingest_data()

    logging.info("Data Transoformation")
    data_trans = DataTransformation()
    train_arr, test_arr = data_trans.initiate_data_transformation(train_data, test_data)

    ModelTrainer = ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_arr, test_arr))