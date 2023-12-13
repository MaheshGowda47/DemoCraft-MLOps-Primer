import os 
import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataIngestionCongfig:
    train_data_set: str=os.path.join("artifacts", "train.csv")
    test_data_set: str=os.path.join("artifacts", "test.csv")
    raw_data_set: str=os.path.join("artifacts", "data.csv")

class DataIngest:
    def __init__(self):
        self.ingestion_config = DataIngestionCongfig()

    def ingest_data(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r"C:\Users\aethe\Desktop\work\MLOPS\notebook\Data\stud.csv")
            data = pd.DataFrame(df)
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
    ob = DataIngest()
    ob.ingest_data()
