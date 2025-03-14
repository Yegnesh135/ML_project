import os
import sys
from src.exception import CustomException
from src.loggers import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig: #This is to create the address where the data has to be saved 
    train_data_path: str= os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion Object or Method")
        try:
            df = pd.read_csv("notebook\data\StudentsPerformance.csv") #This is to get the data from data source and load it in a folder in the project file.
            logging.info("The data read as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #This is the condition to make a folder with the name obtained in the code snippet
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #The data loaded into dataframe is being saved in the project folder 
            
            logging.info("Train Test Split intiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42) #Split the data 

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion() 
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
