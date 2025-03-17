import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.loggers import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        # This fuction is responsible for data transformation.
        
        try:
            numerical_features = ["reading score","writing score"]
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            # Create a numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("standard scaler",StandardScaler())
                ]
            )
            # Create a categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("standard scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns standardized: {numerical_features}")

            logging.info(f"Categorical Columns encoding completed: {categorical_features}")

            # Transform the columns
            preprocess = ColumnTransformer(
                [
                    ("numerical_transform",num_pipeline,numerical_features),
                    ("categorical_transform",cat_pipeline,categorical_features)
                ]
            )

            return preprocess

        except Exception as e:
            raise CustomException(e,sys)
            

    def initiate_data_transformation(self,train_data_path,test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("The train and test data have been read from ingestion module.")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_col_name = "math score"
            numerical_columns = ["reading score","writing score"]

            input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df=train_df[target_col_name]

            input_feature_test_df=test_df.drop(columns=[target_col_name])
            target_feature_test_df=test_df[target_col_name]

            logging.info("Applying preprocessing_obj on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj,
                obj=preprocessing_obj
            )

            logging.info("Saved Preprocessing Object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj
            )

        except Exception as e:
            raise CustomException(e,sys)
            

