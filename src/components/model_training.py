import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.loggers import logging
from src.utils import save_obj,evaluate_model


@dataclass
class ModelTrainerConfig():
    trained_model_path = os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trianer_config = ModelTrainerConfig()

    def initiate_model_training(self,training_array,test_array):
        try:
            logging.info("Splitting Training and Test input data")
            X_train,y_train,X_test,y_test = (
                training_array[:,:-1],
                training_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            ) #Here we take the train and test data which has been processed in transfromation part and making split as 
            # input training features and input target features along with the similar split in test data obtained from transformation step.

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbor Regressor": KNeighborsRegressor(),
                "Catboosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # To get best model score
            best_model_score = max(sorted(model_report.values()))

            # To get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # To get the model
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best model Found")
            logging.info(f"Best found model on both training and testing dataset.")

            save_obj(
                file_path=self.model_trianer_config.trained_model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            
            r2_square = r2_score(y_test,predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e,sys)

