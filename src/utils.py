import os
import sys
import numpy as np
import pandas as pd

import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(filepath,obj):
    """
    Save an object to a file using dill.
    
    Parameters:
    filepath (str): The path where the object will be saved.
    obj: The object to be saved.
    """
    try: 

        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath,'wb') as file_obj:
              dill.dump(obj,file_obj)

    except Exception as e:
            raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
        """
        Evaluate the performance of multiple regression models.
    
        Parameters: 
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable.
        models (dict): Dictionary of models to evaluate.
        Returns:        
        dict: A dictionary containing model names and their corresponding R2 scores.
    
        """
        try:
            from sklearn.metrics import r2_score
            report = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                params = param[list(models.keys())[i]]

                gs = GridSearchCV(model, params, cv=3)
                gs.fit(X_train,y_train)


                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)
    
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
    
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
    
                report[list(models.keys())[i]] = test_model_score
    
            return report
        except Exception as e:
            raise CustomException(e, sys)


def load_object(filepath):
    try:
        with open(filepath, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)


        