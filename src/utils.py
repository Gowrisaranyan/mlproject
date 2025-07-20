import os
import sys
import numpy as np
import pandas as pd

import dill

from src.exception import CustomException

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