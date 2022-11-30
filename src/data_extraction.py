import pandas as pd 
import numpy as np 


def data_extraction(data_path): 
    data = pd.read_csv(data_path)
    return data