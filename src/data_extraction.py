import pandas as pd 
import numpy as np 


def data_extraction(data_path): 
    """
    data_extraction : 
        function to extract data from the end location/path required. 

    Parameters: 
    ------------
        data_path : string 
            path required to get the csv files 
        
    Returns: 
    -------
        data: pandas dataframe
    
    """
    data = pd.read_csv(data_path)
    return data