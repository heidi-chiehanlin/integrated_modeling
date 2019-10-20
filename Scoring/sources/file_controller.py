# -*- coding: utf-8 -*-
"""
Module holds classes related to file controlling (save/load from file)

@author: chialing
"""

from sklearn.externals import joblib
import pandas as pd
import sys 

class FileController(): 
    """
    | Class holds basic file control functions to save / load python objects through joblib.
    | File path is set at at init
    """
    def __init__(self, filepath): 
        self.__path = filepath
    
    def get_path(self): 
        """
        Data Type: ``Str``
        """
        return self.__path
    
    def pickle_get_current_class(self, obj):
        name = obj.__class__.__name__
        module_name = getattr(obj, '__module__', None)
        obj2 = sys.modules[module_name]
        for subpath in name.split('.'):
            obj2 = getattr(obj2, subpath)
        return obj2
        
    def save(self, object_): 
        """
        Function saves object to file through joblib
        """
        object_.__class__ = self.pickle_get_current_class(object_)
        joblib.dump(object_, self.get_path())
    
    def load(self): 
        """
        Function loads object from file through joblib
        """
        return joblib.load(self.get_path())
    
#%%
class DataframeFileController(FileController): 
    """
    Class holds file control functions for ``pandas.DataFrame``
    """
    def __init__(self, filepath, encoding="big5", index=0): 
        super().__init__(filepath)
        self.__encoding = encoding
        self.__index = index
    
    def save(self, dataframe): 
        """
        Function saves ``pandas.DataFrame`` object to file through ``pandas`` functions
        """
        dataframe.to_csv(self.get_path(), encoding = self.__encoding, index=self.__index)
    
    def load(self): 
        """
        Function loads ``pandas.DataFrame`` object to file through ``pandas`` functions
        """
        return pd.read_csv(self.get_path(), encoding = self.__encoding)
