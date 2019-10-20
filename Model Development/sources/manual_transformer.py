# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:07:12 2019

@author: chialing
"""

import copy
import pandas as pd
from sources.data_set import DataSet

class ManualTransformer(): 
    def __init__(self): 
        pass
    def fit(self): 
        pass
    
    def transform(self, source_dataset): 
   
        destination_x = source_dataset.x.copy(deep=True)
        destination_y = source_dataset.y.copy(deep=True)
        destination_attribute_list = copy.deepcopy(source_dataset.attribute_list)
        
#        x_df = destination_x["INQ_P03_2"].apply(lambda x: (1 if x<=1 else x))
#        destination_x["INQ_P03_2_BIN"] = x_df
#        destination_x = destination_x.drop(['INQ_P03_2'], axis=1)
#        
#        add_list = pd.DataFrame(data=[['X', 'INQ_P03_2_BIN', 'numeric']], columns=['Category', 'Field', 'Type'])
#        drop_list = ['INQ_P03_2']
#        destination_attribute_list.adjust_x_list(add_list = None, drop_list=drop_list)
#        destination_attribute_list.adjust_x_list(add_list = add_list, drop_list=None)
        
        destination_dataset = DataSet()
        destination_dataset.set_all(destination_x, destination_y, destination_attribute_list)

        return destination_dataset


#class CcTxnTransformer(ManualTransformer): 
#    def transform(self, source_dataset): 
#   
#        destination_x = source_dataset.x.copy(deep=True)
#        destination_y = source_dataset.y.copy(deep=True)
#        destination_attribute_list = copy.deepcopy(source_dataset.attribute_list)
#        
##        x_df = destination_x["INQ_P03_2"].apply(lambda x: (1 if x<=1 else x))
##        destination_x["INQ_P03_2_BIN"] = x_df
##        destination_x = destination_x.drop(['INQ_P03_2'], axis=1)
##        
##        add_list = pd.DataFrame(data=[['X', 'INQ_P03_2_BIN', 'numeric']], columns=['Category', 'Field', 'Type'])
##        drop_list = ['INQ_P03_2']
##        destination_attribute_list.adjust_x_list(add_list = None, drop_list=drop_list)
##        destination_attribute_list.adjust_x_list(add_list = add_list, drop_list=None)
#        
#        destination_dataset = DataSet()
#        destination_dataset.set_all(destination_x, destination_y, destination_attribute_list)
#
#        return destination_dataset
