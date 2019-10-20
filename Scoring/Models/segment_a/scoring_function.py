# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:43:10 2019

@author: Heidi Lin, Brady Huang
"""


# 套件安裝
import numpy as np
import pandas as pd
import copy
from itertools import product
from sklearn.model_selection._search import ParameterGrid
from sklearn.pipeline import Pipeline

from sources.data_set import DataSet 
from sources.data_transformation import RangeLimiter, NominalImputer, NumericImputer
from sources.data_transformation import OneHotEncoder, Standardizer
from sources.feature_selection import ConstantFilterer, UnsupportedTypeFilterer, MissingValueFilterer, CorrelationFilterer
from sources.feature_selection import VoteFeatureSelector
from sources.feature_profiling import XProfiler, XYAnalyzer
from sources.modeler_collector import ModelerCollector
from sources.modeler import Modeler
from sources.scorecard_modeler import ScorecardGiver    
from sources.evaluator import Evaluator

from sources.file_controller import FileController, DataframeFileController
from sources.console_settings import ConsoleSettings
from sources.feature_profiling import LiftPlotter   
from sources.data_transformation import ManualTransformer
from sources.generalscorer import GeneralScorer

#%%
class SegmentATransformer(ManualTransformer): 
    def transform(self, source_dataset): 
        import pandas as pd
        destination_x = source_dataset.x.copy(deep=True)
        destination_y = source_dataset.y.copy(deep=True)
        destination_attribute_list = copy.deepcopy(source_dataset.attribute_list)
        
        destination_x['USE_P12_0063'] = destination_x["USE_P12_0063"].apply(lambda x: (1 if x >0.22 else 0))
        destination_x.fillna({'HOL_P12_0009':0}, inplace=True)
        destination_x.fillna({'RT_P12_14':0}, inplace=True)        
        destination_x.fillna({'RT_P12_18':1}, inplace=True)        
       
        destination_dataset = DataSet()
        destination_dataset.set_all(destination_x, destination_y, destination_attribute_list)

        return destination_dataset


#%%
        
class SegmentAScorer(GeneralScorer): 
    def __init__(self, seg_name = 'segment_a', model_name = 'LogisticRegression_369', n_bin = 10, modeling_dataset_path = 'models\\modeling_data.csv'):
        super(SegmentAScorer, self).__init__(seg_name, model_name, n_bin, modeling_dataset_path)
         
    def transform(self, dataset):
        preprocessor_path = 'models\\' + self.seg_name + '\preprocessors'
        
        file_controller = FileController(filepath='{}\Binner.pkl'.format(preprocessor_path))
        binner = file_controller.load()
    
        file_controller = FileController(filepath='{}\DataPreprocessor.pkl'.format(preprocessor_path))
        data_preprocessor = file_controller.load()
    
        file_controller = FileController(filepath='{}\Standardizer.pkl'.format(preprocessor_path))
        standardizer = file_controller.load()
        
        try:
            manualtransformer = SegmentATransformer()
            dataset = manualtransformer.transform(dataset)
            dataset = binner.transform(dataset)  
            dataset = data_preprocessor.transform(dataset)    
            std_dataset = standardizer.transform(dataset)
            print('Dataset has been transformed')
            
        except:
            print('Error: dataset transformation')
            
        return std_dataset
