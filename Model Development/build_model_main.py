# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:15:56 2019

@author: Kate Hung, Heidi Lin
"""

#%%
import pandas
print (pandas.__version__)

#%%
import numpy as np
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


#%%   
from sources.data_transformation import ManualTransformer

class SegmentATransformer(ManualTransformer): 
    def transform(self, source_dataset): 
        import pandas as pd
        destination_x = source_dataset.x.copy(deep=True)
        destination_y = source_dataset.y.copy(deep=True)
        destination_attribute_list = copy.deepcopy(source_dataset.attribute_list)
        
        destination_x['column_1'] = destination_x["column_1"].apply(lambda x: (1 if x >0.22 else 0))
        destination_x.fillna({'column_2':0}, inplace=True)    
       
        destination_dataset = DataSet()
        destination_dataset.set_all(destination_x, destination_y, destination_attribute_list)

        return destination_dataset

#%%

if __name__ =='__main__': 
    settings = ConsoleSettings()
    settings.set_all()
    __spec__ = None

# =============================================================================
# data import and preprocessing
# =============================================================================

    # data import
    schema_path = 'Data//data_schema.csv'
    modeling_data_path = 'Data//training.csv'
    validation_data_path = 'Data//validation.csv'
    test_data_path = 'Data//testing.csv'

    root_modeling_dataset = DataSet()
    root_modeling_dataset.load_all_from_csv(dataset_filepath=modeling_data_path, schema_filepath=schema_path)
    print(root_modeling_dataset)

    root_validation_dataset = DataSet()
    root_validation_dataset.load_all_from_csv(dataset_filepath=validation_data_path, schema_filepath=schema_path)
    print(root_validation_dataset)    

    root_testing_dataset = DataSet()
    root_testing_dataset.load_all_from_csv(dataset_filepath=test_data_path, schema_filepath=schema_path)
    print(root_testing_dataset)    
        
#%%    
    # data split
    segment_column = 'segment'
    segment_rules = {
            'segment_a': '(assigned_data.segment.str.strip() == "a")',
            'segment_b':'(assigned_data.segment.str.strip() == "b")',
            'segment_c':'(assigned_data.segment.str.strip() == "c")'
            }    
    scorecard_giver = ScorecardGiver(segment_column, segment_rules)   

    # select the segment to process
    seg_name = 'segment_a'
     
    modeling_dataset = scorecard_giver.get_scorecard_datasets(root_modeling_dataset, filepath = test_data_path, card_name = seg_name, dataset_name = 'modeling')   
    validation_dataset = scorecard_giver.get_scorecard_datasets(root_validation_dataset, filepath = validation_data_path, card_name = seg_name, dataset_name = 'validation')       
    testing_dataset = scorecard_giver.get_scorecard_datasets(root_testing_dataset, filepath = test_data_path, card_name = seg_name, dataset_name = 'testing')   

#%%
#    # primary stage variable selection
#    from sources.feature_selection import L1Selector
#    from sklearn.svm import LinearSVC
#    
#    feature_selector = L1Selector()
#    modeling_dataset = feature_selector.fit_transform(source_dataset=modeling_dataset, model=LinearSVC(C=10, penalty='l1', dual=False, class_weight='balanced'), maximum_vote=30)
#    print(feature_selector)
    

    # variable exploration
    x_profiler = XProfiler()
    x_profiler.create_report(modeling_dataset)
    x_profiler.to_html(filepath=r'Models\{0}\reports\XProfileReport.html'.format(seg_name))
    
#%%
    # XY analyzer
    checkpoint, segment_by, max_bins = 'original','auto', 3    #segmenty_by='auto','tree','range'
    
    xy_analyzer = XYAnalyzer(dataset = modeling_dataset, segment_by = segment_by, max_bins = max_bins)
    xy_analyzer.plot()
    xy_analyzer.plot_to_pdf(filepath=r"Models\{0}\reports\XYAnalysisReport_{1}_{2}_M.pdf".format(seg_name,checkpoint,segment_by))  
    
    xy_analyzer = XYAnalyzer(dataset = validation_dataset, segment_by = segment_by, max_bins = max_bins)
    xy_analyzer.plot()
    xy_analyzer.plot_to_pdf(filepath=r"Models\{0}\reports\XYAnalysisReport_{1}_{2}_V.pdf".format(seg_name,checkpoint,segment_by))  

#%%
    # Manual manipulation before binning
    manual_transformer = SegmentATransformer()
    modeling_dataset = manual_transformer.transform(modeling_dataset)        

    
#%%
    # binning
    from sources.data_transformation import Binner        
    segment_by = 'auto'
    max_bin = 10
    columns_to_bin = ['column1','column2']
    
    binner = Binner()    
    binner.fit(modeling_dataset, segment_by, max_bin) 
    binner.set_transform_columns(columns=columns_to_bin) 
    
    file_controller = FileController(filepath="Models\{0}\preprocessors\Binner.pkl".format(seg_name))
    file_controller.save(binner)

    modeling_dataset = binner.transform(source_dataset=modeling_dataset)  

#%%
    # XY analyzer
    checkpoint, segment_by, max_bins = 'binned','range', 10    #segmenty_by='auto','tree','range'
    
    xy_analyzer = XYAnalyzer(dataset = modeling_dataset, segment_by = segment_by, max_bins = max_bins)
    xy_analyzer.plot()
    xy_analyzer.plot_to_pdf(filepath=r"Models\{0}\reports\XYAnalysisReport_{1}_{2}_M.pdf".format(seg_name,checkpoint,segment_by))  
    
    
    validation_dataset_temp = copy.deepcopy(validation_dataset)
    validation_dataset_temp = manual_transformer.transform(validation_dataset_temp)         
    validation_dataset_temp = binner.transform(source_dataset=validation_dataset_temp)         
    
    xy_analyzer = XYAnalyzer(dataset = validation_dataset_temp, segment_by = segment_by, max_bins = max_bins)
    xy_analyzer.plot()
    xy_analyzer.plot_to_pdf(filepath=r"Models\{0}\reports\XYAnalysisReport_{1}_{2}_V.pdf".format(seg_name,checkpoint,segment_by))  


#%%
#    # Manual impute after binning
    manual_transformer = SegmentATransformer()
    modeling_dataset = manual_transformer.transform(modeling_dataset)        

#%%
    # fit: data-preprocessing
    data_preprocessor = Pipeline([  
#                                  ('constant_filterer', ConstantFilterer())
#                                  , ('range_limiter', RangeLimiter())
                                   ('unsupported_datatype_filterer', UnsupportedTypeFilterer())
#                                  , ('missing_value_filterer', MissingValueFilterer())
#                                  , ('nominal_imputer', NominalImputer())
#                                  , ('numeric_imputer', NumericImputer()) 
                                  , ('correlation_filterer', CorrelationFilterer()) 
                                  , ('one_hot_encoder', OneHotEncoder()) 
#                                  , ('vote_feature_selector', VoteFeatureSelector())
                                  ])
    
    param_grid = {    
#                    'range_limiter__method': ['auto']
#                    ,'missing_value_filterer__missing_value_threshold': [0.5]
#                    ,'nominal_imputer__strategy': ["most_frequent"]
#                    ,'numeric_imputer__strategy': ["median"]
                    'correlation_filterer__correlation_threshold': [0.9]
#                    , 'vote_feature_selector__maximum_vote': [12]
                 }
    
    param_combinations = list(ParameterGrid(param_grid))
    parameters = param_combinations[0]
    
    modeling_dataset = data_preprocessor.fit_transform(modeling_dataset, **parameters)
    
    file_controller = FileController(filepath="Models\{}\preprocessors\DataPreprocessor.pkl".format(seg_name))
    file_controller.save(data_preprocessor)
    
    
#    print(data_preprocessor.named_steps["range_limiter"])
#    print(data_preprocessor.named_steps["constant_filterer"])
    print(data_preprocessor.named_steps["unsupported_datatype_filterer"])
#    print(data_preprocessor.named_steps["missing_value_filterer"])
#    print(data_preprocessor.named_steps["nominal_imputer"])
#    print(data_preprocessor.named_steps["numeric_imputer"])
    print(data_preprocessor.named_steps["correlation_filterer"])
    print(data_preprocessor.named_steps["one_hot_encoder"])

#%%
    # fit & transform: standardizing
    standardizer = Standardizer()
    std_modeling_dataset = standardizer.fit_transform(modeling_dataset)
    standardizer.to_pkl(filepath="Models\{}\preprocessors\Standardizer.pkl".format(seg_name))

#%%
    # transform validartion & testing datasets
    validation_dataset = manual_transformer.transform(validation_dataset)       
    validation_dataset = binner.transform(source_dataset=validation_dataset)  
    validation_dataset = data_preprocessor.transform(validation_dataset)    
    std_validation_dataset = standardizer.transform(validation_dataset)
    print('validation_dataset has been transformed')

    testing_dataset = manual_transformer.transform(testing_dataset)   
    testing_dataset = binner.transform(source_dataset=testing_dataset)  
    testing_dataset = data_preprocessor.transform(testing_dataset)    
    std_testing_dataset = standardizer.transform(testing_dataset)
    print('testing_dataset has been transformed')


#%%   
# =============================================================================
# modeling 
# =============================================================================
    
    # create empty collector
    from sources.modeler_collector import ModelerCollector
    modeler_collector = ModelerCollector()   
    print(modeler_collector)
    
    n_bins = 10
    

    # model fitting
    from sources.modeler import Modeler
    from sklearn.linear_model import LogisticRegression
    param_grid = {   'penalty': ['l1','l2'] #done
#                    ,'dual': [True,False]   #"dual" does not change the result, only potentially the speed of the algorithm
                    ,'tol': np.logspace(-5,1,7) #done
                    ,'C': np.logspace(-3,1,5) #done
#                    ,'fit_intercept': [True,False] #done
#                    ,'intercept_scaling': [1]
                    ,'class_weight': ['balanced',{0:0.01,1:0.99},{0:0.005,1:0.995},{0:0.001,1:0.999}]
                    ,'random_state': [None]
                    ,'solver': ['newton-cg','lbfgs','liblinear','sag','saga'] #done
#                    ,'max_iter': [100]
#                    ,'multi_class': ['ovr','multinomial']
#                    ,'verbose': [0]
#                    ,'warm_start': [True,False] #doesn't impact much 
                    ,'n_jobs': [-1]
                 }
    
    param_combinations = list(ParameterGrid(param_grid))

    root_clf = LogisticRegression()
    for parameters in product(param_combinations): 
        try:
            clf = copy.deepcopy(root_clf)
            clf = clf.set_params(**parameters[0])
            modeler = Modeler(clf)
            modeler.fit_transform_evaluate(std_modeling_dataset,std_validation_dataset, n_bins)
            modeler_collector.append_to_model_list(modeler)
        except ValueError: 
            print(error_message, ":", parameters[0])

#%%
    # save all the models in collector (if needed)
    file_controller = FileController(filepath="Models\{}\collector.pkl".format(seg_name))
    file_controller.save(modeler_collector)            

#%%
# =============================================================================
# model selection
# =============================================================================

    # selection criterions and ranking rules
    std_coef = 1
    filter_criterion = [
         ['validation','KS', 'min', 35]
        ,['modeling','KS', 'min', 35]
        ,['validation','PSI', 'max', 0.05]
        ,['validation','ks_gap', 'max', 8]
#        ,['modeling','bin_cnt', 'min', 10]
        ,['modeling','bounce_pct', 'max', 0.1]
        ,['validation','bounce_pct', 'max', 0.1]
#        ,['modeling','LR_Correction','is null','']
                    ]                             
      
    orders = {
     'validation_std_check':False
     ,'modeling_bounce_cnt':True
     ,'validation_bounce_cnt':True
     ,'modeling_KS': False 
     ,'validation_KS': False  
     } # false = descending order, true = ascending order
        
    ranking_table = modeler_collector.select_and_rank_models(filter_criterion, std_coef, orders)            


    
#%%
    # plotting   
    model_list = modeler_collector.rank_n_list(10)

    for m in model_list:
        modeler = modeler_collector.get_modeler(m)
        plotter = LiftPlotter()
        plotter.plot(modeler)

#%%
    # add on additional validation datasets, plot again (if needed)
    for m in model_list:
        modeler = modeler_collector.get_modeler(m)
        modeler.transform_and_evaluate(std_testing_dataset, n_bins, name_of_dataset = 'testing')
        modeler_collector.update_listed_modeler(modeler)        

    for m in model_list:
        modeler = modeler_collector.get_modeler(m)
        plotter = LiftPlotter()
        plotter.plot(modeler)  
        
#%%
    # output selected model
    model_list = [
                  'LogisticRegression_369'
#                  ,'LogisticRegression_168'
#                  ,'LogisticRegression_172'
                  ,'LogisticRegression_82'
#                  ,'LogisticRegression_66'
#                  ,'LogisticRegression_123'
                  ]
     
    for m in model_list:
        modeler = modeler_collector.get_modeler(m)
        plotter = LiftPlotter()
        plotter.plot(modeler)
        plotter.to_pdf(model_id=m)    
        modeler_collector.get_modeler(m).output_to_file(filepath = r'Models\{}'.format(seg_name))
        print(modeler_collector.get_modeler(m).formula_of_linearmodels())
