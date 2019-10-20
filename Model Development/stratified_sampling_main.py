# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:08:27 2019

@author: Heidi Lin
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sources.scorecard_modeler import ScorecardGiver



if __name__ =='__main__':   

    # data split
    segment_column = 'segment'
    segment_rules = {
            'segment_a': '(assigned_data.segment.str.strip() == "a")',
            'segment_b':'(assigned_data.segment.str.strip() == "b")',
            'segment_c':'(assigned_data.segment.str.strip() == "c")'
            }    
    scorecard_giver = ScorecardGiver(segment_column, segment_rules)    

    df_dict = scorecard_giver.get_multiple_scorecard_dataframes('df_filepath=Data\\data.csv', debug_flag=False)
    
    
    
    # stratified sampling
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=1)
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    # sampling in each segments
    for card in df_dict.keys():
        print('***** {} *****'.format(card))    
        y_column = ['event']
        x_columns = list(set(df_dict[card].columns)-set(y_column))
      
        X = df_dict[card][x_columns]
        y = df_dict[card][y_column]
        
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]        
            train = pd.concat([X_train, y_train], axis=1, sort= False)
            test = pd.concat([X_test, y_test], axis=1, sort= False)        
            df_train = pd.concat([df_train, train], axis=0, sort=False)
            df_test = pd.concat([df_test, test], axis=0, sort=False)
            print("TRAIN:", len(train_index), ",Y% :", round(df_train['event'].sum()/df_train['event'].count(),5), '\n',
                  "TEST:", len(test_index), ",Y% :", round(df_test['event'].sum()/df_test['event'].count(),5))
    
    print('*************** Overall ***************', '\n', 
          "TRAIN:", len(df_train.index), ",Y% :", round(df_train['event'].sum()/df_train['event'].count(),5), '\n',
          "TEST:", len(df_test.index), ",Y% :", round(df_test['event'].sum()/df_test['event'].count(),5) )
    
    
    
    # data output
    df_train.to_csv(r'Data\modeling_data.csv')
    df_test.to_csv(r'Data\validation_data.csv')




