# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:19:32 2019

@author: Brady Huang, Heidi Lin
"""

#%%
import numpy as np
import pandas as pd
import copy

from models.segment_a.scoring_function import SegmentAScorer
from models.segment_b.scoring_function import SegmentBScorer
from models.segment_c.scoring_function import SegmentCScorer


#%%
if __name__ == '__main__':
    
    datalist = ['scoring_data_1', 'scoring_data_2']
    
    for i in datalist:
        # get the dataset to be scored
        root_scoring_dataset_path = 'input/'+ i +'.csv'
        
        segmentascorer = SegmentAScorer()  
        df_segment_a = segmentascorer.score(root_scoring_dataset_path, 'scoring') 
        
        segmentbscorer = SegmentBScorer()  
        df_segment_b = segmentbscorer.score(root_scoring_dataset_path, 'scoring') 

        segmentcscorer = SegmentCScorer()  
        df_segment_c = segmentcscorer.score(root_scoring_dataset_path, 'scoring') 

    
        # concatenate predicted value from each scorecard
        score = pd.concat([df_segment_a, df_segment_b, df_segment_c], ignore_index=True)
        
        # get rnum
        score = score.sort_values(['old_index'], ascending = True)
        root_data = pd.read_csv(root_scoring_dataset_path, encoding = 'big5')[['RNUM', 'segment']].reset_index()
        score = score.merge(root_data, left_on  = 'old_index', right_on = 'index', how = 'inner')
        
        score = score[['RNUM', 'card', 'bin', 'y_proba']]
        score['product'] = i.split('_')[-1]
        
        # get rank
        bin_dict = {                
                'segment_c':{
                        0:['segment_c_0', 'R01'],
                        1:['segment_c_1', 'R02'],
                        2:['segment_c_2', 'R03'],
                        3:['segment_c_3', 'R03'],
                        4:['segment_c_4', 'R04'],
                        5:['segment_c_5', 'R04'],
                        6:['segment_c_6', 'R04'],
                        7:['segment_c_7', 'R05'],
                        8:['segment_c_8', 'R06'],
                        9:['segment_c_9', 'R07']},
                
                'segment_b':{
                        0:['segment_b_0', 'R02'],
                        1:['segment_b_1', 'R03'],
                        2:['segment_b_2', 'R03'],
                        3:['segment_b_3', 'R05'],
                        4:['segment_b_4', 'R05'],
                        5:['segment_b_5', 'R06'],
                        6:['segment_b_6', 'R06'],
                        7:['segment_b_7', 'R06'],
                        8:['segment_b_8', 'R07'],
                        9:['segment_b_9', 'R07']},

                
                'segment_a':{
                         0:['segment_a_0', 'R01'],
                         1:['segment_a_1', 'R04'],
                         2:['segment_a_2', 'R04']}
                }
              
        def get_rank(dataframe):
            rank = bin_dict[dataframe.card][dataframe.bin][1]
            return rank
        
    #    def get_binname(dataframe):
    #        binname = bin_dict[dataframe.card][dataframe.bin][0]
    #        return binname
        
    
        # assign final rank 
        score['rank'] = score.apply(get_rank, axis = 1)
    #    score['binname'] = score.apply(get_binname, axis = 1)
    
        
        # output to csv
        score.to_csv('output\\scoring_results_' + i + '.csv', index = False)
        '''
            type: dataframe
            ============ ===============
            column name  content
            ============ ===============
            rnum         key值
            rank
                  
        '''