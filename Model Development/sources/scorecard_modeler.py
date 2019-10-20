# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:44:31 2019

@author: Heidi
"""

import pandas as pd 
import numpy as np
import copy
import datetime
import re
from sources.data_set import DataSet
from sources.modeler import Modeler
from sources.evaluator import Evaluator
from sources.file_controller import FileController, DataframeFileController

class ScorecardGiver():
   
    def __init__(self, segment_column, segment_rules):
        self.segment_column = segment_column
        self.segment_rules = segment_rules
        self.scorecard_list = self.get_scorecard_list(value = None)         # componenet為 str


    def get_scorecard_list (self, value = None):        
        return [k for k, v in self.segment_rules.items() if v != value]    

    
    def assign_scorecards(self, x, dataset_name):       
        '''
        Add <scorecard> column and assign the correspondent scorecard for each record.
        '''
        assigned_data = copy.deepcopy(x)                
        assigned_data["scorecard"] = None           
        for i in self.scorecard_list:      
            if self.segment_rules[i] != 'the rest':
                exec('assigned_data.scorecard.loc[{0}] = i'.format(self.segment_rules[i]))
            else:  # if segment_rules[i] == 'the rest', all the nulls will be assigned 'rest' at the end of assigning process
                assigned_data["scorecard"] = assigned_data["scorecard"].fillna('rest')                 
        print('***** {0} Sample Distribution *****\n{1}\n'.format(dataset_name, assigned_data.scorecard.value_counts()))
        return assigned_data
        

    def get_scorecard_datasets(self, dataset, filepath, card_name, dataset_name, debug_flag = False):      
        x = copy.deepcopy(dataset.x)
        y = copy.deepcopy(dataset.y)
        attribute_list = copy.deepcopy(dataset.attribute_list)
        segment_data = self.get_segment_data_columns(filepath, debug_flag)
        
        assigned_data = self.assign_scorecards(segment_data, dataset_name)     
        assigned_data_sliced = assigned_data[assigned_data.scorecard == card_name]       
        segmented_index = assigned_data_sliced.index.values         
        x_sliced = x.iloc[segmented_index,:]
        y_sliced = y.iloc[segmented_index,]
        
        dataset_sliced = DataSet()
        dataset_sliced.set_all(x_sliced, y_sliced, attribute_list)
        if debug_flag == True:
            print('***** {0} dataset infromation: <{1}> *****\nx: {2}\ny: {3}\n'.format(dataset_name, card_name, dataset_sliced.x.shape, dataset_sliced.y.shape))             
        return dataset_sliced


    def get_multiple_scorecard_datasets(self, dataset, filepath, dataset_name, debug_flag = False):      
        x = copy.deepcopy(dataset.x)
        y = copy.deepcopy(dataset.y)
        attribute_list = copy.deepcopy(dataset.attribute_list)
        segment_data = self.get_segment_data_columns(filepath, debug_flag)
        
        assigned_data = self.assign_scorecards(segment_data, dataset_name)     
        dataset_dict = {}
        
        for card in assigned_data.scorecard.unique():
            assigned_data_sliced = assigned_data[assigned_data.scorecard == card]       
            segmented_index = assigned_data_sliced.index.values         
            x_sliced = x.iloc[segmented_index,:]
            y_sliced = y.iloc[segmented_index,]
            dataset_sliced = DataSet()
            dataset_sliced.set_all(x_sliced, y_sliced, attribute_list)
            if debug_flag == True:
                print('***** {0} dataset infromation: <{1}> *****\nx: {2}\ny: {3}\n'.format(dataset_name, card, dataset_sliced.x.shape, dataset_sliced.y.shape))             
            dataset_dict[card] = dataset_sliced
        return dataset_dict


    def get_multiple_scorecard_dataframes(self, df_filepath, debug_flag = False):      
        
        df = pd.read_csv(df_filepath)
        assigned_data = self.assign_scorecards(df, dataset_name="")     
        df_dict = {}
        
        for card in assigned_data.scorecard.unique():
            assigned_data_sliced = assigned_data[assigned_data.scorecard == card]       
            segmented_index = assigned_data_sliced.index.values         
            df_sliced = df.iloc[segmented_index,:]           
            if debug_flag == True:
                print('***** <{0}> {1} *****'.format(card, df_sliced.shape))             
            df_dict[card] = df_sliced
        return df_dict


    def get_segment_data_columns(self, filepath, debug_flag):                    

        data = DataframeFileController(filepath).load()
        segment_data = data.loc[:, [self.segment_column]]
        if debug_flag == True:
            print('\n***** {0} *****\n{1}\n'.format(filepath, segment_data[self.segment_column].value_counts()))
        return segment_data







class ScorecardModeler(Modeler):
    
    def __init__(self, ScorecardGiver): 
        self.__clf = None
        self.__model_desc = 'LogisticRegression_SC'
        self.__scorecard_container = {}
        self.__scorecard_list = ScorecardGiver.scorecard_list       
        self.__scorecard_rule = ScorecardGiver.scorecard_rule  


    def __str__(self):  
        boundry_t = pd.Series(self.__boundry).round(5)        
        return '\n******************** {0} ********************\n******************** {1} ********************\n{2}\n******************** {3} ********************\n{4}\n******************** {5} ********************\n{6}\n******************** {7} ********************\n{8}'.format(
                self.__model_desc, 'scorecards', self.__scorecard_container.keys() ,'performance',self.__performance,'bin_table', self.__bin_table,'boundry', boundry_t)

    def get_scorecards(self, model_desc):
        return self.__scorecard_container.get(model_desc)
    
    @property
    def scorecard_container(self):
        return self.__scorecard_container

# ==============================same as modeler===============================================
    @property
    def performance(self):
        return self.__performance
    
    @property
    def bin_table(self):
        return self.__bin_table
    
    @property
    def boundry(self):
        return self.__boundry
    
    @property
    def m_KS(self):
        return self.__performance.modeling.KS

    @property
    def v_KS(self):
        return self.__performance.validation.KS

    @property
    def PSI(self):
        return self.__performance.validation.PSI
    
    @property
    def model_desc(self):
        return self.__model_desc
    
    @property
    def model_id(self):
        return self.__model_id
# ==============================same as modeler END===============================================
        
    @property
    def parameters(self):
        return 'Parameters of individual scorecards can be retreive through <scorecardmodeler.cardX.parameter>'

    @property
    def card_bin_table(self):
        return self.__card_bin_table

#    @property
#    def scoring_data(self):
#        data = pd.concat([self.y_agg, self.y_proba], axis = 1)
        



    
# use in main
# =============================================================================

    def collect_into_scorecardmodeler(self, modeler, card_name):
        self.__scorecard_container[card_name] = modeler
        return '*** <{}> has been collected into scorecard_container ***'.format(card_name)


    def set_model_id(self, model_id):
        self.__model_id = model_id 


    def combine_data(self, n_bins):
        self.combine_prediction_data_in_each_card()
        self.__card_bin_table = self.create_card_bin_table()

            
    def combine_data_and_get_performance(self, n_bins):
        self.combine_prediction_data_in_each_card()
        self.__card_bin_table = self.create_card_bin_table()
        self.__boundry, self.__bin_table, self.__performance = Evaluator().evaluate_modeling_validation(self, self.m_y_agg, self.m_y_pred_agg, self.m_y_proba_agg, self.v_y_agg, self.v_y_pred_agg, self.v_y_proba_agg, n_bins)


    def output_to_file(self):
        self.save_pkl_to_file()
        self.save_perf_to_xlsx()



# inner components
# =============================================================================

    def combine_prediction_data_in_each_card(self): #[須改]不可直接等價合併，須經轉換
        '''
        combine_y 
        combine_y_pred
        combine_y_proba
        '''
        m_y_agg = pd.Series()
        m_y_pred_agg = pd.Series()
        m_y_proba_agg = pd.Series()
        v_y_agg = pd.Series()
        v_y_pred_agg = pd.Series()
        v_y_proba_agg = pd.Series()
        
        for model in self.__scorecard_container.keys():   
            modeler = self.__scorecard_container[model]
            m_y_agg = pd.concat([m_y_agg, modeler.m_y], axis=0, ignore_index=False)
            m_y_pred_agg = pd.concat([m_y_pred_agg, modeler.m_y_pred], axis=0, ignore_index=False)
            m_y_proba_agg = pd.concat([m_y_proba_agg, modeler.m_y_proba], axis=0, ignore_index=False)
            v_y_agg = pd.concat([v_y_agg, modeler.v_y], axis=0, ignore_index=False)
            v_y_pred_agg = pd.concat([v_y_pred_agg, modeler.v_y_pred], axis=0, ignore_index=False)
            v_y_proba_agg = pd.concat([v_y_proba_agg, modeler.v_y_proba], axis=0, ignore_index=False)
#            v_y_agg = v_y_agg.append(model.v_y)
#            v_y_pred_agg = v_y_pred_agg.append(model.v_y_pred)
#            v_y_proba_agg = v_y_proba_agg.append(model.v_y_proba)        
        self.m_y_agg = m_y_agg
        self.m_y_pred_agg = m_y_pred_agg
        self.m_y_proba_agg = m_y_proba_agg
        self.v_y_agg = v_y_agg
        self.v_y_pred_agg = v_y_pred_agg
        self.v_y_proba_agg = v_y_proba_agg
        return m_y_agg, m_y_pred_agg, m_y_proba_agg, v_y_agg, v_y_pred_agg, v_y_proba_agg



    def create_card_bin_table(self):  
        '''
        收錄每張評分卡的bin_table資訊
        '''                      
        bin_table_of_all_cards = pd.DataFrame()
        for model in self.__scorecard_container.keys():
            card_name = model
            modeler = self.__scorecard_container[card_name]
            bin_table = modeler.bin_table.validation.loc[:, ['bin','max_scr','Event','total','Event_rate','KS']]
            bin_table.columns = pd.MultiIndex.from_product([[card_name], bin_table.columns])
            bin_table_of_all_cards = pd.concat([bin_table_of_all_cards, bin_table], axis = 1)
            
        return bin_table_of_all_cards    



    ### save to file - PKL ###    
   
    def mkdir(self,path):
        import os
        folder = os.path.exists(path)
        if folder != True:                   #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        else:
            print("---  The folder already exists ---")

        
    def save_pkl_to_file(self):
        '''
        Object: Modeler
        '''
        folder = 'Models\\pkl\\{}'.format(self.__model_desc) 
        self.mkdir(folder)
        card_cnt = 0
        for i in list(self.__scorecard_container.keys()):
            filepath = 'Models\\pkl\\{0}\\{1}.pkl'.format(self.__model_desc, i)
            file_controller = FileController(filepath)
            file_controller.save(self.__scorecard_container[i])
            card_cnt += 1
        print('*** {0} pkls have been saved to folder : {1} ***'.format(card_cnt,folder))



    ### save to file - CSV ###
    
    def save_perf_to_xlsx(self):        
        filepath = 'Models\\{}.xlsx'.format(self.__model_desc)
        with pd.ExcelWriter(filepath) as writer:  
#            performance_df = pd.DataFrame.from_dict(self.__performance, orient='index') 
#            performance_df.to_excel(writer, sheet_name='performance')
#            self.__bin_table.to_excel(writer, sheet_name='bin_table')        
#            boundry = pd.DataFrame(self.__boundry)
#            boundry.to_excel(writer, sheet_name='boundry')
            scorecard_rule = pd.DataFrame.from_dict(self.__scorecard_rule, orient='index') 
            scorecard_rule.to_excel(writer, sheet_name='scorecard_rule')
            self.__card_bin_table.to_excel(writer, sheet_name='card_bin_table')
            
            # scorecard results
            scorecard_performance = pd.DataFrame()
            scorecard_bin_table = pd.DataFrame()
            scorecard_boundry = pd.DataFrame()
            for i in list(self.__scorecard_container.keys()):
                scorecard = self.__scorecard_container[i]
                performance = pd.DataFrame(scorecard.performance)
                performance.columns = pd.MultiIndex.from_product([[i], performance.columns])
                scorecard_performance = pd.concat([scorecard_performance,performance], axis = 1)
                scorecard.bin_table['scorecard'] = i
                scorecard_bin_table = pd.concat([scorecard_bin_table,scorecard.bin_table], axis = 0)
                boundry = pd.DataFrame(scorecard.boundry, columns = [i])
                scorecard_boundry = pd.concat([scorecard_boundry,boundry], axis = 1)
                formula = scorecard.formula_of_linearmodels() 
                formula.to_excel(writer, sheet_name='formula_{}'.format(i))                
            scorecard_performance.to_excel(writer, sheet_name='scorecard_performance')
            scorecard_bin_table.to_excel(writer, sheet_name='scorecard_bin_table')
            scorecard_boundry.to_excel(writer, sheet_name='scorecard_boundry')

        print('*** xlsx has been saved to file: {} ***'.format(filepath))
