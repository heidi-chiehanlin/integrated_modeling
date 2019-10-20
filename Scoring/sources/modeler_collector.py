# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:23:41 2019

@author: Heidi
"""

import pandas as pd
import numpy as np
from sources.file_controller import FileController
import copy
import os
import sys
#os.environ['PATH'] = os.environ['PATH'] + (r';D:\Program Files\Graphviz2.38') # 將Graphiz路徑加入環境變量 # 目前已不需在此處手動添加
#os.environ['PATH'] = os.environ['PATH'] + (r';D:\Program Files\Graphviz2.38\bin')



class ModelerCollector():

    def __init__(self):
        self.models = {}
        self.model_no = 0
        
    def __str__(self):
       return '*************** {} models collected in this collector: ***************\n{}'.format(len(list(self.models.keys())),list(self.models.keys()))

    @property
    def ranking_table(self):
        return self.__ranking_table


    @property
    def rankables(self):
        return self.__rankables
    

    @property
    def base_table(self):
        return self.__base_table


    def get_modeler(self, model_id):
        return copy.deepcopy(self.models.get(model_id))
    
    
    def rank_n_list(self, rank_n):
        '''
        Get the list of model_id that fulfills the criterion: (1) rank <= rank_n (2) belongs to the assigned kind of model.
        '''
        rank_n_list = self.__ranking_table[self.__ranking_table['rank'] <= rank_n].index.tolist()             
        return rank_n_list
        

    def append_to_model_list(self, modeler):    
        '''
        Append class Modeler into class Modeler_Collector.
        '''
        model = copy.deepcopy(modeler)
        self.model_no += 1
        model_id = model.model_desc + '_' + str(self.model_no)
        model.set_model_id(model_id)
        self.models[model_id] = copy.deepcopy(model)   


    def update_listed_modeler(self, modeler):    
        '''
        Update Modeler after addtional validation dataset evaluation was done. 
        '''
        model = copy.deepcopy(modeler)
        model_id = model.model_id
        self.models[model_id] = copy.deepcopy(model)   


    def select_and_rank_models(self, filter_criterion, std_coef, orders):
        '''
        Gather all the model performance and parameters in ranking table. 
        Give rank to models that fulfill the criterion: (1) meet the standard (2) has all the information asked in "rank_by"
        '''
        self.add_on_evaluations(std_coef)
        self.__rankables = self.rankables_filter(filter_criterion)
        self.__base_table = self.create_base_table()
        self.rank_models(orders)
        return self.__ranking_table


    def insert_into_new_collector(self, new_collector, model_list):
        for m in model_list:    
            new_collector.models[m] = copy.deepcopy(self.models.get(m))


    def save_pkl_to_file(self, collector_name):
        """
        Save ``ModelerCollector`` to file as an pickle object.
        """
        filepath = 'Models\\{}.pkl'.format(collector_name)
        file_controller = FileController(filepath)
        file_controller.save(self)
        print('*** Collector has been saved to file: {} ***'.format(filepath))


# inner components
# =============================================================================


    def add_on_evaluations(self, std_coef):
        
        for m in list(self.models.keys()):
            self.std_interval_check(self.models[m], std_coef)
            self.ks_gap_check(self.models[m])



    def rankables_filter(self, filter_criterion):          # LogisticRegression_Scorecard should pass the same quality test too.
        '''
        Find rankable models that fulfill the criterion.   
        '''
        models = self.models
        model_list = list(self.models.keys())        
        rankables = []
        
        for m in model_list:
            md = models[m]    
            md_result = True

            for cond in range(len(filter_criterion)):
                dataset = filter_criterion[cond][0]
                indicator = filter_criterion[cond][1]
                threshold = filter_criterion[cond][2]
                value = filter_criterion[cond][3]    
                            
                # cond_result: 該條件是否通過
                try: # 有該dataset的情況
                    if threshold == 'max':
                        cond_result = md.performance.loc[indicator, dataset] <= value
                    elif threshold == 'min':
                        cond_result = md.performance.loc[indicator, dataset] >= value
                    elif threshold == 'equal':
                        cond_result = md.performance.loc[indicator, dataset] == value  
                    elif threshold == 'is not null':
                        cond_result = md.performance.loc[indicator, dataset] == ''
                except KeyError: # 無該dataset的情況
                    cond_result = False
                    
                # md_result: 該模型否所有條件都通過
                md_result = (md_result & cond_result) 
                
            if md_result == True:
                rankables.append(m)      

        return rankables


    def create_base_table(self):
        '''
        Gather model performance and parameters in ranking table. 
        '''      
        models = self.models
        model_list = list(self.models.keys())
        bt = pd.Series(list(models.keys()), name='model_id')
        bt = pd.DataFrame(data=bt).set_index('model_id')    
        bt = bt.astype('object')

        for i in model_list:                    
            # information from modeling dataset: parameters, correction, mincoef, zerofi, minfi
            bt.loc[i, 'parameters'] = models[i].parameters
            
            l1 = models[i].performance.loc['LR_Correction','modeling']
            str1 = ', '.join(l1)
            bt.loc[i, 'LR_Correction'] = str1                
            bt.loc[i, 'LR_Mincoef'] = models[i].performance.loc['LR_Mincoef','modeling']

            l2 = models[i].performance.loc['DT_Zerofi','modeling']
            str2 = ', '.join(l2)
            bt.loc[i, 'DT_Zerofi'] = str2
            bt.loc[i, 'DT_Minfi'] = models[i].performance.loc['DT_Minfi','modeling']

            # information from all the datasets:
            tmp_dataset_list = models[i].performance.columns.tolist()
            for col in tmp_dataset_list:  
                col_name_ks = '{}_KS'.format(col)        
                col_name_psi = '{}_PSI'.format(col)
                col_name_bcnt = '{}_bounce_cnt'.format(col)
                col_name_bpct = '{}_bounce_pct'.format(col)
                col_name_lift = '{}_Lift'.format(col)
                col_name_std = '{}_std_check'.format(col)
            
                bt.loc[i, col_name_ks] = models[i].performance.loc['KS',col]
                bt.loc[i, col_name_psi] = models[i].performance.loc['PSI',col]
                bt.loc[i, col_name_bcnt] = models[i].performance.loc['bounce_cnt',col]    
                bt.loc[i, col_name_bpct] = models[i].performance.loc['bounce_pct',col]  
                bt.loc[i, col_name_lift] = models[i].performance.loc['Lift',col]
                bt.loc[i, col_name_std] = models[i].performance.loc['std_check',col]  
        
        # remove unused columns                
        bt = bt.drop(['modeling_PSI'], axis=1)
        bt = bt.drop(['modeling_std_check'], axis=1)
      
        base_table = bt        
        return base_table  


    def rank_models(self, orders):     
        '''
        Rank models by the assigned criteria (can be multiple). Models that failed to meet the criterion will not be ranked but remained in the table.
        '''
        rank_by = []
        asc_by = []       
        for ind in orders.keys():
            rank_by.append(ind)
            asc_by.append(orders[ind])
       
        # pick out models that available for ranking
        df_for_rank = self.__base_table.loc[self.__rankables,:]     
        df_for_rank = df_for_rank.loc[:,rank_by]                 
        df_for_rank = df_for_rank.dropna(axis=0,how='any')   
        # rank models
        tups = df_for_rank[rank_by].sort_values(rank_by, ascending=asc_by).apply(tuple, 1)
        f, i = pd.factorize(tups)
        factorized = pd.Series(f + 1, tups.index)
        # organize the rank order to be displayed in final ranking_table
        by_rank = ['rank']
        for i in rank_by:    
            by_rank.append(i)
        by_asc = [True]
        for i in asc_by:    
            by_asc.append(i)
        # assign ranking to basic_table
        ranked_table = self.__base_table.assign(rank=factorized)
        self.__ranking_table = ranked_table.sort_values(by=by_rank, ascending=by_asc)         
        return self.__ranking_table
    
 

    def std_interval_check(self, modeler, std_coef):
        '''
        Evaluate the difference level between validation and modeling dataset. Information will be updated in bin_table.
        '''
        bin_table = modeler.bin_table
        performance = modeler.performance
        # define interval   
        bin_table['modeling','std_upper'] = bin_table.modeling['event_rate'] + bin_table.modeling['std']*std_coef
        bin_table['modeling','std_lower'] = bin_table.modeling['event_rate'] - bin_table.modeling['std']*std_coef        
        # interval check
        temp_dataset_list = bin_table.columns.levels[0].tolist()
        temp_dataset_list.remove('modeling')
        # count 
        for d in temp_dataset_list:
            bin_table[d,'interval_check'] = (bin_table['modeling','std_upper']  > bin_table[d,'event_rate']) & (bin_table['modeling','std_lower']  < bin_table[d,'event_rate'])
            good_cnt = 0
            for i in bin_table[d,'interval_check']:
                if i == True:
                    good_cnt += 1 
                else:
                    good_cnt += 0    
            performance.loc['std_check', d] = good_cnt
            


    def ks_gap_check(self, modeler):
        '''
        Evaluate the difference level between validation and modeling dataset. Information will be updated in performance table.
        '''
        bin_table = modeler.bin_table
        performance = modeler.performance
              
        temp_dataset_list = performance.columns.to_list()
        temp_dataset_list.remove('modeling')

        for d in temp_dataset_list:
            performance.loc['ks_gap',d] = abs(performance.loc['KS', d] - performance.loc['KS', 'modeling'])
            
