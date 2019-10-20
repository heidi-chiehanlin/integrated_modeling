# -*- coding: utf-8 -*-
"""
``Evaluator`` is an object that implements the process of model performance evaluation. \n
Basically, the evaluator generates 3 objects:
    - bin_table: Provides detail infromation of model performance in each dataset and bins. This includes KS, PSI, IV,
    - performance: Provides summary information of model performance in each dataset. This includes KS, PSI, IV, br_bounce, accuracy, recall, f1_score, auc etc.
    - boundry: The lower edge of each bins in Modeling dataset. Used in binning of all the datasets.
    
    
@author: Heidi 

"""

import pandas as pd 
import numpy as np
import datetime


class Evaluator():
    '''
    When dataset contains real Y, all the performance indicators are available(i.e. Modeling and Validation dataset).
    While simply scoring, only PSI is available, while most performance indicators are not.
    '''
    
    def __init__(self):
        pass
    
    def evaluate_modeling_validation(self, modeler, m_y, m_y_pred, m_y_proba, v_y, v_y_pred, v_y_proba, n_bins):
        '''
        Evaluate *two datasets*: Modeling and Validation at the same time. Two groups of predicted value is required.
        '''               
        boundry = self.define_boundry(m_y_proba, n_bins)
        m_binned_data = self.assign_bin(m_y_proba, boundry)
        m_bin_table = self.create_bin_table(m_binned_data, m_y, m_y_proba, dataset_name = 'modeling')   
#        print('boundry:{}'.format(boundry))
#        print('m_binned_data:{}'.format(m_binned_data.head()))
#        print('m_bin_table:{}'.format(m_bin_table))
        
        v_binned_data = self.assign_bin(v_y_proba, boundry)       
        v_bin_table = self.create_bin_table(v_binned_data, v_y, v_y_proba, dataset_name = 'validation')            
        v_bin_table = self.calculate_psi(m_binned_data, v_bin_table, dataset_name = 'validation') 
        bin_table = self.merge_bin_tables(m_bin_table, v_bin_table, v_dataset_name = 'validation')
        
        
        s_modeling = self.organize_sample_performance(modeler, m_y, m_y_pred, m_bin_table, dataset_name = 'modeling')
        s_validation = self.organize_sample_performance(modeler, v_y, v_y_pred, v_bin_table, dataset_name = 'validation')
        performance = pd.concat([s_modeling, s_validation], axis=1)
        
        return bin_table, performance

    
    


    def evaluate_extra_validation(self, y, y_pred, y_proba, n_bins, m_y_proba, modeler_bin_table, modeler_performance, dataset_name):
        '''
        Evaluate *one validation dataset* after model was built. Information of this dataset will be append to existing performance/bin_table.
        '''                               
        modeler = None
        boundry = self.define_boundry(m_y_proba, n_bins)
        m_binned_data = self.assign_bin(m_y_proba, boundry)
        binned_data = self.assign_bin(y_proba, boundry)
        
        extra_v_bin_table = self.create_bin_table(binned_data, y, y_proba, dataset_name)  
        extra_v_bin_table = self.calculate_psi(m_binned_data, extra_v_bin_table, dataset_name)        
        bin_table = self.merge_bin_tables(modeler_bin_table, extra_v_bin_table, dataset_name)
        
        s_extra_validation = self.organize_sample_performance(modeler, y, y_pred, extra_v_bin_table, dataset_name)
        performance = pd.concat([modeler_performance, s_extra_validation], axis=1, sort=True)
 
        return bin_table, performance    






           
    def evaluate_scoring(self, y_proba, n_bins, m_y_proba, modeler_bin_table, modeler_performance, dataset_name):
        '''
        For scoring dataset (without real Y)
        '''     
        boundry = self.define_boundry(m_y_proba, n_bins)
        m_binned_data = self.assign_bin(m_y_proba, boundry)
        binned_data = self.assign_bin(y_proba, boundry)
        
        s_bin_table = self.create_scoring_bin_table(binned_data, y_proba, dataset_name)   
        s_bin_table = self.calculate_psi(m_binned_data, s_bin_table, dataset_name)    
        bin_table = self.merge_bin_tables(modeler_bin_table, s_bin_table, dataset_name)
        
        s_scoring = self.organize_scoring_performance(s_bin_table, dataset_name)
        performance = pd.concat([modeler_performance, s_scoring], axis=1)
        return bin_table, performance


    
    
    def bivariate_analysis_binning(self, binned_data, y_data):
        '''
        Calculate KS, IV.
        
        **Input** 
            ============ ============================= ====================================================
            Parameter    Data Type                     Description
            ============ ============================= ====================================================
            binned_data  ``pd.Series`` (n_records,)    Contains index and their correspondent bins.
            y_data       ``pd.Series`` (n_records,)    Contains index and their correspondent y.
            ============ ============================= ====================================================
            
        **Output** 
            ============ ============================= ===========================================================
            Parameter    Data Type                     Description
            ============ ============================= =========================================================== 
            bin_table    ``pd.DataFrame``              Return performance calculation in each bins.
            IV           ``float``                     IV value of variable X. 
            KS           ``float``                     KS value of variable X.
            br_bounce    ``int``                       The number of times when bad rate doesn't follow the trend.
            ============ ============================= ===========================================================
        
        '''        
        # data prep
        df = pd.DataFrame()
        df['bin'] = binned_data
        df['event'] = y_data
        df['non_event'] = 1-y_data
        df['NonEvent'] = 1-y_data
        
        # create bin table
        grouped = df.groupby('bin', as_index = False)
        bin_table = pd.DataFrame(grouped.min().bin, columns = ['bin'])  
        bin_table['event'] = grouped.sum().event
        bin_table['non_event'] = grouped.sum().non_event
        bin_table['total'] = bin_table.event + bin_table.non_event
        bin_table['p_total'] = (bin_table.total / bin_table.total.sum())
        bin_table['event_rate'] = (bin_table.event / bin_table.total)
        bin_table['p_event'] = (bin_table.event / df.event.sum())
        bin_table['p_non_event'] = (bin_table.non_event / df.non_event.sum())
        with np.errstate(divide='ignore'):      # silence "RuntimeWarning: divide by zero encountered in log"
            bin_table['IV'] = (bin_table.p_event - bin_table.p_non_event)*np.log(bin_table.p_event/bin_table.p_non_event)
            bin_table['KS'] = abs(np.round(((bin_table.event / df.event.sum()).cumsum() - (bin_table.non_event / df.non_event.sum()).cumsum()), 4) * 100)
        bin_table.IV.loc[(bin_table.p_event == 0) | (bin_table.p_non_event == 0)] = 0   # 防爆       
        IV = bin_table['IV'].sum()
        KS = bin_table['KS'].max()

        # counting bounce of bad rate
        br = bin_table.loc[:,['bin','event_rate']][bin_table.bin != 'NaN' ]  # exclude bin = 'NaN' when counting bounce
        br = br.reset_index(drop=True)
                
        br_order = ''   # testify the trend of bad rate
        min_br_bin = br.bin[br.event_rate == br.event_rate.min()].min() # the bin's name with lowest bad rate
        max_br_bin = br.bin[br.event_rate == br.event_rate.max()].min() # the bin's name with highest bad rate
        if min_br_bin > max_br_bin: 
            br_order = 'descending'
        elif min_br_bin < max_br_bin: 
            br_order = 'ascending'
        else:  # 2 possible situations: (1)all the bins have same bad rate (2)there's only one bin (3)there's only one bin and one 'NaN' bin
            br_order = 'no order'    
                   
        bad_rate_bounce = 0     # counting bounce when bins don't follow the trend
        if br_order == 'descending':
            for i in range(0,len(br.index)-1):
                if br.event_rate[i] < br.event_rate[i+1]: 
                    bad_rate_bounce += 1  
        elif br_order == 'ascending':
            for i in range(0,len(br.index)-1):
                if br.event_rate[i] > br.event_rate[i+1]: 
                    bad_rate_bounce += 1
        elif br_order == 'no order':
            bad_rate_bounce= 0
            
        return bin_table, IV, KS, bad_rate_bounce



# inner component for modeling & validation
# =============================================================================

    def define_boundry(self, m_y_proba, n_bins):
        '''
        Follow predicted probability of Modeling dataset to cut records into n bins, with each bin contains equal number of records. 
        
        **Input** 
            ============ ============================= =============================================================
            Parameter    Data Type                     Description
            ============ ============================= =============================================================
            m_y_proba    ``pd.Series`` (n_records,)    The predicted probability of each record in modeling dataset.
            n_bins       ``int``                       The number of bins to cut.
            ============ ============================= =============================================================

        **Ouput** : The lower edge (minimum value) for each thresholds. Data type: ``pd.DataFrame``.
        
        '''        
        # binned_data
        data = pd.DataFrame(m_y_proba)
        data['rank'] = pd.DataFrame.rank(data,method='first',ascending=False)   
        data['bin'] = pd.qcut(data['rank'], n_bins, labels = list(range(0,n_bins)))           # divide evenly into bins by rank
        
        # boundry
        boundry = []
        for i in list(range(0,n_bins)):
            min_scr = data[data.bin == i].iloc[:,0].min()
            boundry.append(min_scr)
        boundry = pd.Series(boundry)
        return boundry



    def assign_bin(self, y_proba, boundry):  
        '''
        Assign the bins for each record in the dataset according to boundry (Modeling dataset).
        '''                         
        df = pd.DataFrame(y_proba, columns = ['y_proba'])
        df['bin'] = None
        
        for i in range(len(boundry.index)):
            if i == 0 :
                CMD = 'df["bin"].loc[ df["y_proba"] > boundry[0] ] = 0'
                exec(CMD)
            elif i in range(1, len(boundry.index)-1):
                CMD = 'df["bin"].loc[ (df["y_proba"] > boundry[{0}]) & (df["y_proba"] <= boundry[{1}])] = {0}'.format(i,i-1)
                exec(CMD)
            else : 
                CMD = 'df["bin"].loc[ df["y_proba"]  <= boundry[{1}] ] = {0}'.format(i,i-1)
                exec(CMD)               
        binned_data = df.bin
        return binned_data
 
    
        
    def create_bin_table(self, binned_data, y, y_proba, dataset_name):  
        '''
        Divide dataset into bins according to boundry (Modeling dataset). Calculate KS, IV, event rate...etc. of each bins.
        '''              
        # data prep
        y = pd.Series(y, name = 'event')
        y_proba = pd.Series(y_proba, name = 'y_proba')
        df = pd.concat([binned_data, y, y_proba], axis = 1)
        df['non_event'] = 1-df.event                 

        # create bin table 
        grouped = df.groupby('bin', as_index = False)
        bin_table = pd.DataFrame(grouped.min().bin, columns = ['bin'])  
        bin_table['min_proba'] = grouped.min().y_proba
        bin_table['max_proba'] = grouped.max().y_proba
        bin_table['event'] = grouped.sum().event
        bin_table['non_event'] = grouped.sum().non_event
        bin_table['total'] = bin_table.event + bin_table.non_event
        bin_table['p_total'] = (bin_table.total / bin_table.total.sum())
#        bin_table['event_rate'] = (bin_table.event / bin_table.total).apply('{0:.2%}'.format)
        bin_table['event_rate'] = (bin_table.event / bin_table.total)
        if dataset_name == 'modeling':
            bin_table['std'] = (bin_table.event_rate*(1-bin_table.event_rate)/(bin_table.total-1))**0.5
        bin_table['p_event'] = (bin_table.event / df.event.sum())
        bin_table['p_non_event'] = (bin_table.non_event / df.non_event.sum())
        with np.errstate(divide='ignore'):      # silence "RuntimeWarning: divide by zero encountered in log"
            bin_table['IV'] = (bin_table.p_event - bin_table.p_non_event)*np.log(bin_table.p_event/bin_table.p_non_event)
            bin_table['KS'] = np.round(((bin_table.event / df.event.sum()).cumsum() - (bin_table.non_event / df.non_event.sum()).cumsum()), 4) * 100
        bin_table.IV.loc[(bin_table.p_event == 0) | (bin_table.p_non_event == 0)] = 0   # 防爆               

        br = bin_table
        br_range = br.event_rate.max() - br.event_rate.min()
                
        # testify the trend of bad rate
        min_br_bin = br.bin[br.event_rate == br.event_rate.min()].min() # the bin's name with lowest bad rate
        max_br_bin = br.bin[br.event_rate == br.event_rate.max()].min() # the bin's name with highest bad rate
        if min_br_bin > max_br_bin: 
            br_order = 'descending'
        elif min_br_bin < max_br_bin: 
            br_order = 'ascending'
        else:  # 2 possible situations: (1)all the bins have same bad rate (2)there's only one bin (3)there's only one bin and one 'NaN' bin
            br_order = 'no order'    
                   
        # measure bounce level
        bin_table['bounce_pct'] = None       
        if br_order == 'descending':
            for i in range(0,len(bin_table.index)-1):
                if bin_table.event_rate[i] < bin_table.event_rate[i+1]: 
                    bin_table.loc[i,'bounce_pct']  = (bin_table.event_rate[i+1]-bin_table.event_rate[i])/br_range
        elif br_order == 'ascending':
            for i in range(0,len(bin_table.index)-1):
                if bin_table.event_rate[i] > bin_table.event_rate[i+1]: 
                    bin_table.loc[i,'bounce_pct']  = (bin_table.event_rate[i+1]-bin_table.event_rate[i])/br_range

        # multilevel column index
        bin_table.columns = pd.MultiIndex.from_product([[dataset_name], bin_table.columns])
        return bin_table
        
            

    
    def calculate_psi(self, m_binned_data, v_bin_table, dataset_name):        

        # distribution of modeling dataset
        binned_df = pd.DataFrame(m_binned_data)
        binned_df['total'] = 1
        grouped = binned_df.groupby(['bin'], as_index = False)
        tmp_bin_table = pd.DataFrame(grouped.count(), columns = ['bin', 'total'])
        tmp_bin_table['p_total'] = (tmp_bin_table.total / tmp_bin_table.total.sum())        

        # calculate psi
        v_bin_table[dataset_name,'PSI'] = round((tmp_bin_table['p_total'] - v_bin_table[dataset_name,'p_total']) * np.log(tmp_bin_table['p_total'] / v_bin_table[dataset_name,'p_total']),5)        

        return v_bin_table





    def merge_bin_tables(self, bin_table_m, bin_table_v, v_dataset_name):
        '''
        Merge binning tables of different datasets into one single multiindex dataframe. Then, calculate PSI between the given validation dataset and modeling dataset. 
        Input of 'v_dataset_name' parameter should be datasets other than modeling dataset.
        
        **Input**
            ================ ============================= ==============================================================
            Parameter        Data Type                     Description
            ================ ============================= ==============================================================
            bin_table_m      ``pd.DataFrame``              The bin table of Modeling dataset.
            bin_table_v      ``pd.DataFrame``              The bin table of any given Validation dataset.
            v_dataset_name   ``str``                       Self-defined dataset name. Suggested format: *validaion_02*.
            ================ ============================= ==============================================================
            
        **Output** 
            ============ ============================= ==================================================================
            Parameter    Data Type                     Description
            ============ ============================= ==================================================================
            bin_table    ``pd.DataFrame``              Return performance calculation of M & V datasets in each bins. 
            bin_table_v  ``pd.DataFrame``              Return performance calculation of V dataset in each bins.
            ============ ============================= ==================================================================        
        
        '''
        # concat bin tables
        bin_table = pd.concat([bin_table_m, bin_table_v], axis=1)
        return bin_table


 

    def calculate_metrics (self, y, y_pred):
        '''
        calculate accuracy, recall, f1-score and auc of predicted datasets. See more in `sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_ .
        '''        
        from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, classification_report    
        import warnings
        import sklearn.exceptions
        warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) # shut down the warning for not having a complete metrix 
        
        accuracy = round(accuracy_score(y, y_pred),5)
        recall = round(recall_score(y, y_pred, average = 'macro'),5)
        f1 = round(f1_score(y, y_pred),5)
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        auc = round(auc(fpr, tpr),5)           
        return (accuracy, recall, f1, auc)
        # [說明]
        # http://www.cnblogs.com/robert-dlut/p/5276927.html
        # recall:召回率，TP/(TP+FN)，y事件樣本中被預測為y的比例，模型對正樣本的識別能力，越高越強
        # precision:準確率，TP/(TP+FP)，預測為y的樣本中確實為y的比例，模型對負樣本的區分能力，越高越強
        # f1 score:以上兩者綜合，越高模型越穩 (weighted average of precision and recall)


    def logistic_coef_check(self, modeler):
        coef = modeler.fitted_clf.coef_ 
        coef_df = pd.DataFrame(dict(zip(modeler.variable_orders, coef[0])), index=['coefficient']).T     
        corr = {'variables': list(modeler.variable_orders),'correlation': modeler.corr_list}
        corr_df = pd.DataFrame(corr)
        corr_df = corr_df.set_index('variables')

        lr_correction_factors = []   
        for i in list(coef_df.index):
            if coef_df.coefficient[i]*corr_df.correlation[i] < 0:
                lr_correction_factors.append(i)

        lr_minimum_coef = coef_df.coefficient.abs().min()   
        return lr_correction_factors, lr_minimum_coef


    def tree_featureimportance_check(self, modeler):
        feature_importance = modeler.fitted_clf.feature_importances_ 
        fi_df = pd.DataFrame(dict(zip(modeler.variable_orders, feature_importance)), index=['feature_importance']).T
        
        tree_minimum_fi = fi_df.feature_importance[feature_importance != 0].abs().min()        
        tree_zero_fi = []
        for i in list(fi_df.index):
            if fi_df.feature_importance[i] == 0:
                tree_zero_fi.append(i)
        return tree_minimum_fi, tree_zero_fi


    def bad_rate_bounce_check(self, bin_table, dataset_name):
        br = (bin_table[dataset_name,'event'] / bin_table[dataset_name,'total'])
        bounce_cnt = 0
        for i in range(0,len(list(br.index))-1):
          if br[i] < br[i+1]: 
              bounce_cnt += 1
        return bounce_cnt



    def organize_sample_performance(self, modeler, y, y_pred, bin_table, dataset_name):
        '''
        Summary performance of the given dataset, and append to the original performance table.
        '''   
        # lr_correction_factors, lr_minimum_coef
        try:
            if dataset_name.find('modeling') != -1: 
                if modeler.model_desc.find('LogisticRegression') != -1:
                    lr_correction_factors, lr_minimum_coef= self.logistic_coef_check(modeler)

                else:
                    lr_correction_factors, lr_minimum_coef = '', ''
            else:
                lr_correction_factors, lr_minimum_coef = '', ''
        except AttributeError:
            lr_correction_factors, lr_minimum_coef = '', ''                

        # tree_importance
        if dataset_name.find('modeling') != -1:
            if modeler.model_desc.find('DecisionTree') != -1:
                tree_minimum_fi, tree_zero_fi = self.tree_featureimportance_check(modeler)
            else:
                tree_minimum_fi, tree_zero_fi = '', ''
        else:
            tree_minimum_fi, tree_zero_fi = '', ''
        
        # accuracy, recall, f1, auc
        try:
            accuracy, recall, f1, auc = self.calculate_metrics(y, y_pred)        
        except ValueError:
            accuracy, recall, f1, auc  = '', '', '', ''
        
        # PSI
        if dataset_name.find('modeling') == -1:         
            PSI = bin_table[dataset_name,'PSI'].abs().sum()
        else:
            PSI = ''
        
        # bad rate bounce
        bounce_cnt = self.bad_rate_bounce_check(bin_table, dataset_name)
        bounce_pct = bin_table[dataset_name, 'bounce_pct'].fillna(0).abs().sum()

        # lift
        min_br = bin_table[dataset_name,'event_rate'][bin_table[dataset_name,'event_rate'] != 0].min()
        max_br = bin_table[dataset_name,'event_rate'].max()            
        lift = (max_br/min_br)
        
        # bin_cnt
        bin_cnt = bin_table[dataset_name,'bin'].count()
        
        # overall event rate
        overall_y = (bin_table[dataset_name,'event'].sum()/bin_table[dataset_name,'total'].sum())
        
        s =  pd.Series({'cnt':bin_table[dataset_name,'total'].sum(), 'Y%': overall_y, 
                        'KS':bin_table[dataset_name,'KS'].abs().max(), 'Lift': lift, 'PSI':PSI,
                        'bounce_cnt':bounce_cnt, 'bounce_pct':bounce_pct, 'bin_cnt':bin_cnt,
                        'LR_Correction':lr_correction_factors, 'LR_Mincoef':lr_minimum_coef, 'DT_Minfi':tree_minimum_fi, 'DT_Zerofi':tree_zero_fi,
                        'IV':bin_table[dataset_name,'IV'].sum(), 
                        'accuracy':accuracy, 'recall':recall, 'f1':f1, 'auc':auc}, name = dataset_name)
        return s       


        
# inner components for scoring dataset
# =============================================================================

    def create_scoring_bin_table(self, binned_data, y_proba, dataset_name): 
        '''
        Group dataset into bins.
        '''                 
        # data prep
        y_proba = pd.Series(y_proba, name = 'y_proba')
        df = pd.concat([binned_data, y_proba], axis = 1)            

        # create bin table 
        grouped = df.groupby('bin', as_index = False)
        bin_table = pd.DataFrame(grouped.min().bin, columns = ['bin'])  
        bin_table['min_proba'] = grouped.min().y_proba
        bin_table['max_proba'] = grouped.max().y_proba
        bin_table['total'] = grouped.count().y_proba
        bin_table['p_total'] = (bin_table.total / bin_table.total.sum())
        
        # multilevel column index
        bin_table.columns = pd.MultiIndex.from_product([[dataset_name], bin_table.columns])
        return bin_table
    
 
       
    def organize_scoring_performance(self, bin_table, dataset_name):
        '''
        Summary performance of the given dataset, only for scoring (without Y).
        '''         
        s =  pd.Series({'cnt':bin_table[dataset_name,'total'].sum(),'PSI':bin_table[dataset_name,'PSI'].abs().sum()}, name = dataset_name)
        return s   
        



    def lift_plotting(self, modeler, n_bins):
        from sources.feature_profiling import LiftPlotter

        boundry = self.define_boundry(modeler.v_y_proba, n_bins)
        binned_data = self.assign_bin(modeler.v_y_proba, boundry)
#        print(binned_data.head(10))
        
        x = pd.DataFrame(binned_data, columns = ['bin'])
        y = modeler.v_y
        
        plotter = LiftPlotter()
        plotter.plot(x, y)
        plotter.to_pdf(filepath="Models\\", model_id=modeler.model_id)
        
    
    
    
class MyEvaluator(Evaluator): 
    def evaluate_modeling_validation(self, m_y, m_y_proba, m_y_pred, m_bin, v_y, v_y_proba, v_y_pred, v_bin, modeler=None): 
        m_binned_data = m_bin.copy(deep=True)
        m_binned_data.columns = ['bin']
        m_bin_table = self.create_bin_table(m_binned_data, m_y, m_y_proba, dataset_name = 'modeling')
        
        v_binned_data = v_bin.copy(deep=True)
        v_binned_data.columns = ['bin']
        v_bin_table = self.create_bin_table(v_binned_data, v_y, v_y_proba, dataset_name = 'validation')
        v_bin_table = self.calculate_psi(m_binned_data, v_bin_table, dataset_name = 'validation') 
        
#        bin_table = m_bin_table
        bin_table = self.merge_bin_tables(m_bin_table, v_bin_table, v_dataset_name = 'validation')
        
        s_modeling = self.organize_sample_performance(modeler, m_y, m_y_pred, m_bin_table, dataset_name = 'modeling')
        s_validation = self.organize_sample_performance(modeler, v_y, v_y_pred, v_bin_table, dataset_name = 'validation')
#        performance = s_modeling
        performance = pd.concat([s_modeling, s_validation], axis=1)
        
        return bin_table, performance


    def evaluate_extra_validation(self, y, y_pred, y_proba, y_proba_bin, m_bin, modeler_bin_table, modeler_performance, dataset_name):
        '''
        Evaluate *one validation dataset* after model was built. Information of this dataset will be append to existing performance/bin_table.
        '''                               
        modeler = None
        m_binned_data = m_bin.copy(deep=True)
        m_binned_data.columns=['bin']
        binned_data = y_proba_bin.copy(deep=True)
        binned_data.columns=['bin']
        
        extra_v_bin_table = self.create_bin_table(binned_data, y, y_proba, dataset_name)  
        extra_v_bin_table = self.calculate_psi(m_binned_data, extra_v_bin_table, dataset_name) 
        bin_table = self.merge_bin_tables(modeler_bin_table, extra_v_bin_table, dataset_name)
        
        s_extra_validation = self.organize_sample_performance(modeler, y, y_pred, extra_v_bin_table, dataset_name)
        performance = pd.concat([modeler_performance, s_extra_validation], axis=1)
 
        return bin_table, performance    