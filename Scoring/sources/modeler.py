# -*- coding: utf-8 -*-
"""
``Modeler`` is an object that implements the whole process of modeling , including model training, fitting, and performance evaluation. \n
One single ``Modeler`` stands for one single model. After ``Modeler`` is created, the object ``Modeler`` can hold:
    - Train the classifier to build a model (**fit_and_transform_evaluate**)
    - Insert additional validation datasets to evaluate the existed model (**transform_and_evaluate**)
    - Insert a set of predicted values to evaluate model performance without doing fit_and_transform (**evaluate_without_clf**)
    - Score datasets to get predicted values (**simply_scoring**)
    - Being collected by ``ModelerCollector`` for further model comparing and selection
    
@author: Heidi

"""

import pandas as pd 
import numpy as np
import datetime
import copy
import re
from sources.file_controller import FileController
from sources.evaluator import Evaluator


class Modeler(): 
    """
    """
    clf=None
    
    def __init__(self, clf): 
        self.__clf = clf 
        try:
            string = str(clf)
            regex = re.compile(r'(.*)[(]((?:.*\n*)*?)[)]')
            match = regex.search(string)
            self.__model_desc = match.group(1)
        except AttributeError:
            self.__model_desc = ''  # model desc would be blank when input datasets are not built within python.


    def __str__(self):       
        return '\n******************** {0} ********************\n\n******************** {1} ********************\n{2}\n\n******************** {3} ********************\n{4}\n'.format(
                self.__model_desc, 'fitted_clf', self.__fitted_clf ,'performance',self.__performance)
    
    @property
    def fitted_clf(self):
        """
        The fitted/trained classifier, a model. The component differs according to the algorithm.
        
        Data Type: ``Classifier``
        """
        return self.__fitted_clf
    
    @property
    def performance(self):
        """
        Provides summary information of model performance in each dataset.
        
        Data Type: ``pandas.DataFrame`` 
        """
        return self.__performance
    
    @property
    def bin_table(self):
        """
        Provides detail infromation of model performance in each dataset and bins. 
        
        Data Type: ``pandas.DataFrame``
        """
        return self.__bin_table
    
    
    @property
    def model_desc(self):
        """
        The name of the classifier, basically indicates the algorithm/training methods .
        
        Data Type: ``str``
        """
        return self.__model_desc
    
    @property
    def model_id(self):
        """
        The id of the model. ``model_id`` will be given after ``Modeler`` is collected into ``ModelerCollector``.
        
        Data Type: ``str``
        """
        return self.__model_id
    
    @property
    def parameters(self):
        """
        The parameters set in the classifier.
        
        Data Type: ``str``
        """
        if self.__fitted_clf is None:
            parameters = None        
        else:
            string = str(self.__fitted_clf)
            regex = re.compile(r'(.*)[(]((?:.*\n*)*?)[)]')
            match = regex.search(string)
            param = match.group(2)
            parameters = self.cleanse_parameters(param)        
        return parameters

    @property
    def variable_orders(self):
        return self.__variable_orders

    @property
    def corr_list(self):
        '''
        Spearman correlation coefficient of each variable, varies between -1 and +1 with 0 implying no correlation. Calaulation base on Modeling dataset.
        
        Data Type: ``List`` (n_variables,)
        '''
        return self.__corr_list
 
    
    @property
    def pvalue_list(self):
        '''
        The p-value to test for non-correlation. Calaulation base on Modeling dataset.
        
        Data Type: ``List`` (n_variables,)
        '''    
        return self.__pvalue_list


# use in main
# =============================================================================
   
    def fit_transform_evaluate(self, modeling_dataset, validation_dataset, n_bins): 
        """
        Train the model, get prediction data, and then evaluate model performance.

        **Input** 
            =================== ===================================================== =================================================
            Parameter           Data Type                                             Description
            =================== ===================================================== ================================================= 
            modeling_dataset    ``Dataset`` (see module `data_set <data_set.rst>`_)   Modeling dataset after preprocessing.
            validation_dataset  ``Dataset`` (see module `data_set <data_set.rst>`_)   Validation dataset after preprocessing.
            n_bins              ``int``                                               The number of bins to cut in evaluation process.
            =================== ===================================================== =================================================
            
        **Output** 
            None. The evaluation result can be called in ``Modeler.bin_table`` and ``Modeler.performance``.
            
        """
        # data prep
        m_x = modeling_dataset.x
        m_y = modeling_dataset.y
        v_x = validation_dataset.x
        v_y = validation_dataset.y
        self.__variable_orders = list(modeling_dataset.attribute_list.x_list)
        self.__corr_list, self.__pvalue_list = self.calculate_correlation_and_p_value(m_x, m_y)        
        
        # train model                
        self.__clf.fit(m_x, m_y)
        self.__fitted_clf = copy.deepcopy(self.__clf)   
        
        # get predicted value
        self.m_y_pred, self.m_y_proba = self.predict_and_sync_index(m_x)               
        self.v_y_pred, self.v_y_proba = self.predict_and_sync_index(v_x)       
        
        # evaluation (call <Evaluator>)
        self.__bin_table, self.__performance = Evaluator().evaluate_modeling_validation(self, m_y, self.m_y_pred, self.m_y_proba, v_y, self.v_y_pred, self.v_y_proba, n_bins)  



    def set_model_id(self, model_id):
        """
        Return the model_id into ``Modeler`` after modeler was collected into ``ModelerCollector``. (model_id is given in ``ModelerCollector``.)
        """
        self.__model_id = model_id 


    def output_to_file(self, filepath):
        """
        Output the modeler into files, including current model information and evaluation results. 
        For more details, see function **save_pkl_to_file**, **save_perf_to_xlsx**, **decision_tree_dot_and_plot**.
        """
        self.save_pkl_to_file(filepath)
        self.save_evaluation_to_xlsx(filepath)        
        if self.__model_desc.find('DecisionTree') != -1: 
            self.decision_tree_dot_and_plot()
 


    def transform_and_evaluate(self, validation_dataset, n_bins, name_of_dataset):
        '''
        Insert a new set of validation data to existed ``Modeler``. The function deals with predicting event probability and returning the evaluation result.

        **Input** 
            ===================== ==================================================== =============================================================
            Parameter             Data Type                                            Description
            ===================== ==================================================== =============================================================
            validation_dataset    ``Dataset`` (see module `data_set <data_set.rst>`_)  Validation dataset (with Y) after pre-processing.
            n_bins                ``int``                                              The number of bins to cut in evaluation process.                           
            name_of_dataset       ``str``                                              Self-defined dataset name. Suggested format: *validation_02*.
            ===================== ==================================================== =============================================================
        
        **Output** 
            None. The evaluation result can be called in ``Modeler.bin_table.name_of_dataset`` and ``Modeler.performance.name_of_dataset``.

        '''
        validation_dataset.order_x(self.__variable_orders)
        x = validation_dataset.x
        y = validation_dataset.y
        y_pred, y_proba = self.predict_and_sync_index(x)      
        # add new columns to existing bin_table and performance_table
        self.__bin_table, self.__performance = Evaluator().evaluate_extra_validation(y, y_pred, y_proba, n_bins, self.m_y_proba, self.__bin_table, self.__performance, dataset_name=name_of_dataset)
 
       

#    def evaluate_without_clf(self, m_y, m_y_pred, m_y_proba, v_y, v_y_pred, v_y_proba, n_bins, external_model_desc):       
#        '''
#        Evaluate model performance using real Y and predicted values, thus NO classifier is needed during the process.
#
#        **Input** 
#            ===================== ============================== ====================================================================================================================================
#            Parameter             Data Type                      Description
#            ===================== ============================== ====================================================================================================================================
#            m_y                   ``pd.Series`` or ``np.array``  Real Y of modeling dataset.
#            m_y_pred              ``pd.Series`` or ``np.array``  Predicted Y of modeling dataset. *Not necessarily required*. Setting it ``None`` will skip the calculation of accuracy/recall. 
#            m_y_proba             ``pd.Series`` or ``np.array``  Prediced probability of modeling dataset. *Cannot be skipped*. Could be replaced with scores or anything that enables rank ordering.                                    
#            v_y                   ``pd.Series`` or ``np.array``  Real Y of validation dataset.
#            v_y_pred              ``pd.Series`` or ``np.array``  Predicted Y of validation dataset. Not necessarily required.
#            v_y_proba             ``pd.Series`` or ``np.array``  Prediced probability of validation dataset. Cannot be skipped.                                    
#            external_model_desc   ``str``                        Self-defined model description. Suggested format: *SAS_Logistic*.
#            ===================== ============================== ====================================================================================================================================
#        
#        **Output** 
#            None. The evaluation result can be called in ``Modeler.bin_table`` and ``Modeler.performance``.
#        
#        '''
#        self.__fitted_clf = None
#        self.__model_desc = external_model_desc
#        self.__boundry, self.__bin_table, self.__performance = Evaluator().evaluate_modeling_validation(None, m_y, m_y_pred, m_y_proba, v_y, v_y_pred, v_y_proba, n_bins)



    def evaluate_without_clf(self, m_filepath, v_filepath, y_column, proba_column, n_bins, external_model_desc):
        from sources.file_controller import DataframeFileController
        self.__fitted_clf = None
        self.__model_desc = external_model_desc
        
        # data prep
        m_data = DataframeFileController(m_filepath).load()
        m_y = m_data.loc[:,y_column]
        m_y_proba = m_data.loc[:,proba_column]
        m_y_pred = None
        v_data = DataframeFileController(v_filepath).load()
        v_y = v_data.loc[:,y_column]
        v_y_proba = v_data.loc[:,proba_column]
        v_y_pred = None
        
        # evaluate
        self.__bin_table, self.__performance = Evaluator().evaluate_modeling_validation(self, m_y, m_y_pred, m_y_proba, v_y, v_y_pred, v_y_proba, n_bins)
        


    def simply_scoring(self, scoring_dataset, n_bins, name_of_dataset):
        '''
        Return predicted values and evaluation results of given datasets.

        **Input** 
            ===================== ==================================================== =============================================================
            Parameter             Data Type                                            Description
            ===================== ==================================================== =============================================================
            scoring_dataset       ``Dataset`` (see module `data_set <data_set.rst>`_)  Scoring dataset after pre-processing (Y is not required).
            n_bins                ``int``                                              The number of bins to cut in evaluation process.                                    
            dataset_name          ``str``                                              Self-defined dataset name. Suggested format: *scoring_01*.
            ===================== ==================================================== =============================================================
        
        **Output** 
            ===================== ================ =============================================================
            Parameter             Data Type        Description
            ===================== ================ =============================================================
            y_pred                ``pd.Series``    The prediction of event. Range: [0,1].
            y_proba               ``pd.Series``    The prediction of event probability. Range: [0~1].                             
            ===================== ================ =============================================================             
         
        '''
        scoring_dataset.order_x(self.__variable_orders)
        s_x = scoring_dataset.x
        y_pred, y_proba = self.predict_and_sync_index(s_x)         
        # add new columns to existing bin_table and performance_table
        self.__bin_table, self.__performance = Evaluator().evaluate_scoring(y_proba, n_bins, self.m_y_proba, self.__bin_table, self.__performance, dataset_name=name_of_dataset)        
        return y_pred, y_proba




    def predict_and_sync_index(self, x):   
        """
        Make predictions, then sync the index of original input dataset and predicted data (keep the original index).
        
        **Input** 
            =========== ========================================== 
            Parameter   Data Type                                 
            =========== ========================================== 
            x           ``pd.DataFrame`` (n_samples, n_variables)                              
            =========== ==========================================
        
        **Output** 
            ============= ================ ====================================================
            Parameter     Data Type        Description
            ============= ================ ====================================================
            y_pred        ``pd.Series``    The prediction of event. Range: [0,1].
            y_proba       ``pd.Series``    The prediction of event probability. Range: [0~1].                             
            ============= ================ ====================================================
            
        """        
        y_pred = pd.Series(self.__fitted_clf.predict(x), name = 'y_pred')
        y_proba = pd.DataFrame(self.__fitted_clf.predict_proba(x))     
        y_proba = pd.Series(y_proba.loc[:,1], name = 'y_proba')
        
        #sync index (for discontinuous index situation, such as scorecard)
        pd.options.mode.chained_assignment = None  # silence the warning
        y_pred.index = x.index
        y_proba.index = x.index     
        
        return y_pred, y_proba
        


    def save_pkl_to_file(self, filepath):
        """
        Save ``Modeler`` to file as an pickle object.
        """
        pklpath = '{0}\{1}.pkl'.format(filepath, self.__model_id)
        file_controller = FileController(pklpath)
        file_controller.save(self)
        print('*** pkl has been saved to file: {} ***'.format(pklpath))



    def save_evaluation_to_xlsx(self, filepath):     
        """
        Save mevaluation results into Excel worksheet.
            - Classifier (algorithm, parameters)
            - performance
            - bin_table
            - formula (Linearmodels)
            - feature importance (Tree/RandomForest)
        """
        xlsxpath = '{0}\{1}.xlsx'.format(filepath, self.__model_id)
        with pd.ExcelWriter(xlsxpath) as writer:  
           
            d = {'clf': [self.__model_desc], 'parameters': [self.parameters]}
            df = pd.DataFrame.from_dict(d)
            df.to_excel(writer, sheet_name='fitted_clf')
            self.__performance.to_excel(writer, sheet_name='performance')
            self.__bin_table.to_excel(writer, sheet_name='bin_table')
            
            if self.__model_desc.find('LogisticRegression') != -1: 
                formula = self.formula_of_linearmodels() 
                formula.to_excel(writer, sheet_name='formula')
                
            if self.__model_desc.find('DecisionTree') != -1: 
                fi_table = self.feature_importance_of_treemodels() 
                fi_table.to_excel(writer, sheet_name='feature_importance')
                
            if self.__model_desc.find('RandomForest') != -1: 
                fi_table = self.feature_importance_of_treemodels() 
                fi_table.to_excel(writer, sheet_name='feature_importance')                
        print('*** xlsx has been saved to file: {} ***'.format(xlsxpath))

   
    
    def formula_of_linearmodels(self):  
        """
        Data type: ``pd.DataFrame``
        """
        intercept = self.__fitted_clf.intercept_
        coefficient = self.__fitted_clf.coef_      
        df0 = pd.DataFrame({'coefficient': intercept}, index=['intercept'])
        df0.index.name = 'Variable'
        df1 = pd.DataFrame(dict(zip(self.__variable_orders, coefficient[0])), index=['coefficient']).T
        df1.index.name = 'Variable'
        formula = df0.append(df1)                   
        return formula        


    def feature_importance_of_treemodels(self):
        """
        Data type: ``pd.DataFrame``
        """
        feature_importance = self.__fitted_clf.feature_importances_
        fi_table = pd.DataFrame(dict(zip(self.__variable_orders, feature_importance)), index=['feature_importance']).T
        fi_table.index.name = 'Variable'                   
        return fi_table


    def decision_tree_dot_and_plot(self):
        """
        Save decision nodes to file in both txt abd pdf format.
        """
        self.create_tree_dot()
        self.create_tree_plot()
        print('*** dot and plot has been saved to file: Models\Decision_Tree_Classifier\ ***')
        

    def create_tree_dot(self):
        filepath = 'Models\\{}.dot'.format(self.__model_id)
        from sklearn import tree        
        with open(filepath, 'w') as f:
            dot_data = tree.export_graphviz(self.__fitted_clf, out_file=None, feature_names = self.__variable_orders)
            f.write(dot_data)        
    
    
    def create_tree_plot(self):
        import pydotplus    
        from sklearn import tree
        filepath = 'Models\\{}.pdf'.format(self.__model_id)
        dot_data = tree.export_graphviz(self.__fitted_clf, out_file=None, feature_names = self.__variable_orders)   
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(filepath)         
        


    def cleanse_parameters(self,text): 
        """
        Text cleansing for parameters.
        """
        if(text==None): 
            return None 
        else: 
            import re
            text = re.sub('[\s\t]*?\n[\s\t]*?','', text)
            text = re.sub(' ','', text)
            text = re.sub(',',', ', text)
            return text


    def calculate_correlation_and_p_value(self, x, y):
        """       
        Calculate a Spearman rank-order correlation coefficient and the p-value to test for non-correlation. 
        See more in `Scipy.stats.spearmanr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html>`_ .
        
        **Output** 
            ============= ======================== ====================================================================
            Parameter     Data Type                Description
            ============= ======================== ====================================================================
            corr_list     ``List`` (n_variables,)  Used in LogisticRegression to determine correction factors (修正項).
            pvalue_list   ``List`` (n_variables,)  Currently not in use.                          
            ============= ======================== ====================================================================

        """
        from scipy import stats   
        corr_list = []
        pvalue_list = []
        for i in self.__variable_orders:
            corr, pvalue = stats.spearmanr(x[i], y)
            corr_list.append(corr)
            pvalue_list.append(pvalue)
        return corr_list, pvalue_list 
            