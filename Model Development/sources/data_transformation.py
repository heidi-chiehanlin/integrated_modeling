# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:04:51 2019

@author: chialing
"""

from sources.file_controller import FileController 
from sources.data_set import DataSet
import copy
import pandas as pd

class __DataTransformer(): 
    def fit_transform(self, source_dataset, fake_y=None): 
        self.fit(source_dataset)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset
    
    def to_pkl(self, filepath): 
        """
        """
        file_controller = FileController(filepath = filepath)
        file_controller.save(self)

class ManualTransformer(__DataTransformer): 
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

#from sources.data_transformation import ManualTransformer
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


class RangeLowerUpper():
    """
    """
    def __init__(self, feature, lower, upper): 
        self.feature = feature
        self.lower = lower 
        self.upper = upper 
    
    def __str__(self): 
        return "{0}: {1:,.2f} ~ {2:,.2f}\n".format(self.feature, self.lower, self.upper)

class RangeLimiter(): 
    def __init__(self): 
        self.__range_list = dict()
        
    def __str__(self): 
        str_ = ""
        for feature in self.__transform_features: 
            str_ = str_ + "{}".format(self.__range_list[feature])
        return str_

    def fit_transform(self, source_dataset, fake_y=None, method='auto'): 
        """
        """
        self.fit(source_dataset, fake_y=fake_y, method='auto')
        destination_dataset = self.transform(source_dataset)
        return destination_dataset
    
    def fit(self, dataset, fake_y=None, method='auto'): 
        """
        Uses Winsorizing methodology: Sets upper / lower bounds for feature values        
        - 'quantile_range': Upper = q3 + 1.5*iqr , Lower = q1 - 1.5*iqr        
        - 'standard_deviation': Upper = mean + 3*standard_deviation , Lower = mean - 3*standard_deviation 
        - 'auto': Uses 'quantile_range' as default, but if q3-q1=0, then use 'standard_deviation'
        
        * Reference Site: https://heartbeat.fritz.ai/how-to-make-your-machine-learning-models-robust-to-outliers-44d404067d07
        
        """
        self.create_range_limits(dataset, method)
        self.set_transform_features(exclude='all')
    
    def create_range_limits(self, dataset, method): 
        """
        """
        x = dataset.x_numeric.copy(deep=True)
        for feature in x.columns: 
            range_ = self.__create_feature_range_limit(x[[feature]], method=method)
            self.__range_list[feature] = range_
        
        self.__transform_features = list(x.columns)
        print(self.__str__())


    def set_transform_features(self, exclude='all'): 
        """
        """
        if(exclude == 'all'): 
            self.__transform_features = self.__transform_features
        else: 
            self.__transform_features = [x for x in self.__transform_features if x not in exclude]

        
    def transform(self, dataset, fake_y=None): 
        """
        """
        x_limited = dataset.x_numeric.copy(deep=True)
        for feature in self.__transform_features: 
            upper_threshold = self.__range_list[feature].upper 
            lower_threshold = self.__range_list[feature].lower
            
            x_limited[feature] = x_limited[feature].clip(lower=lower_threshold, upper=upper_threshold)
        
        limited_dataset = self.__create_dataset(x_numeric = x_limited, original_dataset = dataset)
        limited_dataset.order_x_by_attribute_list()
        
        return limited_dataset
    
    
    def __create_dataset(self, x_numeric, original_dataset): 
        data_set = DataSet()
        data_set.set_all_by_parts(x_nominal = original_dataset.x_nominal, 
                                  x_numeric = x_numeric, 
                                  y = original_dataset.y, 
                                  attribute_list = original_dataset.attribute_list)
        return data_set
    
    
    
    def __create_feature_range_limit(self, feature_df, method='auto'): 
        if(method=='auto'): 
            range_ = self.__create_feature_range_limit_quantile(feature_df)
            
            if(range_.upper == range_.lower): 
                range_ = self.__create_feature_range_limit_std_dev(feature_df)
            
        elif(method=='quantile'): 
            range_ = self.__create_feature_range_limit_quantile(feature_df)
            
        elif(method == 'std_dev'): 
            range_ = self.__create_feature_range_limit_std_dev(feature_df)
        
        return range_


    def __create_feature_range_limit_quantile(self, feature_df): 
        feature = feature_df.columns[0]
        feature_df = feature_df[feature]
#        print(feature, type(feature_df[0]))
        
        q3 = feature_df.quantile(.75)
        q1 = feature_df.quantile(.25)
        inter_quantile_range = q3-q1 
#        print(q3, q1)
#        print(inter_quantile_range)
        
        lower_threshold = q1-inter_quantile_range*1.5
        upper_threshold = q3+inter_quantile_range*1.5
        
        range_ = RangeLowerUpper(feature, lower_threshold, upper_threshold)
        
        return range_

    def __create_feature_range_limit_std_dev(self, feature_df): 
        feature = feature_df.columns[0]
        
        mean = feature_df[feature].mean()
        std = feature_df[feature].std()
        
        upper_threshold = mean+3*std
        lower_threshold = mean-3*std
        
        range_ = RangeLowerUpper(feature, lower_threshold, upper_threshold)
        
        return range_
    
    def to_pkl(self, filepath = "Models\\range_limiter.pkl"): 
        """
        """
        file_controller = FileController(filepath = filepath)
        file_controller.save(self)
    
#%%
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

class Imputer(__DataTransformer): 
    def __init__(self): 
        pass

    def __str__(self): 
        str_ = "{}".format(self._imputer)
        return str_

    def to_pkl(self, filepath = "Models\\NominalImputer.pkl"): 
        """
        """
        file_controller = FileController(filepath = filepath)
        file_controller.save(self)

    def fit_transform(self, source_dataset, fake_y=None, strategy=None): 
        self.fit(source_dataset, strategy=strategy)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset

class NominalImputer(Imputer): 
    """
    """
    def fit(self, dataset, fake_y=None, strategy="most_frequent"): 
        """
        - most frequent
        - constant (default: null_string)
        """
        if(dataset.x_nominal.empty): 
            self._imputer = None 
        else: 
            x_nominal = dataset.x_nominal
            self._imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            self._imputer.fit(x_nominal)
    
    def transform(self, source_dataset, fake_y=None): 
        """
        """
        if(source_dataset.x_nominal.empty): 
            print("No nominal feature to impute")
            return source_dataset
        else: 
            x_nominal = source_dataset.x_nominal 
            
            destination_x_nominal = self._imputer.transform(x_nominal)
            destination_x_nominal = pd.DataFrame(destination_x_nominal, columns=source_dataset.x_nominal.columns)
            destination_x_nominal.index = source_dataset.x_nominal.index
            
            destination_dataset = DataSet()
            destination_dataset.set_all_by_parts(destination_x_nominal, source_dataset.x_numeric, source_dataset.y, source_dataset.attribute_list)
            
            destination_dataset.order_x_by_attribute_list()
            
            return destination_dataset 

class NumericImputer(Imputer): 
    """
    """
    def fit(self, dataset, fake_y=None, strategy="median"): 
        """
        - median 
        - mean 
        - constant (default: 0)
        """
        if(dataset.x_numeric.empty): 
            self._imputer = None 
        else: 
            x_numeric = dataset.x_numeric.copy(deep=True)
            
            self._imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            self._imputer.fit(x_numeric)
    
    def transform(self, source_dataset, fake_y=None): 
        """
        """
        if(source_dataset.x_numeric.empty): 
            print("No numeric feature to impute")
            return source_dataset
        else: 
            x_numeric = source_dataset.x_numeric.copy(deep=True)
            
            destination_x_numeric = self._imputer.transform(x_numeric.values)
#            print(destination_x_numeric.head())
            print(type(destination_x_numeric))
            
            destination_x_numeric = pd.DataFrame(destination_x_numeric, columns=source_dataset.x_numeric.columns)
            destination_x_numeric.index = source_dataset.x_numeric.index
            
            destination_dataset = DataSet()
            destination_dataset.set_all_by_parts(source_dataset.x_nominal, destination_x_numeric, source_dataset.y, source_dataset.attribute_list)
            
            destination_dataset.order_x_by_attribute_list()
            
            return destination_dataset 
        
        return None

#%%
from sklearn.feature_extraction import DictVectorizer
import re
import copy

class OneHotEncoder(__DataTransformer): 
    """
    | Class: 
    | - holds methods to fit and transform nominal features with ``sklearn.feature_extraction.DictVectorizer``
    """
    def __init__(self): 
        pass
    
    def __str__(self): 
        if(self.__vectorizer is None): 
            return "No nominal feature to encode"
        else: 
            str_ = "{}".format(self.__vectorizer)
            return str_
    
    def fit(self, dataset, fake_y=None): 
        """
        Function fits vectorizer with nominal features in the dataset 
        
        **Args**
            +--------------+--------------------------------------------------------------------------------+-----------------------------------------+
            |**Parameter** |**Data Type**                                                                   |**target_dataset.x_nominal Content**     |
            +--------------+--------------------------------------------------------------------------------+-----------------------------------------+
            |       dataset|``DataSet`` (see module `data_set <data_set.html>`_)                            |                                       A1| 
            |              |                                                                                +-----------------------------------------+
            |              |                                                                                |                                  1/2/3/4|
            +--------------+--------------------------------------------------------------------------------+-----------------------------------------+
            
        **Returns**
            None
        
        """
        
        if(dataset.x_nominal.empty): 
            self.__vectorizer = None 
        else: 
            x_nominal = self.__to_dict_vectorizer_format(dataset.x_nominal)
    
            self.__vectorizer = DictVectorizer(sparse=False, separator='=', dtype=int)
            self.__vectorizer.fit(x_nominal)
    
    def transform(self, source_dataset, fake_y=None): 
        """
        Function transforms nominal features in the target_dataset to vectorized format with fitted result
        
        **Args**
            +--------------+--------------------------------------------------------------------------------+-----------------------------------------+
            |**Parameter** |**Data Type**                                                                   |**target_dataset.x_nominal Content**     |
            +--------------+--------------------------------------------------------------------------------+-----------------------------------------+
            |source_dataset|``DataSet`` (see module `data_set <data_set.html>`_)                            |                                       A1| 
            |              |                                                                                +-----------------------------------------+
            |              |                                                                                |                                  1/2/3/4|
            +--------------+--------------------------------------------------------------------------------+-----------------------------------------+
            
        **Returns**
            +--------------------+-------------------------------------------------------------------------------------------+-----------------------+
            |**Parameter**       |**Data Type**                                                                              |**Content**            |
            +--------------------+-------------------------------------------------------------------------------------------+-----+-----+-----+-----+
            |destination_datset  |``pandas.DataFrame``                                                                       | A1=1| A1=2| A1=3| A1=4| 
            |                    |(number_of_samples, number_of_nominal_features\*number_of_values_each_feature)             +-----+-----+-----+-----+
            |                    |                                                                                           |    1|    0|    0|    0|
            +--------------------+-------------------------------------------------------------------------------------------+-----+-----+-----+-----+
            
        **Raises**
            =============== ================================= =================================================================================================================
            Exception       Scenario                          Description
            =============== ================================= =================================================================================================================
            ValueError      Failed to apply one-hot encoding  Values of a nominal attribute in target_datatset.x exceeds expected values set by fit_and_transform() function
            =============== ================================= =================================================================================================================
            
        """
        if(source_dataset.x_nominal.empty): 
            return source_dataset
        else: 
            x_nominal = source_dataset.x_nominal
            vectorizer = self.__vectorizer
            
            x_nominal = self.__to_dict_vectorizer_format(x_nominal)
            
            destination_x_nominal = vectorizer.transform(x_nominal)
            destination_x_nominal = pd.DataFrame(destination_x_nominal, columns = vectorizer.get_feature_names()) 
            destination_x_nominal.index = source_dataset.x_nominal.index
            
            if(self.__is_vectorized_correct(destination_x_nominal) == False): 
                raise ValueError('One-Hot Encoding Failed in dataset of shape: {0}'.format(x_nominal.shape)) 
            
            destination_attribute_list = self.__adjust_attribute_list(source_dataset.attribute_list, destination_x_nominal.columns)
            
            destination_dataset = DataSet()
            destination_dataset.set_all_by_parts(destination_x_nominal, source_dataset.x_numeric, source_dataset.y, destination_attribute_list)
    
            return destination_dataset

    def __create_column_mapping_table(self, vectorized_columns): 
        """
        Function creates mapping table (column - nominal attribute - value)

        **Args**
            ==================== ========================================================================== ========================
            Parameter            Data Type                                                                  Content
            ==================== ========================================================================== ========================
            vectorized_columns   ``pandas.DataFrame`` (number_of_features\*number_of_values_each_feature,)  ['A1=1', 'A1=2', ...]
            ==================== ========================================================================== ========================
            
        **Returns**
            +--------------+---------------------------------------------------------------------------+---------------------------------------------------+
            |**Parameter** |**Data Type**                                                              |**Content**                                        |
            +--------------+---------------------------------------------------------------------------+-------+------------------+------------------------+
            |mapping_table |``pandas.DataFrame``                                                       | column| nominal_attribute| nominal_attribute_value| 
            |              |(number_of_nominal_features\*number_of_values_each_feature,3)              +-------+------------------+------------------------+
            |              |                                                                           |A1=1   |A1                |1                       |
            +--------------+---------------------------------------------------------------------------+-------+------------------+------------------------+
            
        """
        nominal_attributes = []
        nominal_attribute_values = []
        vectorized_columns_df = pd.DataFrame(vectorized_columns)
        
        for column in vectorized_columns:   
            attribute = re.findall("(.+?)=.+", column)
            attribute_value = re.findall(".+?=(.+)", column)
            
            nominal_attributes = nominal_attributes+[attribute]
            nominal_attribute_values = nominal_attribute_values+[attribute_value]
        
        nominal_attributes = pd.DataFrame(nominal_attributes)
        nominal_attribute_values = pd.DataFrame(nominal_attribute_values)
        
        vectorized_columns_df = pd.concat([vectorized_columns_df, nominal_attributes, nominal_attribute_values], axis=1)
        vectorized_columns_df.columns = ["column", "nominal_attribute", "nominal_attribute_value"]
        
        return vectorized_columns_df 
    
    def __to_dict_vectorizer_format(self, x_nominal): 
        """
        Function shifts ``pandas.DataFrame`` to format that ``sklearn.feature_extraction.DictVectorizer`` accepts
        """
        x_nominal = x_nominal.astype(str)
        x_nominal = x_nominal.to_dict(orient='records')
        return x_nominal
    
    def __is_vectorized_correct(self, vectorized_x):
        """
        Function checks if x is vectorized correctly (Each sample's A1=1~A1=n values' sum = 1)
        """
        if(set(vectorized_x.columns)==set()): 
            return True
        
        mapping_table = self.__create_column_mapping_table(vectorized_x.columns)
        success = True
        
        for nominal_attribute in np.unique(mapping_table.nominal_attribute):  
            columns_of_attribute = mapping_table[mapping_table['nominal_attribute']==nominal_attribute] 
            column_list = columns_of_attribute['column']    
            
            x_columns = vectorized_x.loc[:, column_list]    
            x_sum_of_columns = x_columns.sum(axis=1)   
            
            abnormal_rows = x_sum_of_columns[x_sum_of_columns!=1]   # calculate how many rows' sum != 1 
            abnormal_rows_cnt = abnormal_rows.count()   

            if(abnormal_rows_cnt > 0):    
                success = False
                print("For variable {0}, One-Hot Encoding is limited to {1}, but variable other values".format(nominal_attribute, columns_of_attribute['nominal_attribute_value'].values))
        return success
    
    def __adjust_attribute_list(self, source_attribute_list, destination_columns): 
        """
        """
        mapping_table = self.__create_column_mapping_table(destination_columns)
        mapping_table = pd.merge(mapping_table, source_attribute_list.schema, left_on='nominal_attribute', right_on='Field', how='left')

        drop_list =  np.unique(mapping_table.nominal_attribute)
        
        add_list = mapping_table.loc[:, ['Category','column','Type']]
        add_list = add_list.rename(columns={'column': 'Field'})
        
        destination_attribute_list = copy.deepcopy(source_attribute_list)
        destination_attribute_list.adjust_x_list(add_list=add_list, drop_list=drop_list)
        
        return destination_attribute_list

#%%
from sklearn.preprocessing import StandardScaler


class Standardizer(__DataTransformer): 
    """
    | Class: 
    | - holds methods to fit and transform numeric features with ``sklearn.preprocessing.StandardScaler``
    """
    def __init__(self): 
        pass
    
    def __str__(self): 
        if(self.__scaler is None): 
            return "No numeric feature to standardize"
        else: 
            str_ = "{}".format(self.__scaler)
            return str_

    def fit(self, dataset, fake_y=None): 
        """
        """
        if(dataset.x_numeric.empty): 
            self.__scaler = None
        else:     
            x_numeric = dataset.x_numeric
            self.__scaler = StandardScaler()
            self.__scaler.fit(x_numeric)
    
    def transform(self, source_dataset, fake_y=None): 
        """
        Function transforms target_dataset with fitted result 
        
        **Args**
            =============== ============================================================= ===================================================================
            Parameter       Data Type                                                     Description
            =============== ============================================================= ===================================================================
            source_dataset  ``DataSet`` (see module `data_set <data_set.html>`_)          Data set to be transformed according to fit scale 
            =============== ============================================================= ===================================================================
            
        **Returns**
            =================== ===================================================================== ===================================================================
            Parameter           Data Type                                                             Description
            =================== ===================================================================== ===================================================================
            destination_dataset ``pandas.DataFrame`` (number_of_samples, number_of_numeric_features)  Scaled numeric features 
            =================== ===================================================================== ===================================================================

        """

        if(source_dataset.x_numeric.empty): 
            return source_dataset
        else: 
            x_numeric = source_dataset.x_numeric
            destination_x_numeric = pd.DataFrame(self.__scaler.transform(x_numeric), columns=list(x_numeric)) 
            destination_x_numeric.index = x_numeric.index
            
            destination_dataset = DataSet()
            destination_dataset.set_all_by_parts(source_dataset.x_nominal, destination_x_numeric, source_dataset.y, source_dataset.attribute_list)
            return destination_dataset

#%%
from sources.feature_profiling import XYAnalyzer
import math 

class Binner(): 
    """
    Only bins numeric features 
    """
    def __init__(self): 
        self.__xy_analyzer = None
        self.__all_columns = []
        self.__transform_columns = []
        self.__skip_columns_manual = []
        self.__skip_columns_nominal = []
    
    def __str__(self): 
        return "Bin Columns: {0}\n\nSkip Columns: \n\t- Nominal: {1}\n\t- Manual: {2}\n\n".format(self.transform_columns, self.skip_columns_nominal, self.skip_columns_manual)
    
    @property 
    def xy_analyzer(self): 
        return self.__xy_analyzer
    
    @property 
    def transform_columns(self): 
        return self.__transform_columns

    @property 
    def skip_columns_nominal(self): 
        return self.__skip_columns_nominal

    @property 
    def skip_columns_manual(self): 
        self.__skip_columns_manual = [item for item in self.__all_columns if (item not in self.__transform_columns) and item not in self.skip_columns_nominal]
        return self.__skip_columns_manual
    
    def set_transform_columns(self, columns='all'): 
        if(columns=='all'): 
            columns = self.__all_columns
        self.__transform_columns = [item for item in columns if item not in self.__skip_columns_nominal]
    
    def set_fitted_analyzer(self, xy_analyzer): 
        self.__all_columns = xy_analyzer.columns
        self.__skip_columns_nominal = xy_analyzer.nominal_columns
        
        self.__xy_analyzer = xy_analyzer 
    
    def fit(self, dataset, segment_by='auto', max_bin=10, minimum_size=1/20): 
        self.__all_columns = list(dataset.x.columns)
        self.__skip_columns_nominal = list(dataset.x_nominal.columns)
        self.set_transform_columns()
        
        self.__xy_analyzer = XYAnalyzer(dataset, segment_by, max_bin, minimum_size)

    def transform(self, source_dataset): 
        self.__source_dataset = copy.deepcopy(source_dataset)
        destination_dataset = copy.deepcopy(source_dataset)
        
        if(self.__transform_columns == 'all'): 
            transform_columns = self.__transform_columns
        else:
            transform_columns = self.__transform_columns 
        
        for feature in list(transform_columns):
            binned_feature = "{}_BIN".format(feature)
            
            lift_table = self.__xy_analyzer.lift_tables[feature]
                
            feature_df = source_dataset.x[[feature]]
            binned_df = self.__create_bin_df(feature_df, lift_table)
            
            schema = destination_dataset.attribute_list.schema
            schema = schema[schema["Field"]==feature]
            schema["Field"] = binned_feature
            schema["Type"] = "numeric"
            destination_dataset.adjust_x(add_data = binned_df, drop_columns = set([feature]), add_schema = schema)
        
        destination_dataset.order_x_by_attribute_list()
        
        return destination_dataset
    
    def __create_bin_df(self, source_df, lift_table): 
        feature = lift_table.feature
        bin_table = lift_table.bin_table
        bin_table = bin_table.transpose(copy=True)
        
        destination_column = "{}_BIN".format(feature)
        destination_df = pd.DataFrame()
        if(feature in self.__source_dataset.x_nominal): 
            destination_df[destination_column] = source_df[feature].copy(deep=True)
        else: 
            destination_df[destination_column] = source_df[feature].apply(self.__assign_bin, args=(feature, bin_table, ))

        return destination_df
    
    def __assign_bin(self, x, feature, bin_table): 
        bin_ = None
        if math.isnan(x): 
            bin_ = x
        elif(len(bin_table.index)==1): 
            bin_ = 0
        elif type(x) == str: 
            bin_ = x
        else: 
            for i in bin_table.index: 
                if x > bin_table['MAX'][len(bin_table.index)-2]: 
                    bin_ = len(bin_table.index)-1
                    break
                elif x <= bin_table['MAX'][i]: 
                    bin_ = i 
                    break
        return bin_

#%%
#from sources.feature_profiling import XYAnalyzer

class CorrelationAbsOrderer(): 
    def __init__(self): 
        pass
    
    def fit(self, source_dataset): 
        xy_analyzer = XYAnalyzer(dataset = source_dataset, segment_by = 'range', max_bins = 10)
        self.__correlation_order = list(xy_analyzer.corr_abs_table.index)

    def transform(self, source_dataset): 
        destination_dataset = copy.deepcopy(source_dataset)
        destination_dataset.order_x(self.__correlation_order)
        
        return destination_dataset
