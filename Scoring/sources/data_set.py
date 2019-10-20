# -*- coding: utf-8 -*-
"""
Module holds classes for self-defined storage (DataSet, AttributeList) and classes to parse data (DataParser)
@author: chialing
"""
from sources.file_controller import DataframeFileController
import pandas as pd
import copy

class DataSet(): 
    """
    Class holds components of modeling/validation datasets: x, y, and attribute_list. 
    """
    __x = None 
    __y = None 
    __attribute_list = None 
    
    def __init__(self): 
        pass
    
    def __str__(self):
        return 'x: {0}\ny: {1}\nattribute_list: \n{2}'.format(self.__x.shape, self.__y.shape, self.__attribute_list)
    
    @property 
    def x(self):
        """
        Data Type: ``pandas.DataFrame`` (number_of_samples, number_of_features)
        """
        return self.__x
    
    @property 
    def y(self): 
        """
        Data Type: ``pandas.DataFrame`` (number_of_samples, )
        """
        return self.__y 
    
    @property 
    def attribute_list(self):
        """
        Data Type: ``AttributeList`` (see below)
        """
        return self.__attribute_list

    @property 
    def x_nominal(self):
        """
        Data Type: ``pandas.DataFrame`` (number_of_samples, number_of_nominal_features)
        """
        data_parser = DataParser(self.__x, self.__attribute_list)
        x_nominal = data_parser.parse_x_nominal()
        return x_nominal

    @property 
    def x_numeric(self): 
        """
        Data Type: ``pandas.DataFrame`` (number_of_samples, number_of_numeric_features)
        """
        data_parser = DataParser(self.__x, self.__attribute_list)
        x_numeric = data_parser.parse_x_numeric()
        return x_numeric
    
    @property 
    def xy(self): 
        xy = pd.concat([self.x, self.y], axis=1)
        return xy
    
    def order_x(self, order_list): 
        self.__x = self.__x[order_list]
        self.__attribute_list.order_x_list(order_list)
        
        if(order_list != self.__x.columns.tolist()): 
            raise ValueError('order_x failed')
    
    def order_x_by_attribute_list(self): 
        order_list = self.__attribute_list.x_list.tolist()
        self.order_x(order_list)


    def set_all(self, x, y, attribute_list): 
        """
        
        Function sets x, y and attribute_list according to input 
        
        **Args** 
            ================ ============================= ======================== 
            Parameter        Data Type                     Description
            ================ ============================= ======================== 
            x                ``pandas.DataFrame``          Features for Modeling
            y                ``pandas.DataFrame``          Y for Modeling
            attribute_list   ``AttributeList`` (see below)    
            ================ ============================= ======================== 
            
        **Returns** 
            None 
            
        """
        self.__x, self.__y, self.__attribute_list = copy.deepcopy(x), copy.deepcopy(y), copy.deepcopy(attribute_list)
        self.order_x_by_attribute_list()

    def add_x_columns(self, add_data = None, add_schema = None): 
        self.__x = pd.concat([self.__x, add_data], axis=1)
        if(add_schema is None): 
            self.__attribute_list.adjust_x_list_from_schema(add_columns = list(add_data.columns), drop_columns=None)
        else: 
            self.__attribute_list.adjust_x_list(add_list = add_schema, drop_list = None)
        self.order_x_by_attribute_list()
        
    def drop_x_columns(self, drop_columns = set()): 
        keep_columns = list(set(self.__x.columns)-set(drop_columns))
        self.__x = self.__x.loc[:, keep_columns]
        self.__attribute_list.adjust_x_list_from_schema(add_columns = None, drop_columns = drop_columns)
        self.order_x_by_attribute_list()
    
    def adjust_x(self, add_data = None, drop_columns = set(), add_schema = None): 
        """
        """
        if(not add_data is None): 
            self.add_x_columns(add_data, add_schema)
        self.drop_x_columns(drop_columns)
        self.order_x_by_attribute_list()
        
    
    def set_all_by_parts(self, x_nominal, x_numeric, y, attribute_list): 
        """
        """
        self.__x = pd.concat([x_nominal, x_numeric], axis=1)
        self.__y, self.__attribute_list = copy.deepcopy(y), copy.deepcopy(attribute_list)
        self.order_x_by_attribute_list()
    
    def load_all_from_csv(self, dataset_filepath, schema_filepath):
        """

        Function loads x, y, attribute_list from csv file 
        
        **Args** 
            ================ =========== ============================== 
            Parameter        Data Type   Description
            ================ =========== ============================== 
            data_filepath    ``str``     File path for data (.csv)
            schema_filepath  ``str``     File path for schema (.csv)
            ================ =========== ============================== 
            
        **Returns** 
            None 
        """

        self.__attribute_list = AttributeList(filepath=schema_filepath)

        df_controller = DataframeFileController(filepath=dataset_filepath)
        df = df_controller.load()
        
        data_parser = DataParser(df, self.__attribute_list)
        self.__x, self.__y = data_parser.parse_x(), data_parser.parse_y()
        self.order_x_by_attribute_list()
        
    def load_dataset_from_csv(self, dataset_filepath, attribute_list): 
        """
        
        Function loads x, y from csv file 
        
        **Args** 
            ============== =================== ==================================== 
            Parameter      Data Type           Description
            ============== =================== ==================================== 
            data_filepath  ``str``             File path for data (.csv)
            attribute_list ``AttributeList``   AttributeList in program
            ============== =================== ==================================== 
            
        **Returns** 
            None 
        """
        self.__attribute_list = attribute_list

        df_controller = DataframeFileController(filepath=dataset_filepath)
        df = df_controller.load()
        
        data_parser = DataParser(df, self.__attribute_list)
        self.__x, self.__y = data_parser.parse_x(), data_parser.parse_y()
        self.order_x_by_attribute_list()

#%%
class AttributeList(): 
    """
    Class holds components for a complete attribute list
    """
    __x_list = None
    __y_list = None
    __key_list = None 
    
    def __init__(self, filepath = 'Data\\Data_Schema'): 
        schema = DataframeFileController(filepath).load()
        schema = schema.loc[:,['Category','Field','Type']]
        
        self.__original_schema = schema
        self.__x_list = schema[schema['Category'].isin(['X'])]
        self.__y_list = schema[schema['Category'].isin(['Y'])]
        self.__key_list = schema[schema['Category'].isin(['Key'])]
    
    def __str__(self):
        return '\tx_list: {0}\n\ty_list: {1}\n\tkey_list: {2}'.format(self.x_list.tolist(), self.y_list.tolist(), self.key_list.tolist())

    
    @property 
    def x_list(self): 
        """
        Data type: ``pandas.DataFrame`` (number_of_features, )
        """
        return self.__x_list['Field']
    
    @property 
    def y_list(self): 
        """
        Data type: ``pandas.DataFrame`` (1, )
        """
        return self.__y_list['Field']

    @property 
    def key_list(self): 
        """
        Data type: ``pandas.DataFrame`` (1, )
        """
        return self.__key_list['Field']    

    @property 
    def x_list_nominal(self): 
        """
        Data type: ``pandas.DataFrame`` (number_of_nominal_features, )
        """
        x_list = self.__x_list 
        x_list_nominal = x_list[x_list['Type']=='nominal']
        return x_list_nominal['Field']
    
    @property 
    def x_list_numeric(self): 
        """
        Data type: ``pandas.DataFrame`` (number_of_numeric_features, )
        """
        x_list = self.__x_list 
        x_list_numeric = x_list[x_list['Type']=='numeric']
        return x_list_numeric['Field']

    @property 
    def schema(self): 
        """
        | Data type: ``pandas.DataFrame`` (number_of_attributes, 3)
        | Content: 
        
            ============= =============== =============
            Category      Field           Type
            ============= =============== =============
            X             feature1        nominal
            X             feature2        numeric
            Y             y               numeric
            Key           key             numeric
            ============= =============== =============
            
        """
        schema = pd.concat([self.__key_list, self.__y_list, self.__x_list], axis=0)
        schema.reset_index(inplace=True, drop=True)
        return schema
    
    @property
    def original_schema(self): 
        """
        """
        return self.__original_schema

    def order_x_list(self, order_list): 
        order = pd.DataFrame(data=order_list, columns=['Field'])
        order['order'] = order.index
        
        x_list_new = pd.merge(left=self.__x_list, right=order, how='left', on='Field', copy=True)
        x_list_new.index = x_list_new['order']
        x_list_new = x_list_new.sort_index(ascending=True)
        self.__x_list = x_list_new[['Category', 'Field', 'Type']]

    def keep_x_list(self, keep_list): 
        x_list = self.__x_list
        self.__x_list = x_list[x_list['Field'].isin(keep_list)]
    
    def adjust_x_list(self, add_list, drop_list): 
        x_list = self.__x_list
        
        if(not drop_list is None): 
            x_list = x_list[~x_list['Field'].isin(drop_list)]
        
        if(not add_list is None): 
            x_list = pd.concat([x_list, add_list], axis=0, sort=True)
        
        x_list.reset_index()
        
        self.__x_list = x_list 

    def adjust_x_list_from_schema(self, add_columns = None, drop_columns = None): 
        if(not add_columns is None): 
            add_list = self.__original_schema[self.__original_schema['Field'].isin(add_columns)]
        else: 
            add_list = None
        self.adjust_x_list(add_list, drop_columns)
        
    def to_csv(self, filepath='Models\\x_list_final.csv'): 
        schema = self.schema
        DataframeFileController(filepath).save(schema)

#%%
class DataParser(): 
    def __init__(self, dataset, attribute_list): 
        self.__dataset = dataset
        self.__x_list, self.__y_list, self.__key_list = attribute_list.x_list, attribute_list.y_list, attribute_list.key_list
        self.__x_list_nominal, self.__x_list_numeric = attribute_list.x_list_nominal, attribute_list.x_list_numeric
        
    def keep_columns(self, dataset, keep_list): 
        return dataset.loc[:,keep_list]
    
    def __concat_columns(self, dataset_1, dataset_2): 
        return pd.concat([dataset_1, dataset_2], axis=1)

    def __parse_y_or_key(self, attr_list): 
        attr_name = attr_list.max(axis=0)
        df_attr = self.keep_columns(self.__dataset, attr_name)
        return df_attr 
    
    def parse_x(self): 
        x = self.keep_columns(self.__dataset, self.__x_list)
        return x 
    
    def parse_y(self): 
        return self.__parse_y_or_key(attr_list=self.__y_list)
        
    def parse_key(self): 
        return self.__parse_y_or_key(attr_list=self.__key_list)
    
    def parse_x_nominal(self): 
        x_nominal = self.keep_columns(self.__dataset, self.__x_list_nominal)
        return x_nominal
    
    def parse_x_numeric(self): 
        x_numeric = self.keep_columns(self.__dataset, self.__x_list_numeric)
        return x_numeric
    
    def parse_xy(self, x_list, y_list): 
        x, y = self.parse_x(), self.parse_y() 
        xy = self.__concat_columns(x, y)
        return xy
        
