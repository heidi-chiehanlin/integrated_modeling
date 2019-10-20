# -*- coding: utf-8 -*-
"""
Feature Selection Methods 

- Filter Method: Constant, Unsupported Type, Missing Value, Correlation
- Wrapper Method: <TBD>
- Embedded Method: implemented in Modeling Step 
"""

from sources.feature_profiling import XProfiler
import copy
from sources.data_transformation import CorrelationAbsOrderer

class Selector(): 
    _drop_list = None 
    
    def __init(self): 
        pass

    def __str__(self): 
        return '{0}: {1} attribute(s) removed -- {2}'.format(self._rule, len(self._drop_list), self._drop_list) 
    
    def set_fitted_drop_list(self, drop_list): 
        self.__drop_list = drop_list
    
    def fit_transform(self, source_dataset, fake_y=None): 
        """
        """
        self.fit(source_dataset)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset
        
    def transform(self, source_dataset, fake_y=None): 
        """
        """
        try: 
            destination_dataset = copy.deepcopy(source_dataset)
            destination_dataset.adjust_x(drop_columns=self._drop_list)
            destination_dataset.order_x_by_attribute_list()
            return destination_dataset
        except: 
            print("{0}: No features removed")
            return source_dataset
    

class __Filterer(Selector): 
    def create_report(self, source_dataset): 
        """
        """
        self._x_profiler = XProfiler()
        profile_report = self._x_profiler.create_report(source_dataset) 
        return profile_report

class ConstantFilterer(__Filterer): 
    """
    Filter Method: remove features with a constant value
    """
    
    def fit(self, source_dataset, fake_y=None): 
        self._rule = 'CONSTANT'        
        self.profile_report = self.create_report(source_dataset)

        description_set = self.profile_report.description_set['variables']
        self._drop_list = description_set[description_set.type == 'CONST'].index.tolist()

class UnsupportedTypeFilterer(__Filterer): 
    """
    Filter Method: remove features in unsupported datatype
    """

    def fit(self, source_dataset, fake_y=None): 
        self._rule = 'UNSUPPORTED TYPE'
        self.profile_report = self.create_report(source_dataset)
        
        description_set = self.profile_report.description_set['variables']
        self._drop_list = description_set[description_set.type == 'UNSUPPORTED'].index.tolist()


class MissingValueFilterer(__Filterer):
    """
    Filter Method: remove values with high percentage of missing values
    """

    def fit_transform(self, source_dataset, fake_y=None, missing_value_threshold=0.5): 
        self.fit(source_dataset, missing_value_threshold)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset
        
    def fit(self, source_dataset, fake_y=None, missing_value_threshold=0.5): 
        self._rule = 'MISSING VALUE >= {}'.format(missing_value_threshold)
        self.profile_report = self.create_report(source_dataset)
        
        description_set = self.profile_report.description_set['variables']
        index  = (description_set['p_missing'] >= missing_value_threshold)
        self._drop_list = list(description_set.loc[index, 'p_missing'].index)


class CorrelationFilterer(__Filterer): 
    """
    Filter Method: remove correlated features
    """

    def fit_transform(self, source_dataset, fake_y=None, correlation_threshold = 0.9): 
        self.fit(source_dataset, correlation_threshold)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset

    def fit(self, source_dataset, fake_y=None, correlation_threshold = 0.9): 
        self._rule = 'CORRELATION >= {}'.format(correlation_threshold)
        
        orderer = CorrelationAbsOrderer()
        orderer.fit(source_dataset)
        ordered_dataset = orderer.transform(source_dataset)
        
        self.profile_report = self.create_report(ordered_dataset)
        
        self._drop_list = self.profile_report.get_rejected_variables(threshold = correlation_threshold)


#%%
from sources.data_transformation import Standardizer

class __AnySelector(Selector): 
    """
    """
    def standardize_dataset(self, source_dataset): 
        standardizer = Standardizer()
        destination_dataset = standardizer.fit_transform(source_dataset)
        return destination_dataset
    
    def create_drop_list(self, selection_table): 
        column = selection_table.columns[0]
        self._rule = "Feature Selection by {}".format(column)
        
        drop_list = selection_table[selection_table[column]==False].index
        drop_list = list(drop_list)
        
        return drop_list
    
    def fit_transform(self, source_dataset, fake_y=None, maximum_vote = 12): 
        self.fit(source_dataset, fake_y=None, maximum_vote = maximum_vote)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset
    
    @property
    def measurement_table(self): 
        return self._measurement_table

    @property
    def selection_table(self): 
        return self._selection_table

    @property
    def drop_list(self): 
        return self._drop_list
    
class __ValueSelector(__AnySelector): 
    def create_selection_table(self, measurement_table, maximum_vote): 
        """
        """
        column_name = measurement_table.columns
        nlargest_index = measurement_table.nlargest(n=maximum_vote, columns=column_name, keep="all").index
        selection_table = measurement_table.index.isin(nlargest_index)
        selection_table = pd.DataFrame(data=selection_table, index=measurement_table.index, columns = column_name)

        return selection_table

    def fit(self, source_dataset, fake_y=None, maximum_vote = 12): 
        """
        """
        source_dataset = self.standardize_dataset(source_dataset)
        self._measurement_table = self.create_measurement_table(source_dataset)
        self._selection_table = self.create_selection_table(self._measurement_table, maximum_vote)
        self._drop_list = self.create_drop_list(self._selection_table)

class __BooleanSelector(__AnySelector): 
    def create_selection_table(self, measurement_table): 
        return measurement_table
    
    def fit(self, source_dataset, fake_y=None, maximum_vote = 12): 
        """
        """
        source_dataset = self.standardize_dataset(source_dataset)
        self._measurement_table = self.create_measurement_table(source_dataset, maximum_vote)
        self._selection_table = self.create_selection_table(self._measurement_table)
        self._drop_list = self.create_drop_list(self._selection_table)
        
#%%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

class VoteFeatureSelector(__ValueSelector): 
    """
    Wrapper Method: remove according to voted result of various algorithms
    """
    def fit_transform(self, source_dataset, fake_y=None, maximum_vote=12): 
        self.fit(source_dataset, maximum_vote=maximum_vote)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset
    
    def fit(self, source_dataset, fake_y=None, maximum_vote=12): 
        source_dataset = self.standardize_dataset(source_dataset)
        self._selector_list = self.create_selectors(source_dataset, maximum_vote)
        self._measurement_table = self.create_measurement_table(self._selector_list)
        self._voting_table = self.create_voting_table(self._selector_list)
        self._selection_table = self.create_selection_table(self._voting_table.loc[:,["Final_Score"]], maximum_vote)
        self._drop_list = self.create_drop_list(self._selection_table)
        self._rule = "Feature Selection by Voting"
    
    def create_selectors(self, source_dataset, maximum_vote): 
        iv_selector = IVSelector()
        iv_selector.fit(source_dataset, maximum_vote=maximum_vote)
        
        fi_random_forest_classifier_selector = FeatureImportanceSelector()
        fi_random_forest_classifier_selector.fit(source_dataset, model=RandomForestClassifier())
    
        fi_extra_tree_classifier_selector = FeatureImportanceSelector()
        fi_extra_tree_classifier_selector.fit(source_dataset=source_dataset, model=ExtraTreesClassifier())

        rfe_logistic_regression_selector = RFESelector()
        rfe_logistic_regression_selector.fit(source_dataset=source_dataset, fake_y=None, model=LogisticRegression(), maximum_vote=12)
        
        rfe_random_forest_classifier_selector = RFESelector()
        rfe_random_forest_classifier_selector.fit(source_dataset=source_dataset, model=RandomForestClassifier(), maximum_vote=12)
        
        l1_logistic_regression_selector = L1Selector()
        l1_logistic_regression_selector.fit(source_dataset=source_dataset, model=LogisticRegression(C=10, penalty='l1', dual=False, class_weight='balanced'), maximum_vote=12)

        l1_linear_svc_selector = L1Selector()
        l1_linear_svc_selector.fit(source_dataset=source_dataset, model=LinearSVC(C=10, penalty='l1', dual=False, class_weight='balanced'), maximum_vote=12)
            
        chi2_selector = Chi2Selector()
        chi2_selector.fit(source_dataset)
        
        selector_list = [
                iv_selector,
                fi_random_forest_classifier_selector,
                fi_extra_tree_classifier_selector,
                rfe_logistic_regression_selector,
                rfe_random_forest_classifier_selector,
                l1_logistic_regression_selector,
                l1_linear_svc_selector,
                chi2_selector
                ]
        
        return selector_list
    
    def create_measurement_table(self, selector_list): 
        measurement_list = [x.measurement_table for x in selector_list]
        measurement_table = pd.concat(measurement_list, axis=1, sort=False)
        return measurement_table
    
    def create_voting_table(self, selector_list): 
        voting_list = [x.selection_table.astype(int) for x in selector_list]
        voting_table = pd.concat(voting_list, axis=1, sort=False)
        
        voting_table["Final_Score"] = voting_table.sum(axis=1)
        voting_table = voting_table.sort_values(by="Final_Score", ascending=False)
        
        return voting_table
    
    @property
    def selector_list(self):
         return self._selector_list
     
    @property
    def voting_table(self): 
        return self._voting_table

#%%
from sources.feature_profiling import XYAnalyzer

class IVSelector(__ValueSelector): 
    
    def create_measurement_table(self, dataset, segment_by='auto', n_bins=10): 
        xy_analyzer = XYAnalyzer(dataset, segment_by, n_bins)
        measurement_table = xy_analyzer.iv_table
        measurement_table = measurement_table.sort_values(['IV'], ascending=0)
        return measurement_table

#%%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class Chi2Selector(__ValueSelector): 
    """
    ChiSquare
    """
    def create_measurement_table(self, dataset, maximum_vote=None): 
        """
        """
        x_shifted = self.__shift_to_zero(dataset.x)
        
        selector = SelectKBest(score_func=chi2)
        fit = selector.fit(x_shifted, dataset.y)
    
        selector_name = 'ChiSquare'
        measurement_table = pd.DataFrame(fit.scores_, columns=[selector_name], index=x_shifted.columns)
        measurement_table = measurement_table.sort_values(selector_name, ascending=0)    
        return measurement_table
        
    def __shift_to_zero(self, x): 
        x_shifted = x.copy(deep=True)
        for column in x.columns: 
            min_value = x[column].min()
            x_shifted[column] = x[column]-min_value
        return x_shifted

#%%

class FeatureImportanceSelector(__ValueSelector): 
    """
    """
    def fit_transform(self, source_dataset, fake_y=None, model=None, maximum_vote = 12): 
        self.fit(source_dataset, fake_y=None, model=model, maximum_vote = maximum_vote)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset
    
    def fit(self, source_dataset, fake_y=None, model=None, maximum_vote = 12): 
        """
        """
        source_dataset = self.standardize_dataset(source_dataset)
        self._measurement_table = self.create_measurement_table(source_dataset, model)
        self._selection_table = self.create_selection_table(self._measurement_table, maximum_vote)
        self._drop_list = self.create_drop_list(self._selection_table)
    
    def create_measurement_table(self, dataset, model): 
        """
        """
        selector_name = "FI_{}".format(type(model).__name__)
        model.fit(dataset.x, dataset.y)
        measurement_table = pd.DataFrame(model.feature_importances_, columns = [selector_name], index=dataset.x.columns)
        measurement_table = measurement_table.sort_values([selector_name], ascending=0)
        return measurement_table

#%%
from sklearn.feature_selection import RFE

class RFESelector(__BooleanSelector): 
    """
    """
    def fit_transform(self, source_dataset, fake_y=None, model=None, maximum_vote = 12): 
        self.fit(source_dataset, fake_y=None, model=model, maximum_vote = maximum_vote)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset

    def fit(self, source_dataset, fake_y=None, model=None, maximum_vote = 12): 
        source_dataset = self.standardize_dataset(source_dataset)
        self._measurement_table = self.create_measurement_table(source_dataset, model, n_features=maximum_vote)
        self._selection_table = self.create_selection_table(self._measurement_table)
        self._drop_list = self.create_drop_list(self._selection_table)
    
    def create_measurement_table(self, dataset, model, n_features=None): 
        """
        """
        rfe = RFE(model, n_features)
        rfe.fit(dataset.x, dataset.y)
        
        selector_name = 'RFE_{}'.format(type(model).__name__)
        feature_selection_result = pd.DataFrame(rfe.support_, columns=[selector_name], index=dataset.x.columns)
        
        return feature_selection_result

#%%

from sklearn.feature_selection import SelectFromModel

class L1Selector(__BooleanSelector): 
    """
    """
    def fit_transform(self, source_dataset, fake_y=None, model=None, maximum_vote = 12): 
        self.fit(source_dataset, fake_y=None, model=model, maximum_vote = maximum_vote)
        destination_dataset = self.transform(source_dataset)
        return destination_dataset
    
    def fit(self, source_dataset, fake_y=None, model=None, maximum_vote = 12): 
        source_dataset = self.standardize_dataset(source_dataset)
        self._measurement_table = self.create_measurement_table(source_dataset, model, maximum_vote=maximum_vote)
        self._selection_table = self.create_selection_table(self._measurement_table)
        self._drop_list = self.create_drop_list(self._selection_table)
    
    def create_measurement_table(self, dataset, model, maximum_vote=None): 
        """
        """
        model.fit(dataset.x, dataset.y)
        selector = SelectFromModel(model, prefit=True, max_features=maximum_vote)
    
        selector_name = 'L1_{}'.format(type(model).__name__)
        measurement_table = pd.DataFrame(selector.get_support(), columns=[selector_name], index=dataset.x.columns)
        return measurement_table
        
