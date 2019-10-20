# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:10:55 2019

@author: chialing
"""

import pandas as pd
import pandas_profiling as pp

class XProfiler(): 
    """
    Class holds methods for profiling attributes with open-source tool ``pandas_profiling``
    """
    def __init___(self): 
        pass
    
    def create_report(self, dataset): 
        """
        Function creates profile report with open-source tool ``pandas_profiling``
        
        **Args** 
            =============== ================================================================= ==================================== 
            Parameter       Data Type                                                         Description
            =============== ================================================================= ==================================== 
            df_x            ``pandas.DataFrame`` (number_of_samples, number_of_features)      Features for modeling
            df_y            ``pandas.DataFrame`` (number_of_samples, )                        Y for modeling
            output_to_file  ``Boolean``                                                       Whether or not to output 
            filepath        ``Str``                                                           Path to output file
            =============== ================================================================= ==================================== 
            
        **Returns** 
            ================ =================================== ======================== 
            Parameter        Data Type                           Description
            ================ =================================== ======================== 
            profile_report   ``pandas_profiling.ProfileReport``  Reporting Result 
            ================ =================================== ======================== 
            
        """
        df_xy = pd.concat([dataset.x, dataset.y], axis=1)
        self.__profile_report = pp.ProfileReport(df_xy)
        
        return self.__profile_report 

    def to_html(self, filepath = ''): 
        """
        Function creates profile report with open-source tool ``pandas_profiling``
        
        **Args** 
            =============== ================================================================= ==================================== 
            Parameter       Data Type                                                         Description
            =============== ================================================================= ==================================== 
            filepath        ``Str``                                                           Path to output file
            =============== ================================================================= ==================================== 
            
        **Returns** 
            None
        """
        self.__profile_report.to_file(filepath)
        

#%%
from sources.evaluator import Evaluator
import matplotlib.pyplot as plt
import math
import PyPDF2
import os 
from scipy import stats

class LiftTable(): 
    feature = None 
    iv = None 
    ks = None 
    bin_table = None
    bounce = None
    
    def __init__(self, feature, original_corr, original_p_value, bin_corr, bin_p_value, iv, ks, bin_table, bounce): 
        self.feature = feature 
        self.original_corr = original_corr
        self.original_p_value = original_p_value
        self.bin_iv = iv 
        self.bin_ks = ks 
        self.bin_corr = bin_corr
        self.bin_p_vallue = bin_p_value
        self.bin_table = bin_table
        self.bin_bounce = bounce        
        
    def __str__(self): 
        return "{0}{1}".format(self.title, self.bin_table.to_string(header=True))
    
    @property 
    def title(self): 
        if(self.bin_corr == "N/A"): 
            title = "{0} (r= {4:+.2f})\nIV={1:.2f}; KS={2:.2f}; r={5}; bounce={3}\n".format(self.feature, self.bin_iv, self.bin_ks, self.bin_bounce, self.original_corr, self.bin_corr)
        else: 
            title = "{0} (r= {4:+.2f})\nIV={1:.2f}; KS={2:.2f}; r= {5:+.2f}; bounce={3}\n".format(self.feature, self.bin_iv, self.bin_ks, self.bin_bounce, self.original_corr, self.bin_corr)
        return title
    

class XYAnalyzer(): 
    _fit_dataset = None 
    
    def __init__(self, dataset=None, segment_by='auto', max_bins=10, minimum_size=1/20): 
        self._fit_dataset = dataset
        self._segment_by = segment_by
        self._max_bins = max_bins
        
        self._lift_tables = dict()
        self._performance_table = None
        self._temp_filepath = None 
        
#        feature = 'HOL_P09_1'
        for feature in dataset.x.columns: 
            feature_df = dataset.x.loc[:,[feature]]
            segment_df = self.__create_segments(feature_df, dataset.y, segment_by, max_bins, minimum_size)
            lift_table = self.create_lift_table(feature_df, segment_df, dataset.y)
            self._lift_tables[feature] = lift_table

    def __str__(self): 
        str_ = ""
        for lift_table in self._lift_tables: 
            str_ = str_ + "{}\n".format(lift_table)
        return str_

    def plot(self, temp_filepath = "Reports\\temp\\"): 
        self._temp_filepath = temp_filepath
        for feature in self._lift_tables: 
            lift_table = self._lift_tables[feature]
            figure = self.plot_feature(lift_table)
            self.to_pdf_feature_chart(figure, "{0}\\{1}.pdf".format(temp_filepath, feature)) 
            print(lift_table)
    
    @property 
    def columns(self): 
        return list(self._fit_dataset.x.columns)

    @property 
    def nominal_columns(self): 
        return list(self._fit_dataset.x_nominal.columns)

    @property 
    def numeric_columns(self): 
        return list(self._fit_dataset.x_numeric.columns)
    
    @property 
    def corr_table(self): 
        return self.get_performance_table("original_correlation")
    
    @property 
    def corr_abs_table(self): 
        corr_table = self.get_performance_table("original_correlation")
        corr_table["original_correlation_abs"] = corr_table["original_correlation"].apply(abs)
        corr_table = corr_table.sort_values(by="original_correlation_abs", axis=0, ascending=False)
        return corr_table

    @property 
    def iv_table(self): 
        return self.get_performance_table("IV")

    @property 
    def ks_table(self): 
        return self.get_performance_table("KS")
    
    @property 
    def lift_tables(self): 
        return self._lift_tables

    def get_performance_table(self, performance_index): 
        self._performance_table = self.__create_performance_table()
        destination_table = self._performance_table.loc[:,[performance_index]]
        destination_table = destination_table.sort_values(by=[performance_index], axis=0, ascending=False)
        return destination_table
    
    def to_csv(self, filepath = "Reports\\XYAnalysisReport.csv"): 
        try: 
            os.remove(filepath)
        except: 
            pass
        
        with open(filepath, 'w') as file: 
            for feature in self._lift_tables: 
                lift_table = self._lift_tables[feature]
                
                file.write(lift_table.title)
                lift_table.bin_table.to_csv(file, index=True, quoting=2, sep=',', line_terminator="\r")
                file.write("\n\n")
       
    def plot_to_pdf(self, temp_filepath = "Reports\\temp\\", filepath = "Reports\\XYAnalysisReport.pdf"): 
        self.plot(temp_filepath)
        
        merger = PyPDF2.PdfFileMerger()
        for feature in self._fit_dataset.attribute_list.x_list: 
            filepath_temp_file = "{0}\\{1}.pdf".format(self._temp_filepath, feature)
            merger.append(filepath_temp_file)
        merger.write(filepath)
        merger.close()
    

    def __create_segments(self, feature_df, y, segment_by='range', max_bins=20, minimum_size=1/20): 
        feature = feature_df.columns[0]
        feature_df_not_nulls, y_not_nulls, feature_df_nulls, y_nulls = self.__split_nulls(feature_df, y)
        
        segment_df_not_nulls = None 
        if(feature_df_not_nulls.empty or feature in self._fit_dataset.x_nominal.columns): 
            segment_df_not_nulls = feature_df_not_nulls.copy(deep=True)
            segment_df_not_nulls.columns = ['SEG']
        
        elif(segment_by=='auto'): 
            for num in range(max_bins, 0, -1): 
                segment_df_not_nulls = self._segment_by_decision_tree(feature_df_not_nulls, y_not_nulls, num, minimum_size)
                
                evaluator = Evaluator()
                temp_bin_table, iv, ks, bounce = evaluator.bivariate_analysis_binning(segment_df_not_nulls['SEG'], y)
                
                if(bounce==0): 
                    break
        
        elif(segment_by=='tree'):
            segment_df_not_nulls = self._segment_by_decision_tree(feature_df_not_nulls, y_not_nulls, max_bins, minimum_size)
            
        elif(segment_by=='range'): 
            segment_df_not_nulls = self._segment_by_range(data=feature_df_not_nulls, max_bins=max_bins)
            
        else: 
            raise(ValueError("Invalid input for segment_by: {}".format(segment_by)))
        
        segment_df_nulls = feature_df_nulls.copy(deep=True)
        segment_df_nulls.columns=['SEG']
        
        segment_df = pd.concat([segment_df_not_nulls, segment_df_nulls], axis=0)
        segment_df = segment_df.sort_index()
        
        return segment_df


    def __split_nulls(self, feature_df, y):
        feature = feature_df.columns[0]
        feature_df_not_nulls = feature_df[~feature_df[feature].isnull()]
        y_not_nulls = y.loc[feature_df_not_nulls.index]
        
        feature_df_nulls = feature_df[feature_df[feature].isnull()]
        y_nulls = y.loc[feature_df_nulls.index]
        
        return feature_df_not_nulls, y_not_nulls, feature_df_nulls, y_nulls

    def create_lift_table(self, feature_df, segment_df, y): 
        feature = feature_df.columns[0]
        segment_df_without_na = pd.DataFrame()
        segment_df_without_na['SEG'] = segment_df['SEG'].fillna("NaN")

        range_table = self.__create_range_table(feature_df, segment_df)
        
        evaluator = Evaluator()
        temp_bin_table, iv, ks, bounce = evaluator.bivariate_analysis_binning(segment_df_without_na['SEG'], y)
        
        temp_bin_table["COUNT%"] = (temp_bin_table["total"]/temp_bin_table["total"].sum())*100
        temp_bin_table["Y%"] = temp_bin_table["event_rate"]*100
        temp_bin_table.index = temp_bin_table["bin"]
        
        bin_table = pd.concat([temp_bin_table, range_table], axis=1)
        bin_table = bin_table.sort_values(by=["min"])

        nan_row = bin_table[temp_bin_table["bin"]=="NaN"]
        bin_table = bin_table.drop(nan_row.index, axis=0)
        bin_table = pd.concat([bin_table, nan_row], axis=0, ignore_index=True)
        temp_bin_table.reindex()
        
        bin_table = bin_table.fillna("NaN")
        bin_table = bin_table.rename(columns={"event": "EVENT", "total": "COUNT", "min": "MIN", "max": "MAX"})
        bin_table = bin_table.loc[:,["MIN", "MAX", "COUNT", "EVENT", "Y%", "COUNT%"]]

        bin_table = bin_table.transpose(copy=True)
        
        
        feature_df_not_nulls, y_not_nulls, feature_df_nulls, y_nulls = self.__split_nulls(feature_df, y)
        original_corr, original_p_value = stats.spearmanr(feature_df_not_nulls, y_not_nulls)
        
        feature_df_not_nulls, y_not_nulls, feature_df_nulls, y_nulls = self.__split_nulls(segment_df, y)
        bin_corr, bin_p_value = stats.spearmanr(feature_df_not_nulls, y_not_nulls)
        
        if(self._fit_dataset is None): 
            lift_table = LiftTable(feature, original_corr, original_p_value, bin_corr, bin_p_value, iv, ks, bin_table, bounce)
        elif(feature in self._fit_dataset.x_nominal.columns): 
            lift_table = LiftTable(feature, original_corr, original_p_value, "N/A", "N/A", iv, ks, bin_table, "N/A")
        elif(feature in self._fit_dataset.x_numeric.columns): 
            lift_table = LiftTable(feature, original_corr, original_p_value, bin_corr, bin_p_value, iv, ks, bin_table, bounce)
        else: 
            print(feature)
        
        return lift_table

    def _segment_by_decision_tree(self, feature_df_not_nulls, y_not_nulls, max_bins, minimum_size): 
        from sklearn import tree
        feature = feature_df_not_nulls.columns[0]

        param_min_samples_leaf = math.ceil(feature_df_not_nulls[feature].count()*minimum_size)
#        print(minimum_size)        
#        print(param_min_samples_leaf)
        
        clf = tree.DecisionTreeClassifier(
                                           max_leaf_nodes = max_bins
                                          , min_samples_leaf = param_min_samples_leaf
                                          , class_weight='balanced'
                                          , random_state=0
                                          )
        clf.fit(feature_df_not_nulls, y_not_nulls)
        
        leave_id = clf.apply(feature_df_not_nulls)
        seg_not_nulls = pd.DataFrame(leave_id, columns=['leaf'])
        seg_not_nulls.index = feature_df_not_nulls.index
        
        seg_not_nulls = pd.concat([seg_not_nulls, feature_df_not_nulls], axis=1)

        groupby_proba = seg_not_nulls.groupby(['leaf'])
        seg = groupby_proba.min()
        seg = seg.rename(columns={feature: 'SEG'})
        
        seg_not_nulls = pd.merge(seg_not_nulls, seg, left_on='leaf', right_index=True, how='left', validate='many_to_one')
        
        return seg_not_nulls[['SEG']]


    def _segment_by_range(self, data, max_bins): 
        segment_df = data.copy(deep=True)
        feature = segment_df.columns[0]
        
        max_ = segment_df[feature].max()
        min_ = segment_df[feature].min()
        range_ = max_ - min_ 
        
        if(range_==0): 
            segment_df['SEG']=0
        else: 
            segment_range = range_/max_bins
    
            segment_df['calculation'] = (segment_df[feature]-min_)/segment_range
            segment_df['SEG'] = [math.floor(x) if (math.floor(x) in range(max_bins)) else max_bins-1 for x in segment_df['calculation']]
        
        return segment_df.loc[:,['SEG']]


    def __create_performance_table(self): 
        performance_table = pd.DataFrame(columns=['IV','KS', 'original_correlation'])
        for feature in self._lift_tables: 
            lift_table = self._lift_tables[feature]
            feature_performance = pd.DataFrame({'IV':lift_table.bin_iv, 'KS':lift_table.bin_ks, 'original_correlation':lift_table.original_corr}, index=[lift_table.feature])
            performance_table = performance_table.append(feature_performance, ignore_index=False)
        
        return performance_table
    
    def __create_range_table(self, feature_df, segment_df): 
        concat_table = pd.concat([feature_df, segment_df], axis=1)
        
        feature = feature_df.columns[0]
        groupby_segment = concat_table.groupby(['SEG'])
        
        range_table = pd.DataFrame()
        range_table["min"] = groupby_segment[feature].min()
        range_table["max"] = groupby_segment[feature].max()
        
#        print(feature)
#        print(range_table)
        
#        range_table["range"] = range_table["min"].apply(lambda x: "{}".format(x))
#        print(range_table)
#        range_table["range"] = range_table.apply(lambda x: x["min"] if (type(x["min"]) == str or math.isnulln(x["min"])) else ("{0:,.2f}~{1:,.2f}".format(x["min"], x["max"]) if math.ceil(x["min"])!=x["min"] else "{0:,d}~{1:,d}".format(x["min"].astype(int), x["max"].astype(int))), axis=1)
        
        return range_table
    
    def plot_feature(self, lift_table):
        figure = plt.figure(facecolor='w', edgecolor='k')
        distribution_chart = figure.add_subplot(211)
        distribution_chart = self._arrange_distribution_chart(distribution_chart, lift_table)
        
        lift_chart = distribution_chart.twinx()
        lift_chart = self._arrange_lift_chart(lift_chart, lift_table)
        
        table_chart = figure.add_subplot(212)
        table_chart.axis('off')
        table_chart = self._arrange_table_chart(table_chart, lift_table)

        plt.show()
        return figure
        

    def __print_variable(self, x): 
        str_ = None 
        if(type(x) == str): 
            str_ = x
        
        elif(x == math.ceil(x)): 
            str_ = "{:,.0f}".format(x)

        elif(type(x) == float): 
            str_ = "{:,.2f}".format(x)
            
        return str_

    def _arrange_table_chart(self, table_chart, lift_table): 
        bin_table = lift_table.bin_table.transpose(copy=True)
        columns = list(bin_table.index)

        lift_text = pd.DataFrame()
        
        rows = ["MIN","MAX","Y%","COUNT%", "COUNT","EVENT"]
        lift_text["MIN"] = bin_table["MIN"].apply(self.__print_variable)
        lift_text["MAX"] = bin_table["MAX"].apply(self.__print_variable)
        
        lift_text["Y%"] = bin_table["Y%"].apply(lambda x: "{:.1f}%".format(x))
        lift_text["COUNT%"] = bin_table["COUNT%"].apply(lambda x: "{:.1f}%".format(x))
        lift_text["COUNT"] = bin_table["COUNT"].apply(lambda x: "{:,.0f}".format(x))
        lift_text["EVENT"] = bin_table["EVENT"].apply(lambda x: "{:,.0f}".format(x))

        lift_text = lift_text.transpose(copy=True)
        
        cell_text = []
        for row_index in lift_text.index:
            row = lift_text.loc[row_index]
            cell_text.append(["{}".format(x) for x in row])
        
        table_chart = table_chart.table( cellText = cell_text
                                , colLabels = columns 
                                , rowLabels = rows
                                , loc = 'center'
                                , colLoc = 'right'
                                , edges = 'horizontal'
                                , fontsize = 12
                                )
        table_chart.auto_set_font_size(False)
        table_chart.set_fontsize(6)
        table_chart.scale(1, 1.5)
        plt.tight_layout(rect = [0.05, 0, 1, 1])
        
        return table_chart
    
    def _arrange_distribution_chart(self, distribution_chart, lift_table): 
        feature = lift_table.feature
        bin_table = lift_table.bin_table.transpose(copy=True)
        
        width = 0.5
        distribution_chart.bar(bin_table.index, bin_table['COUNT%'], width, label = 'COUNT%', color='silver')
        
        for i in range(len(bin_table)): 
            if(bin_table["MIN"][i]=='NaN'): 
                distribution_chart.bar(bin_table.index[i], bin_table['COUNT%'][i], width, 
                                       label = 'COUNT%', color='whitesmoke', edgecolor='grey')
            else: 
                distribution_chart.bar(bin_table.index[i], bin_table['COUNT%'][i], width, 
                                       label = 'COUNT%', color='silver')
        
        distribution_chart.get_xaxis().set_visible(False)
        distribution_chart.set_xlabel(''.format(feature))
        distribution_chart.set_ylabel('COUNT% (%)', color='dimgrey')
        distribution_chart.set_ylim(0, 100)
        
        distribution_chart.tick_params(axis='y', colors='dimgrey', which='both')
        distribution_chart.legend(['COUNT%'], loc=2)
        
        return distribution_chart
    
    def _arrange_lift_chart(self, lift_chart, lift_table, colors='blue'): 
        bin_table = lift_table.bin_table.transpose(copy=True)
        
        last_row_num = len(bin_table)-1
        if(bin_table["MIN"][last_row_num] == 'NaN'): 
            lift_chart.plot(bin_table.loc[0:last_row_num-1,].index, 
                            bin_table['Y%'].loc[0:last_row_num-1,], 
                            label = 'Y%', color=colors, marker='o', linestyle='dashed')
            lift_chart.plot(bin_table.loc[last_row_num:last_row_num,].index, 
                            bin_table['Y%'].loc[last_row_num:last_row_num,], 
                            label = 'Y%', color=colors, marker='o', linestyle='dashed')
        else: 
            lift_chart.plot(bin_table.index, bin_table['Y%'], 
                            label = 'Y%', color=colors, marker='o', linestyle='dashed')


        lift_chart.set_ylabel('Y% (%)', color='blue')
        max_y = math.ceil(bin_table['Y%'].max()/5)*5
        lift_chart.set_ylim(0, max_y)
        lift_chart.set_title(lift_table.title)
        
        lift_chart.tick_params(axis='y', colors='blue', which='both')
        lift_chart.legend(['Y%'], loc=1)
        
        return lift_chart

    def to_pdf_feature_chart(self, figure, filepath):
        figure.savefig(filepath)

#%%
import copy        

from matplotlib import cm 
import numpy as np

class LiftPlotter(): 
    cnt=0
    
    def __init__(self): 
        self._xy_plotter = XYPlotter()
        self._figure = None 
    
    def _create_performance_info(self, performance): 
        performance = performance.transpose(copy=True)
        performance = performance.rename(columns={'cnt':'Count', 'bounce_cnt': 'Bounce', 'bounce_pct': 'Bounce%', 'LR_Correction': 'Correction'})
        
        performance = performance[['Count','Y%', 'KS','IV', 'Bounce', 'Bounce%', 'Correction']]
        
        performance['Count'] = performance['Count'].apply("{:,}".format)
        performance['Y%'] = (performance['Y%']*100).apply("{:.2f}".format)
        
        performance['KS'] = performance['KS'].apply("{:.2f}".format)
        performance['IV'] = performance['IV'].apply("{:.2f}".format)
        performance['Bounce%'] = (performance['Bounce%']*100).apply("{:.1f}".format)
        
        correction_cnt = performance[performance.index=='modeling']['Correction'].apply(len)
        performance['Correction']['modeling'] = correction_cnt['modeling'] 
        
        return performance
    
    def _arrange_bin_table(self, bin_table): 
        destination_bin_table = bin_table[["bin","event_rate", "p_total"]].copy(deep=True)
        destination_bin_table["event_rate"] = destination_bin_table["event_rate"]*100
        destination_bin_table["p_total"] = destination_bin_table["p_total"]*100
        destination_bin_table = destination_bin_table.rename(columns={"bin": "BIN", "event_rate": "Y%", "p_total": "COUNT%"})
        
        return destination_bin_table
    
    
    def plot(self, modeler, colormap="CMRmap_r", max_=0.9, min_=0.25):
        modeler = copy.deepcopy(modeler)
        model_name = modeler.model_id
        
        table_info = self._create_performance_info(modeler.performance)

        datasets = modeler.bin_table.columns.levels[0]
        bin_tables = dict()
        for dataset in datasets: 
            bin_table = modeler.bin_table[dataset]
            bin_table = self._arrange_bin_table(bin_table)
            bin_tables[dataset] = bin_table

#        for dataset in datasets: 
#            name = "{}_2".format(dataset)
#            bin_table = modeler.bin_table[dataset]
#            bin_table = self._arrange_bin_table(bin_table)
#            bin_table["Y%"] = bin_table["Y%"]+3
#            bin_table["COUNT%"] = bin_table["COUNT%"]
#            bin_tables[name] = bin_table
#        
#        for dataset in datasets: 
#            name = "{}_3".format(dataset)
#            bin_table = modeler.bin_table[dataset]
#            bin_table = self._arrange_bin_table(bin_table)
#            bin_table["Y%"] = bin_table["Y%"]+5
#            bin_table["COUNT%"] = bin_table["COUNT%"]
#            bin_tables[name] = bin_table
        
        
        self._figure = self._xy_plotter.plot(model_name, bin_tables, table_info, colormap, max_, min_)
    
    def to_pdf(self, filepath="Models\\", model_id=None): 
        self._xy_plotter.to_pdf_feature_chart(self._figure, "{0}\\{1}.pdf".format(filepath, model_id))



class MyLiftPlotter(LiftPlotter): 
    def _reverse_bin_table(self, bin_table_dataset): 
        destination_bin_table = copy.deepcopy(bin_table_dataset)
        
        destination_bin_table = destination_bin_table.sort_values(by=["BIN"], inplace=False, ascending=False)
        
        destination_bin_table = destination_bin_table.reset_index(drop=True)
        destination_bin_table["BIN"] = destination_bin_table.index
        
        return destination_bin_table
        
        
    
    def plot(self, bin_table, performance, colormap="CMRmap_r", max_=0.9, min_=0.25, legend_loc=1): 
        model_name = "Model"
        table_info = self._create_performance_info(performance)
        
        datasets = bin_table.columns.levels[0]
        bin_tables = dict()
        for dataset in datasets: 
            bin_table_dataset = bin_table[dataset]
            bin_table_dataset = self._arrange_bin_table(bin_table_dataset)
            bin_table_dataset = self._reverse_bin_table(bin_table_dataset)
            bin_tables[dataset] = bin_table_dataset

        self._figure = self._xy_plotter.plot(model_name, bin_tables, table_info, colormap, max_, min_, legend_loc=legend_loc)
        
#        print(bin_table)
        
        


    
class XYPlotter(XYAnalyzer): 
    def __init__(self): 
        pass
    
    def _arrange_lift_chart(self, lift_chart, bin_tables, model_name, colors, legend_loc=1, fontsize=14): 
        cnt = 0 
        max_y_all = 5 
        for tag in bin_tables: 
            bin_table = bin_tables[tag]
            
            color = colors[cnt]
            cnt = cnt+1 

            max_y = math.ceil(bin_table['Y%'].max()/2.5)*2.5
            max_y_all = max(max_y, max_y_all)
            
            if(tag == 'modeling'): 
                lift_chart.plot(bin_table["BIN"], bin_table['Y%'], 
                                label = 'Y%', 
                                color=color, marker=',', linestyle='-', linewidth=8)
            else: 
                lift_chart.plot(bin_table["BIN"], bin_table['Y%'], 
                                label = 'Y%', 
                                color=color, marker=',', linestyle='-')
        lift_chart.set_ylabel('Y% (%)', color='black', fontsize=fontsize)
        lift_chart.set_ylim(0, max_y_all)
        lift_chart.set_xticks(bin_table["BIN"])
        
        lift_chart.get_xaxis().set_visible(True)
        lift_chart.set_xlim(min(bin_table.BIN)-0.5,max(bin_table.BIN)+0.5)
        lift_chart.set_title("{0}".format(model_name), fontsize=fontsize+2)
        
        lift_chart.tick_params(axis='x', colors='black', which='both', labelsize=fontsize)
        lift_chart.tick_params(axis='y', colors='black', which='both', labelsize=fontsize)
        lift_chart.legend(list(bin_tables.keys()), loc=legend_loc)
        
        return lift_chart
    
    def _arrange_distribution_chart(self, distribution_chart, bin_tables, colors, fontsize): 
        cnt = 0
        max_y_all = 20
        for tag in bin_tables: 
            bin_table = bin_tables[tag]

            color = colors[cnt]
            cnt = cnt+1
            
            max_y = math.ceil(bin_table['COUNT%'].max()/5)*5
            max_y_all = max(max_y, max_y_all)
            
            seg_cnt = len(bin_tables)+2
            half_cnt = len(bin_tables)/2
            width = 1/seg_cnt
            distribution_chart.bar(bin_table.index+(cnt-half_cnt)*(width), bin_table['COUNT%'], width, label = 'COUNT%', color=color)
            
            distribution_chart.get_xaxis().set_visible(False)
            distribution_chart.set_xlabel(''.format(tag), fontsize=fontsize)
            
            distribution_chart.set_xticks(bin_table["BIN"])
            distribution_chart.set_xlim(min(bin_table.BIN)-0.5,max(bin_table.BIN)+0.5)
            distribution_chart.set_ylabel('COUNT% (%)', color='black', fontsize=fontsize)
            distribution_chart.set_ylim(0, max_y_all)
            
            distribution_chart.set_xticks(bin_table["BIN"])
            
            distribution_chart.tick_params(axis='x', colors='black', which='both', labelsize=fontsize)
            distribution_chart.tick_params(axis='y', colors='black', which='both', labelsize=fontsize)

        return distribution_chart
    
    def _arrange_table_chart(self, table_chart, table_info, fontsize): 
        rows = list(table_info.index)
        columns = table_info.columns

        cell_text = []
        for row_index in table_info.index:
            row = table_info.loc[row_index]
            cell_text.append(["{}".format(x) for x in row])
        
        table_chart = table_chart.table( cellText = cell_text
                                , colLabels = columns 
                                , rowLabels = rows
                                , loc = 'center'
                                , colLoc = 'right'
                                , edges = 'horizontal'
                                , fontsize = fontsize
                                )
        table_chart.auto_set_font_size(False)
        table_chart.set_fontsize(fontsize)
        table_chart.scale(1, 1.5)
        plt.tight_layout(rect = [0.05, 0, 1, 1])
        
        return table_chart
    
    def __display_colorlist(self, color_list): 
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        figure = plt.figure()
        axis = figure.add_subplot(111)
        axis.imshow(gradient,  aspect='auto', cmap = color_list, origin='lower')
        figure.show()
    
    
    def _get_colors(self, dataset_cnt, colormap, max_, min_): 
        
        color_list = cm.get_cmap(colormap, 256)
#        self.__display_colorlist(color_list)

        colors=['darkgray']
        max_ = max_
        min_ = min_
        cnt = dataset_cnt-1
        if(cnt==1): 
            gap=0
        else: 
            gap = (max_-min_)/(cnt-1)
        
        for i in range(cnt): 
            colors.append(color_list(max_-i*gap))
        
        return colors
        
    
    def plot(self, model_name, bin_tables, table_info, colormap, max_, min_,legend_loc=1, fontsize=12):
        
        colors = self._get_colors(len(bin_tables), colormap, max_, min_)
        
        figure = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        
        lift_chart = figure.add_subplot(211)
        lift_chart = self._arrange_lift_chart(lift_chart, bin_tables, model_name, colors, legend_loc, fontsize)

        distribution_chart = figure.add_subplot(413)
        distribution_chart = self._arrange_distribution_chart(distribution_chart, bin_tables, colors, fontsize)
        
        table_chart = figure.add_subplot(414)
        table_chart.axis('off')
        table_chart = self._arrange_table_chart(table_chart, table_info, fontsize)
        
        plt.show()
        return figure 