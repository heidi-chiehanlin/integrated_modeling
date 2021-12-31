# Introduction
This is an integrated program that performs data-mining cycle with reusable modules. Major classes/modules include:
- pre-modeling analysis
- data preprocessing pipeline
- regression model training
- performance evaluation and visualization

---

# Main Features
To begin with, [Training notebook](https://github.com/heidi-chiehanlin/integrated_modeling/blob/4141d3c6fd04b99d52390c7e4bfaae3c47758691/Model%20Development/build_model_main.py) contains all the sample code you'll need for starting a data mining project with our libraries.

## Dataset
A class that stores target value, features and properties of features. Most of our function depends on this class as an input.

            ================ ============================= ======================== 
            Parameter        Data Type                     Description
            ================ ============================= ======================== 
            x                ``pandas.DataFrame``          Features for Modeling
            y                ``pandas.DataFrame``          Y for Modeling
            attribute_list   ``AttributeList``             Another class that stores schema    
            ================ ============================= ======================== 
            
            * schema *
            ============= =============== =============
            Category      Field           Type
            ============= =============== =============
            X             feature1        nominal
            X             feature2        numeric
            Y             y               numeric
            Key           key             numeric
            ============= =============== =============

## Binner
A class that group datapoints into bins by setting the threshold either manually or automatically (so called "binning"). After binning the data, we analyze the performance of classification models with a grouped perspective.

`Binner` include three main functions: 
- **FIT:** creates bins & upper_bounds (thresholds) for input data according to specified methods
- **TRANSFORM:** transforms data according to the thresholds 
- **PLOT:** visualize lift charts for performance evaluation

Two kinds of Binner that support different datatypes:
- **NumericFeatureBinner:** for continuous datatype
- **NominalFeatureBinner:** for nominal, discrete datatype

Their differences are mainly on the **FIT** side. Below we take `NumericFeatureBinner` as the example.

`fit(X, y, max_bins=10, method='percentile')`

- `max_bins`: the maximum bin count (It happens when data can't be split into the assigned number of bins.)
- `method`: how to create bins, there are 3 options:
    - percentile: split evenly in sample percentiles. Ex: if bin_count=10, creates decile bins
    - range: split evenly in feature range. Ex: if feature ranges from 1~10 and bin_count=10, bins=(-inf,1], (1,2], ...(9,10]
    - tree: split with DecisionTree algorithm, where max_leaves_count=bin_count
        
`fit_auto_decrease_maxbins(X, y, max_bins, method='tree', criteria='event_rate(%)')`
- start from max_bins, if criteria is not in monotonic order, then decrease bin_count by 1 until criteria reaches monotonic order 
- `criteria`: which criteria to check if ordered, there are 2 options: 
    - woe
    - event_rate(%)

`fit_auto_merge_bins(X, y, max_bins, method='range', criteria='woe')`
- start from max_bins, if criteria is not in monotonic order, then merge the bins not in order with its previous bin until criteria reaches monotonic order 
- `criteria`: which criteria to check if ordered, there are 2 options: 
    - woe
    - event_rate(%)
    
`fit_manual(feature_name, upper_bounds)`
- allows manual-defined  binning rules


## Modeler
A class that handles from model training, fitting, to performance evaluation. A ``Modeler`` can do:

- Train the classifier to build a model (`Modeler.fit_and_transform_evaluate`)
- Insert additional validation datasets to evaluate the existed model (`Modeler.transform_and_evaluate`)
- Insert a set of predicted values to evaluate model performance without doing fit_and_transform (`Modeler.evaluate_without_clf`)
- Score datasets to get predicted values (`Modeler.simply_scoring`)
- Being collected by ``ModelerCollector`` for further model comparing and selection
    
---
documentation updated: Dec. 30, 2021
