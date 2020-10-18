import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import norm, beta, normaltest, kruskal
from sklearn.model_selection import KFold, train_test_split, cross_validate

__author__ = "Francesco Rinarelli"
__email__ = "f.rinarelli@gmail.com"

file_base = "data/"

class SupervisedDataframe:
    '''A class that uses pandas MultiIndex to manage the trasnformation of
    the training and the test data at the same time.
    The use of the MultiIndex requires caution, as deleting rows does not remove them
    but just transform them in NaN, so dropna() is always required when extracting the train
    or the set data.'''
    def __init__(self, train_src, test_src, target_src = ""):
        train = pd.read_csv(file_base + train_src)
        if train_src != "":
            train_target = pd.read_csv(file_base + target_src)
            train = train.merge(train_target, how="left")
        self.target_col = train.columns[-1]
            
        test = pd.read_csv(file_base + test_src)
        
        self.merged = pd.concat({"train":train,"test":test})
        
        self.categoricals = self.merged.select_dtypes("object").columns.tolist()
        self.numericals = self.merged.select_dtypes("number").columns.tolist()
        self.dates = self.merged.select_dtypes("datetime").columns.tolist()
        
        print("created dataframes with implied target column \"{}\"".format(self.target_col))
        print("Implied categorical variables: " + str(self.categoricals))
        print("Implied numerical variables: " + str(self.numericals))
        print("Implied datetime variables: " + str(self.dates))
        
        print("information about the train DataFrame:")
        self.merged.loc["train"].info()
        print("information about the test DataFrame:")
        self.merged.loc["test"].drop(["salary"],axis=1).info()
        
    def set_target(self, new_target_col):
        '''Change the target column.'''
        self.target_col = new_target_col
        
    def get_train(self, dropna=False):
        '''Return the training set (including the target column if present).'''
        train = self.merged.loc["train"] 
        if dropna:
            return train.dropna()
        else:
            return train
    
    def get_test(self):
        '''Return the test set'''
        return self.merged.loc["test"].drop(columns=["salary"]).dropna()
                            
    def split_feature_target(self):
        '''Return the train set splitted into features and target.'''
        train_features = self.merged.loc["train"].drop(columns=[self.target_col]).dropna()
        train_target = self.merged.loc["train"][self.target_col].dropna()
        return (train_features, train_target)
    
    def plot_vs_target(self,col):
        '''Plot the target against one feature column. At the moment, still assumes that
        the feature is numerical.'''
        def key(val):
            return self.merged.loc["train"][self.merged.loc["train"][col]==val]\
                                                        [self.target_col].mean()
        order = sorted(self.merged.loc["train"][col].dropna().unique(), key=key)
        sb.boxplot(x=col, y=self.target_col, data=self.merged.loc["train"].dropna(), order=order)
        
    def plot_variable(self, col, fit=None, type="dist"):
        '''Plot the distribution of a feature, for both the train and the test set.
        if the feature is the target, does not draw the distribution for the test set.'''
        fig,ax = plt.subplots(1,2, sharey=True, figsize=(20,10))
        fig.suptitle(col + ": distribution", fontsize=14, fontweight="bold")
        # using distplot for numerical variables
        if col in self.numericals:
            ax[0].set_title("train:" + col)
            sb.distplot(self.merged.loc["train"][col].dropna(), fit=fit, ax=ax[0])
            if col != self.target_col:
                ax[1].set_title("test:" + col)
                sb.distplot(self.merged.loc["test"][col].dropna(), fit=fit, ax=ax[1])
        # using countplot for categorical variables
        elif col in self.categoricals:
            ax[0].set_title("train:" + col)
            sb.countplot(y=col, data=self.merged.loc["train"], ax=ax[0])
            if col != self.target_col:
                ax[1].set_title("test:" + col)
                sb.countplot(y=col, data=self.merged.loc["test"], ax=ax[1])
        else:
            print("Time series plot has not been implemented yet")
    
    def set_categoricals(self, new_list, replace=False):
        '''Sets the list of the categorical features.'''
        if replace:
            self.categoricals = new_list
        else:
            self.categoricals.extend(new_list)
        
    def set_numericals(self, new_list, replace=False):
        '''Sets the list of the numerical features.'''
        if replace:
            self.numericals = new_list
        else:
            self.numericals.extend(new_list)
        
    def set_dates(self, new_list, replace=False):
        '''Set the list of the date-wise features.'''
        if replace:
            self.dates = new_list
        else:
            self.dates.extend(new_list)
        
    def check_nulls(self, erase=False):
        '''Check whether there are null features in both the train and the test sets.
        There is the option to delete the relative rows.'''
        # we need to be careful, as the target will be NA on the test part of the DF
        null_vals = self.merged.drop([self.target_col], axis=1).isna().values.any()
        null_ixs = self.merged.drop([self.target_col], axis=1).isna().index.values
        if not null_vals:
            print("there are no NA values in the training dataFrame.")
        else:
            if erase:
                self.merged.loc[null_ixs].dropna(inplace=True)
                print("all rows with missing values were deleted.")
            else:
                "missing values were found, but not deleted."
        return null_vals
    
    def create_grouped_stats(self, col_list):
        '''Create group statistics out of a subset of categorical variables.
		
        Args:
        - self: will be used for operating on the class 'merged' dataframe
        - col_list: a list of categorical columns the merged df will be grouped by.
        Returns:
          nothing
        Changes:
        - adds three columns (mean, median, variance) to the class 'merged' dataframe.
        Notes:
        Only the portion corresponding to the training set will be used to create the grouped
        statistics, as the alternative would be a glaring instance of data leaking. 
        '''
        if not all([col in self.categoricals for col in col_list]):
            print("Cannot execute: some of the chosen columns are numerical.")
            return
        
        # group the training set according to the column list provided
        group = self.merged.loc["train"].dropna().groupby(col_list)[self.target_col]
        mean = group.mean().reset_index().rename(columns={self.target_col:"groupMean"})
        median = group.median().reset_index().rename(columns={self.target_col:"groupMedian"})
        var = group.var().reset_index().rename(columns={self.target_col:"groupVar"})
        
        # apply the statistics to both the training and the validation set according to the 
        # columns
        for stat in [mean, median, var]:
            self.merged = self.merged.reset_index(0).merge(stat,how="left").set_index("level_0")
    
    def to_one_hot(self,features):
        '''Transform some of the categorical features into binary features.'''
        for feat in features:
            self.merged = pd.concat([self.merged, pd.get_dummies(self.merged[feat],\
                                                                             prefix=feat)],axis=1)
            self.merged = self.merged.drop(columns=[feat])
    
    def dict_replace(self,col,col_dict):
        '''Transform a column based on a dictionary.'''
        self.merged.replace({col:col_dict}, inplace=True)

    def scale(self, features,method="minmax"):
        '''Scale the selected numerical columns according to minmax or standard deviation.'''
        if method == "standard":
            for feat in features:
                self.merged[feat] = (self.merged[feat] - self.merged[feat].mean())/\
                                        self.merged[feat].std()
        elif method == "minmax":
            for feat in features:
                self.merged[feat] = (self.merged[feat] - self.merged[feat].mean())/\
                                     (self.merged[feat].max() - self.merged[feat].min())
        else:
            "this method is not implemented"
    
    def __getattr__(self, attr):
        '''Get the pandas attributes from the embedded MultiIndex.'''
        getattr(self.merged, attr)