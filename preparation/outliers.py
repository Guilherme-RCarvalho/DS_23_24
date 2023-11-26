'''
file:       outliers.py
'''

from pandas import read_csv, DataFrame, Series
from auxiliar.dslabs_functions import (
    NR_STDEV,
    get_variable_types,
    determine_outlier_thresholds_for_var,
)

class Outliers:

    def __init__(self, filename, file_tag, data, n_std):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.n_std	  = n_std
        self.numeric_vars: list[str] = get_variable_types(self.data)["numeric"]
        self.summary5: DataFrame = self.data[self.numeric_vars].describe()
        
    def drop_outliers(self):
        if self.numeric_vars is not None:
            df: DataFrame = self.data.copy(deep=True)
            for var in self.numeric_vars:
                top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
                    self.summary5[var], threshold=self.n_std
                )
                outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
                df.drop(outliers.index, axis=0, inplace=True)
            df.to_csv(f"data/{self.file_tag}_drop_outliers.csv", index=True)
            print(f"Data after dropping outliers: {df.shape}")
        else:
            print("There are no numeric variables")
            
    def median_outliers(self):
        if [] != self.numeric_vars:
            df: DataFrame = self.data.copy(deep=True)
            for var in self.numeric_vars:
                top, bottom = determine_outlier_thresholds_for_var(
                    self.summary5[var], threshold=self.n_std
                )
                median: float = df[var].median()
                df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)
            df.to_csv(f"data/{self.file_tag}_replacing_outliers.csv", index=True)
            print("Data after replacing outliers:", df.shape)
            print(df.describe())
        else:
            print("There are no numeric variables")

    def truncate_outliers(self):
        if [] != self.numeric_vars:
            df: DataFrame = self.data.copy(deep=True)
            for var in self.numeric_vars:
                top, bottom = determine_outlier_thresholds_for_var(self.summary5[var], threshold=self.n_std)
                df[var] = df[var].apply(
                    lambda x: top if x > top else bottom if x < bottom else x
                )
            df.to_csv(f"data/{self.file_tag}_truncate_outliers.csv", index=True)
            print("Data after truncating outliers:", df.shape)
            print(df.describe())
        else:
            print("There are no numeric variables")
		
        