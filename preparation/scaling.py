
from pandas import DataFrame, Series
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class Scaling:

    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data

    def standard_scaler(self, target):
        vars: list[str] = self.data.columns.to_list()
        target_data: Series = self.data.pop(target)

        transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
            self.data
        )
        df_zscore = DataFrame(transf.transform(self.data), index=self.data.index)
        df_zscore[target] = target_data
        df_zscore.columns = vars
        df_zscore.to_csv(f"data/{self.file_tag}_scaled_zscore.csv", index="id")


    def min_max_scaler(self, target):
        
        target_data: Series = self.data.pop(target)

        transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(self.data)
        df_minmax = DataFrame(transf.transform(self.data), index=self.data.index)
        df_minmax[target] = target_data
        df_minmax.columns = vars
        df_minmax.to_csv(f"data/{self.file_tag}_scaled_minmax.csv", index="id")
        

