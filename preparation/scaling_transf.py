from pandas import DataFrame, Series
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import subplots, show, savefig


class Scaling:

    def __init__(self, filename, file_tag, data, target):
        self.filename = filename
        self.file_tag = file_tag
        self.data = data
        self.target = target
        self.target_data: Series = self.data.pop(self.target)
        self.vars: list[str] = self.data.columns.to_list()

    def standard_scaler(self):
        transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
            self.data
        )
        df_zscore = DataFrame(transf.transform(self.data), index=self.data.index)
        df_zscore.columns = self.vars
        df_zscore[self.target] = self.target_data
        df_zscore.to_csv(f"data/{self.file_tag}_scaled_zscore.csv", index="id")
        return df_zscore

    def min_max_scaler(self):
        transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(self.data)
        df_minmax = DataFrame(transf.transform(self.data), index=self.data.index)
        df_minmax.columns = self.vars
        df_minmax[self.target] = self.target_data
        df_minmax.to_csv(f"data/{self.file_tag}_scaled_minmax.csv", index="id")
        return df_minmax

    def scaler_plot(self, df_zscore, df_minmax):
        fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
        axs[0, 1].set_title("Original data")
        self.data.boxplot(ax=axs[0, 0])
        axs[0, 0].set_title("Z-score normalization")
        df_zscore.boxplot(ax=axs[0, 1])
        axs[0, 2].set_title("MinMax normalization")
        df_minmax.boxplot(ax=axs[0, 2])
        savefig(f"images/{self.file_tag}_scaling_boxplot.png")