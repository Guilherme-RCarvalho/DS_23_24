from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, savefig, show
from auxiliar.dslabs_functions import plot_line_chart, HEIGHT

class ts_Dimensionality:

    def __init__(self, filename, file_tag, data, target):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target

    def univariate_ts(self):
        series: Series = self.data[self.target]
        print("Nr. Records = ", series.shape[0])
        print("First timestamp", series.index[0])
        print("Last timestamp", series.index[-1])

        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            series.index.to_list(),
            series.to_list(),
            xlabel=series.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} weekly {self.target}",
        )
        savefig(f"images/timeSeries/profiling/dimensionality/{self.file_tag}_weekly_{self.target}.png")