'''
file:       profilling_ts.py
'''
from matplotlib.pyplot import figure, show, savefig
from pandas import Series, DataFrame

from auxiliar.dslabs_functions import HEIGHT, plot_line_chart, ts_aggregation_by, plot_ts_multivariate_chart


class Transformation:

    def __init__(self, filename, file_tag, target, index, data, series):
        self.target = target
        self.filename = filename
        self.file_tag = file_tag
        self.index = index
        self.data = data
        self.series = series

    def plot_time_series_line_chart(self):
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            self.series.index.to_list(),
            self.series.to_list(),
            xlabel=self.series.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} hourly {self.target}",
        )
        savefig(f"images/TS/Transformation/{self.file_tag}_ts_line_chart.png")
        show()

    def plot_daily_aggregated_line_chart(self):
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        ss_agg: Series = ts_aggregation_by(self.series, gran_level="D", agg_func="sum")
        plot_line_chart(
            ss_agg.index.to_list(),
            ss_agg.to_list(),
            xlabel=ss_agg.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} daily {self.target}",
        )
        savefig(f"images/TS/Transformation/{self.file_tag}_daily_aggregated_line_chart.png")
        show()
