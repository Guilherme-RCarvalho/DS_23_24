'''
file:       profilling_ts.py
'''
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, show, savefig, subplots
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


    def smoothing_different_winsizes(self):
        sizes: list[int] = [25, 50, 75, 100]
        fig: Figure
        axs: list[Axes]
        fig, axs = subplots(len(sizes), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(sizes)))
        fig.suptitle(f"{self.file_tag} {self.target} after smoothing")

        for i in range(len(sizes)):
            ss_smooth: Series = self.series.rolling(window=sizes[i]).mean()
            plot_line_chart(
                ss_smooth.index.to_list(),
                ss_smooth.to_list(),
                ax=axs[i],
                xlabel=ss_smooth.index.name,
                ylabel=self.target,
                title=f"size={sizes[i]}",
            )
        savefig(f"images/TS/Transformation/{self.file_tag}_winsizes_smoothing.png")
        show()

    def specific_smoothing(self, WIN_SIZE):
        ss_smooth: Series = self.series.rolling(window=WIN_SIZE).mean()

        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            ss_smooth.index.to_list(),
            ss_smooth.to_list(),
            xlabel=ss_smooth.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} {self.target} after smoothing (size={WIN_SIZE})",
        )
        savefig(f"images/TS/Transformation/{self.file_tag}_specific_smoothing.png")
        show()

    def differentiation(self):
        ss_diff: Series = self.series.diff()
        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_line_chart(
            ss_diff.index.to_list(),
            ss_diff.to_list(),
            title="Differentiation",
            xlabel=self.series.index.name,
            ylabel=self.target,
        )
        show()
