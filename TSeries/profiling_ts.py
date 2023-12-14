'''
file:       profilling_ts.py
'''
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, show, savefig, subplots
from pandas import Series, Index, Period
from auxiliar.dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by


class Profiling:

    def __init__(self, filename, file_tag, target, data, series):
        self.target = target
        self.filename = filename
        self.file_tag = file_tag
        self.data = data
        self.series = series


    def data_before(self):
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            self.series.index.to_list(),
            self.series.to_list(),
            xlabel=self.series.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} hourly {self.target}",
        )
        savefig(f"images/TS/{self.file_tag}_profiling_data_before.png")
        show()


    def aggregation_by(self):
        ss_days: Series = ts_aggregation_by(self.series, "D")
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            ss_days.index.to_list(),
            ss_days.to_list(),
            xlabel="days",
            ylabel=self.target,
            title=f"{self.file_tag} daily mean {self.target}",
        )
        savefig(f"images/TS/{self.file_tag}_profiling_daily_mean.png")
        show()

    def aggregation_study(self):
        grans: list[str] = ["D", "W", "M"]
        fig: Figure
        axs: list[Axes]
        fig, axs = subplots(len(grans), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(grans)))
        fig.suptitle(f"{self.file_tag} {self.target} aggregation study")

        for i in range(len(grans)):
            ss: Series = ts_aggregation_by(self.series, grans[i])
            plot_line_chart(
                ss.index.to_list(),
                ss.to_list(),
                ax=axs[i],
                xlabel=f"{ss.index.name} ({grans[i]})",
                ylabel=self.target,
                title=f"granularity={grans[i]}",
            )

        savefig(f"images/TS/{self.file_tag}_granularity_aggregation_study.png")
        show()