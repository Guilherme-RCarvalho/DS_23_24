from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, savefig, show, subplots
from auxiliar.dslabs_functions import plot_line_chart, HEIGHT
from matplotlib.axes import Axes
from matplotlib.figure import Figure

class ts_Smoothing:

    def __init__(self, filename, file_tag, data, target):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target
        self.series: Series = self.data[self.target]

    def ts_smoothing(self):
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
        savefig(f"images/timeSeries/transformation/smoothing/{self.file_tag}_{self.target}_smoothing.png")