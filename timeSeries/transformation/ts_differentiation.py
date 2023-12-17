from pandas import Series
from auxiliar.dslabs_functions import plot_line_chart, HEIGHT
from matplotlib.pyplot import figure, savefig, subplots, suptitle
import matplotlib.pyplot as plt

class ts_Differentiation:

    def __init__(self, filename, file_tag, data, target):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target
        self.series: Series = self.data[self.target]

    def ts_differentiation(self):
        ss_diff_1: Series = self.series.diff()
        ss_diff_2: Series = ss_diff_1.diff()
        fig, axs = subplots(2, 1, figsize=(3 * HEIGHT, HEIGHT))
        fig.suptitle("Differentiation")
        plot_line_chart(
            ss_diff_1.index.to_list(),
            ss_diff_1.to_list(),
            ax=axs[0],
            xlabel=self.series.index.name,
            ylabel=self.target,
            title="First Derivative",
        )
        plot_line_chart(
            ss_diff_2.index.to_list(),
            ss_diff_2.to_list(),
            ax=axs[1],
            xlabel=self.series.index.name,
            ylabel=self.target,
            title="Second Derivative",
        )
        plt.tight_layout()
        savefig(f"images/timeSeries/transformation/differentiation/{self.file_tag}_{self.target}_differentiation.png")