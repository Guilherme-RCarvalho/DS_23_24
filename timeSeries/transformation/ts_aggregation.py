from numpy import save
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, savefig, show, subplot
from auxiliar.dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT

class ts_Aggregation:

    def __init__(self, filename, file_tag, data, target, index):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target
        self.index    = index
        self.series: Series = data[target]

    def ts_aggregation(self):
        figure(figsize=(3 * HEIGHT, 3 * HEIGHT / 2))
        
        # Weekly aggregation
        ax1 = subplot(3, 1, 1)
        plot_line_chart(
            self.series.index.to_list(),
            self.series.to_list(),
            xlabel=self.series.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} weekly {self.target}",
            ax=ax1
        )
        
        # Monthly aggregation
        ax2 = subplot(3, 1, 2)
        ss_agg_monthly: Series = ts_aggregation_by(self.series, gran_level="M", agg_func="sum")
        plot_line_chart(
            ss_agg_monthly.index.to_list(),
            ss_agg_monthly.to_list(),
            xlabel=ss_agg_monthly.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} monthly {self.target}",
            ax=ax2
        )
        
        # Yearly aggregation
        ax3 = subplot(3, 1, 3)
        ss_agg_yearly: Series = ts_aggregation_by(self.series, gran_level="Y", agg_func="sum")
        plot_line_chart(
            ss_agg_yearly.index.to_list(),
            ss_agg_yearly.to_list(),
            xlabel=ss_agg_yearly.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} yearly {self.target}",
            ax=ax3
        )
        savefig(f"images/timeSeries/transformation/aggregation/{self.file_tag}_{self.target}_aggregation.png")
