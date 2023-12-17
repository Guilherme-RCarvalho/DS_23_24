from pandas import read_csv, DataFrame, Series, Index, Period
from matplotlib.pyplot import figure, savefig, show, subplots
from auxiliar.dslabs_functions import plot_line_chart, HEIGHT
from matplotlib.axes import Axes
from matplotlib.figure import Figure

class ts_Granularity:

    def __init__(self, filename, file_tag, data, target):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target
        self.series: Series = self.data[self.target]
        
    def ts_aggregation_by(self,
        data: Series | DataFrame,
        gran_level: str = "D",
        agg_func: str = "mean",
) ->     Series | DataFrame:
        print(data)
        df: Series | DataFrame = self.series.copy()
        index: Index[Period] = df.index.to_period(gran_level)
        df = df.groupby(by=index, dropna=True, sort=True).agg(agg_func)
        df.index.drop_duplicates()
        df.index = df.index.to_timestamp()
    
        return df
    
    def ts_aggregation(self):
        grans: list[str] = ["W", "M", "Y"]
        fig: Figure
        axs: list[Axes]
        fig, axs = subplots(len(grans), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(grans)))
        fig.suptitle(f"{self.file_tag} {self.target} aggregation study")

        for i in range(len(grans)):
            ss: Series = self.ts_aggregation_by(self.series, grans[i])
            plot_line_chart(
                ss.index.to_list(),
                ss.to_list(),
                ax=axs[i],
                xlabel=f"{ss.index.name} ({grans[i]})",
                ylabel=self.target,
                title=f"granularity={grans[i]}",
            )
        savefig(f"images/timeSeries/profiling/granularity/{self.file_tag}_{self.target}_aggregation.png")