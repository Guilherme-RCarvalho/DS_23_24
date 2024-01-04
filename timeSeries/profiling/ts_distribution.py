from numpy import sum,array
from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, savefig, show, subplots, setp
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from auxiliar.dslabs_functions import HEIGHT, plot_line_chart, ts_aggregation_by, set_chart_labels, plot_multiline_chart
#from timeSeries.profiling.ts_granularity import ts_aggregation_by

class ts_Distribtion:

    def __init__(self, filename, file_tag, data, target, index):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target
        self.index    = index
        self.series: Series = data[target]
        self.ss_months: Series = ts_aggregation_by(self.series, gran_level="M", agg_func=sum)
        self.ss_years: Series = ts_aggregation_by(self.series, gran_level="Y", agg_func=sum)
    
    def ts_distribution_boxplots(self):
        fig: Figure
        axs: array
        fig, axs = subplots(2, 3, figsize=(3 * HEIGHT, HEIGHT))
        set_chart_labels(axs[0, 0], title="WEEKLY")
        axs[0, 0].boxplot(self.series)
        set_chart_labels(axs[0, 1], title="MONTHLY")
        axs[0, 1].boxplot(self.ss_months)
        set_chart_labels(axs[0, 2], title="YEARLY")
        axs[0, 2].boxplot(self.ss_years)

        axs[1, 0].grid(False)
        axs[1, 0].set_axis_off()
        axs[1, 0].text(0.2, 0, str(self.series.describe()), fontsize="small")

        axs[1, 1].grid(False)
        axs[1, 1].set_axis_off()
        axs[1, 1].text(0.2, 0, str(self.ss_months.describe()), fontsize="small")

        axs[1, 2].grid(False)
        axs[1, 2].set_axis_off()
        axs[1, 2].text(0.2, 0, str(self.ss_years.describe()), fontsize="small")
        savefig(f"images/timeSeries/profiling/distribution/{self.file_tag}_{self.target}_boxplots.png")

    def ts_distribution_histograms(self):
        grans: list[Series] = [self.series, self.ss_months, self.ss_years]
        gran_names: list[str] = ["Weekly", "Monthly", "Yearly"]
        fig: Figure
        axs: array
        fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
        fig.suptitle(f"{self.file_tag} {self.target}")
        for i in range(len(grans)):
            set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=self.target, ylabel="Nr records")
            axs[i].hist(grans[i].values)
        savefig(f"images/timeSeries/profiling/distribution/{self.file_tag}_{self.target}_histograms.png")

    def get_lagged_series(self, series: Series, max_lag: int, delta: int = 1):
        lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
        for i in range(delta, max_lag + 1, delta):
            lagged_series[f"lag {i}"] = series.shift(i)
        return lagged_series

    def ts_autocorrelation_lag(self):
        figure(figsize=(3 * HEIGHT, HEIGHT))
        lags = self.get_lagged_series(self.series, 20, 10)
        plot_multiline_chart(self.series.index.to_list(), lags, xlabel=self.index, ylabel=self.target)
        savefig(f"images/timeSeries/profiling/distribution/{self.file_tag}_{self.target}_autocorrelation_lag.png")

    def ts_autocorrelation_study(self, series: Series, max_lag: int, delta: int = 1):
        k: int = int(max_lag / delta)
        fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
        gs = GridSpec(2, k, figure=fig)

        series_values: list = series.tolist()
        for i in range(1, k + 1):
            ax = fig.add_subplot(gs[0, i - 1])
            lag = i * delta
            ax.scatter(series.shift(lag).tolist(), series_values)
            ax.set_xlabel(f"lag {lag}")
            ax.set_ylabel("original")
        ax = fig.add_subplot(gs[1, :])
        ax.acorr(series, maxlags=max_lag)
        ax.set_title("Autocorrelation")
        ax.set_xlabel("Lags")
        return

    def ts_autocorrelation_correlogram(self):
        self.ts_autocorrelation_study(self.series, 10, 1)
        savefig(f"images/timeSeries/profiling/distribution/{self.file_tag}_{self.target}_autocorrelation_correlogram.png")