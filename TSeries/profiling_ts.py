'''
file:       profilling_ts.py
'''
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, show, savefig, subplots, setp, legend, plot
from matplotlib.gridspec import GridSpec
from pandas import Series, Index, Period
from numpy import array
from auxiliar.dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by, set_chart_labels, \
    plot_multiline_chart, get_lagged_series, plot_components, eval_stationarity


class Profiling:

    def __init__(self, filename, file_tag, target, index, data, series):
        self.target = target
        self.filename = filename
        self.file_tag = file_tag
        self.index = index
        self.data = data
        self.series = series
        self.ss_weeks : Series = ts_aggregation_by(self.series, gran_level="W", agg_func=sum)


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

    def aggregation_by_w(self):
        ss_weeks: Series = ts_aggregation_by(self.series, gran_level="W", agg_func=sum)
        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_line_chart(
            ss_weeks.index.to_list(),
            ss_weeks.to_list(),
            xlabel="weeks",
            ylabel=self.target,
            title=f"{self.file_tag} weekly {self.target}",
        )
        savefig(f"images/TS/{self.file_tag}_weekly_{self.target}.png")
        show()

    def number_summary(self):
        ss_weeks: Series = ts_aggregation_by(self.series, gran_level="W", agg_func=sum)
        fig: Figure
        axs: array
        fig, axs = subplots(2, 2, figsize=(2 * HEIGHT, HEIGHT))
        set_chart_labels(axs[0, 0], title="HOURLY")
        axs[0, 0].boxplot(self.series)
        set_chart_labels(axs[0, 1], title="WEEKLY")
        axs[0, 1].boxplot(ss_weeks)

        axs[1, 0].grid(False)
        axs[1, 0].set_axis_off()
        axs[1, 0].text(0.2, 0, str(self.series.describe()), fontsize="small")

        axs[1, 1].grid(False)
        axs[1, 1].set_axis_off()
        axs[1, 1].text(0.2, 0, str(ss_weeks.describe()), fontsize="small")
        savefig(f"images/TS/{self.file_tag}_number_summary.png")
        show()

    def variables_distribution(self):
        ss_days: Series = ts_aggregation_by(self.series, gran_level="D", agg_func=sum)
        ss_months: Series = ts_aggregation_by(self.series, gran_level="M", agg_func=sum)
        ss_quarters: Series = ts_aggregation_by(self.series, gran_level="Q", agg_func=sum)

        grans: list[Series] = [self.series, ss_days, self.ss_weeks, ss_months, ss_quarters]
        gran_names: list[str] = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly"]
        fig: Figure
        axs: array
        fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
        fig.suptitle(f"{self.file_tag} variables distribution")
        for i in range(len(grans)):
            set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=self.target, ylabel="Nr records")
            axs[i].hist(grans[i].values)
        savefig(f"images/TS/{self.file_tag}_variables_distribution.png")
        show()

    def lag_plots(self):
        figure(figsize=(3 * HEIGHT, HEIGHT))
        lags = get_lagged_series(self.series, 20, 10)
        plot_multiline_chart(self.series.index.to_list(), lags, xlabel=self.index, ylabel=self.target)
        savefig(f"images/TS/{self.file_tag}_lag_plots.png")
        show()

    def autocorrelation_study(self, max_lag: int, delta: int = 1):
        k: int = int(max_lag / delta)
        fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
        gs = GridSpec(2, k, figure=fig)

        series_values: list = self.series.tolist()
        for i in range(1, k + 1):
            ax = fig.add_subplot(gs[0, i - 1])
            lag = i * delta
            ax.scatter(self.series.shift(lag).tolist(), series_values)
            ax.set_xlabel(f"lag {lag}")
            ax.set_ylabel("original")
        ax = fig.add_subplot(gs[1, :])
        ax.acorr(self.series, maxlags=max_lag)
        ax.set_title("Autocorrelation")
        ax.set_xlabel("Lags")
        savefig(f"images/TS/{self.file_tag}_autocorrelation_study.png")
        show()
        return

    def plot_time_series_components(self):
        plot_components(
            self.series,
            title=f"{self.file_tag} hourly {self.target}",
            x_label=self.series.index.name,
            y_label=self.target,
        )
        savefig(f"images/TS/{self.file_tag}_plot_time_series_components.png")
        show()

    def stationary_study(self):
        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_line_chart(
            self.series.index.to_list(),
            self.series.to_list(),
            xlabel=self.series.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} stationary study",
            name="original",
        )
        n: int = len(self.series)
        plot(self.series.index, [self.series.mean()] * n, "r-", label="mean")
        legend()
        savefig(f"images/TS/{self.file_tag}_stationary_study.png")
        show()

    def stationary_study_2(self):
        BINS = 10
        mean_line: list[float] = []
        n: int = len(self.series)

        for i in range(BINS):
            segment: Series = self.series[i * n // BINS : (i + 1) * n // BINS]
            mean_value: list[float] = [segment.mean()] * (n // BINS)
            mean_line += mean_value
        mean_line += [mean_line[-1]] * (n - len(mean_line))

        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_line_chart(
            self.series.index.to_list(),
            self.series.to_list(),
            xlabel=self.series.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} stationary study",
            name="original",
            show_stdev=True,
        )

        plot(self.series.index, mean_line, "r-", label="mean")
        legend()
        savefig(f"images/TS/{self.file_tag}_stationary_study_2.png")
        show()

    def eval_stationarity(self):
        x = eval_stationarity(self.series)
        print(f"The series {('is' if x else 'is not')} stationary")
