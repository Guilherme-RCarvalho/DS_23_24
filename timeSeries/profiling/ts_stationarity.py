from pandas import Series
from matplotlib.pyplot import savefig, subplots, show, gca, plot, legend, figure
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from auxiliar.dslabs_functions import HEIGHT, set_chart_labels, plot_line_chart

class ts_Stationarity:

    def __init__(self, filename, file_tag, data, target, index):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target
        self.index    = index
        self.series: Series = data[target]

    def plot_components(self,
    series: Series,
    title: str = "",
    x_label: str = "time",
    y_label: str = "",
    ) -> list[Axes]:
        decomposition: DecomposeResult = seasonal_decompose(series, model="add")
        components: dict = {
            "observed": series,
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
        }
        rows: int = len(components)
        fig: Figure
        axs: list[Axes]
        fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
        fig.suptitle(f"{title}")
        i: int = 0
        for key in components:
            set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
            axs[i].plot(components[key])
            i += 1
        return axs
    
    def ts_components(self):
        self.plot_components(
        self.series,
        title=f"{self.file_tag} weekly {self.target}",
        x_label=self.series.index.name,
        y_label=self.target,)  
        savefig(f"images/timeSeries/profiling/stationarity/{self.file_tag}_{self.target}_components.png") 

    def ts_stationarity_study(self):
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
        savefig(f"images/timeSeries/profiling/stationarity/{self.file_tag}_{self.target}_stationarity_study.png")

    def eval_stationarity(self) -> bool:
        result = adfuller(self.series)
        print(f"ADF Statistic: {result[0]:.3f}")
        print(f"p-value: {result[1]:.3f}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.3f}")
        print(f"The series {('is' if result[1] <= 0.05 else 'is not')} stationary")