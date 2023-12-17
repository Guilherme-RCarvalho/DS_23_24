from numpy import arange
from pandas import DataFrame, Series
from matplotlib.pyplot import figure, savefig, show, subplots
from auxiliar.dslabs_functions import plot_line_chart, HEIGHT, series_train_test_split, plot_forecasting_eval, plot_forecasting_series
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression

class ts_Smoothing:

    def __init__(self, filename, file_tag, data, target, index):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target
        self.index    = index
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

    def ts_smoothing_lr(self):
        sizes: list[int] = [25, 50, 75, 100]
        for i in range(len(sizes)):
            lr_series = self.data.rolling(window=sizes[i]).mean()
            train, test = series_train_test_split(lr_series.dropna(), trn_pct=0.90)
            trnX = arange(len(train)).reshape(-1, 1)
            trnY = train.to_numpy()
            tstX = arange(len(train), len(lr_series.dropna())).reshape(-1, 1)
            tstY = test.to_numpy()
            model = LinearRegression()
            model.fit(trnX, trnY)

            prd_trn: Series = Series(model.predict(trnX), index=train.index)
            prd_tst: Series = Series(model.predict(tstX), index=test.index)

            plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"size={sizes[i]}")
            savefig(f"images/timeSeries/transformation/smoothing/{self.file_tag}_{self.target}_smoothing_lr_{sizes[i]}.png")
            plot_forecasting_series(train, test, prd_tst, title=f"{self.file_tag} - Linear Regression", xlabel=self.index, ylabel=self.target,)
            savefig(f"images/timeSeries/transformation/smoothing/{self.file_tag}_{self.target}_smoothing_lr_{sizes[i]}_series.png")