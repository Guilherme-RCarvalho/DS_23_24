from pandas import Series
from numpy import arange
from auxiliar.dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT, series_train_test_split, plot_forecasting_eval, plot_forecasting_series
from matplotlib.pyplot import figure, savefig, subplots, suptitle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class ts_Differentiation:

    def __init__(self, filename, file_tag, data, target, index, series):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target
        self.index    = index
        self.series: Series = series

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
        return ss_diff_1.to_frame().dropna(), ss_diff_2.to_frame().dropna()
    
    def ts_differentiation_lr(self):
        diffs : list[Series] = [self.series.to_frame().diff(), self.series.to_frame().diff().diff()]
        for i in range(len(diffs)):
            train, test = series_train_test_split(diffs[i].dropna(), trn_pct=0.90)
            trnX = arange(len(train)).reshape(-1, 1)
            trnY = train.to_numpy()
            tstX = arange(len(train), len(diffs[i].dropna())).reshape(-1, 1)
            tstY = test.to_numpy()
            model = LinearRegression()
            model.fit(trnX, trnY)

            prd_trn: Series = Series(model.predict(trnX), index=train.index)
            prd_tst: Series = Series(model.predict(tstX), index=test.index)

            plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"diff={i}")
            savefig(f"images/timeSeries/transformation/differentiation/{self.file_tag}_{self.target}_differentiation_lr_diff{i+1}.png")
            plot_forecasting_series(train, test, prd_tst, title=f"{self.file_tag} - Linear Regression", xlabel=self.index, ylabel=self.target,)
            savefig(f"images/timeSeries/transformation/differentiation/{self.file_tag}_{self.target}_differentiation_lr_diff{i+1}_series.png")