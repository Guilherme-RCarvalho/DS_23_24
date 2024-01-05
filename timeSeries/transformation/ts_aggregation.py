import numpy as np
from numpy import arange
from pandas import DataFrame, Series
from matplotlib.pyplot import figure, savefig, show, subplot, subplots
from auxiliar.dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT, series_train_test_split, plot_forecasting_eval, plot_forecasting_series
from sklearn.linear_model import LinearRegression
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
            title=f"{self.file_tag} 15T {self.target}",
            ax=ax1
        )
        
        # Monthly aggregation
        ax2 = subplot(3, 1, 2)
        ss_agg_h: Series = ts_aggregation_by(self.series, gran_level="H", agg_func="sum")
        plot_line_chart(
            ss_agg_h.index.to_list(),
            ss_agg_h.to_list(),
            xlabel=ss_agg_h.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} hourly {self.target}",
            ax=ax2
        )
        
        # Yearly aggregation
        ax3 = subplot(3, 1, 3)
        ss_agg_day: Series = ts_aggregation_by(self.series, gran_level="D", agg_func="sum")
        plot_line_chart(
            ss_agg_day.index.to_list(),
            ss_agg_day.to_list(),
            xlabel=ss_agg_day.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} Daily {self.target}",
            ax=ax3
        )
        savefig(f"images/timeSeries/transformation/aggregation/{self.file_tag}_{self.target}_aggregation.png")
        show()

    # def ts_aggregation_lr(self):
    #     grans: list[str] = ["W", "M", "Y"]
    #     for i in range(len(grans)):
    #
    #         lr_series = ts_aggregation_by(self.data, gran_level=grans[i], agg_func="sum")
    #         train, test = series_train_test_split(lr_series, trn_pct=0.90)
    #         trnX = arange(len(train)).reshape(-1, 1)
    #         trnY = train.to_numpy()
    #         tstX = arange(len(train), len(lr_series)).reshape(-1, 1)
    #         tstY = test.to_numpy()
    #         print("Size of training set:", len(trnX))
    #         print("Size of training set after splitting:", len(train))
    #
    #         if np.isnan(trnX).any() or np.isnan(trnY).any():
    #             print("NaN values found in the training set. Handle missing values before fitting the model.")
    #             continue
    #
    #         # Check for NaN values in training and testing datasets
    #         if train.isnull().any() or test.isnull().any():
    #             print("NaN values found in the dataset. Handle missing values before fitting the model.")
    #             continue
    #
    #         model = LinearRegression()
    #         model.fit(trnX, trnY)
    #
    #         prd_trn: Series = Series(model.predict(trnX), index=train.index)
    #         prd_tst: Series = Series(model.predict(tstX), index=test.index)
    #
    #         plot_forecasting_eval(train, test, prd_trn, prd_tst, title=grans[i])
    #         savefig(f"images/timeSeries/transformation/aggregation/{self.file_tag}_{self.target}_aggregation_lr_{grans[i]}.png")
    #         plot_forecasting_series(train, test, prd_tst, title=f"{self.file_tag} - Linear Regression", xlabel=self.index, ylabel=self.target,)
    #         savefig(f"images/timeSeries/transformation/aggregation/{self.file_tag}_{self.target}_aggregation_lr_{grans[i]}_series.png")

    # def ts_aggregation_lr(self):
    #     grans: list[str] = ["15T", "H", "D"]
    #     for i in range(len(grans)):
    #         lr_series = ts_aggregation_by(self.data, gran_level=grans[i], agg_func="sum")
    #
    #         print("Size of original series for gran:", grans[i], ":", len(lr_series))
    #
    #         train, test = series_train_test_split(lr_series, trn_pct=0.90)
    #
    #         print("Size of training set after splitting for gran:", grans[i], ":", len(train))
    #
    #         if len(train) == 0:
    #             print("Training set is empty after splitting. Check your data or splitting logic.")
    #             continue
    #
    #         trnX = arange(len(train)).reshape(-1, 1)
    #         trnY = train.to_numpy()
    #         # tstX = arange(len(train), len(lr_series)).reshape(-1, 1)
    #         #tstX = arange(len(train), len(lr_series) + len(test)).reshape(-1, 1)
    #         tstX = arange(len(train), len(train) + len(test)).reshape(-1, 1)
    #
    #         tstY = test.to_numpy()
    #
    #         if np.isnan(trnX).any() or np.isnan(trnY).any():
    #             print("NaN values found in the training set. Handle missing values before fitting the model.")
    #             continue
    #
    #         print("trnX:", trnX)
    #         print("trnY:", trnY)
    #         print("tstX:", tstX)
    #         print("tstY:", tstY)
    #
    #         model = LinearRegression()
    #         model.fit(trnX, trnY)
    #
    #         print("Number of points predicted:", len(model.predict(tstX)))
    #
    #         # Antes de criar prd_tst
    #         print("Length of tstX:", len(tstX))
    #         print("Length of test.index:", len(test.index))
    #
    #         prd_trn: Series = Series(model.predict(trnX), index=train.index)
    #         prd_tst: Series = Series(model.predict(tstX), index=test.index)
    #
    #         plot_forecasting_eval(train, test, prd_trn, prd_tst, title=grans[i])
    #         savefig(
    #             f"images/timeSeries/transformation/aggregation/{self.file_tag}_{self.target}_aggregation_lr_{grans[i]}.png")
    #         plot_forecasting_series(train, test, prd_tst, title=f"{self.file_tag} - Linear Regression",
    #                                 xlabel=self.index, ylabel=self.target, )
    #         savefig(
    #             f"images/timeSeries/transformation/aggregation/{self.file_tag}_{self.target}_aggregation_lr_{grans[i]}_series.png")

    def ts_aggregation_lr(self):
        grans: list[str] = ["15T", "H", "D"]
        for i in range(len(grans)):
            lr_series = ts_aggregation_by(self.data, gran_level=grans[i], agg_func="sum")
            train, test = series_train_test_split(lr_series, trn_pct=0.90)
            trnX = arange(len(train)).reshape(-1, 1)
            trnY = train.to_numpy()
            tstX = arange(len(train), len(lr_series)).reshape(-1, 1)
            tstY = test.to_numpy()
            model = LinearRegression()
            model.fit(trnX, trnY)

            prd_trn: Series = Series(model.predict(trnX), index=train.index)
            prd_tst: Series = Series(model.predict(tstX), index=test.index)

            plot_forecasting_eval(train, test, prd_trn, prd_tst, title=grans[i])
            savefig(
                f"images/timeSeries/transformation/aggregation/{self.file_tag}_{self.target}_aggregation_lr_{grans[i]}.png")
            show()
            plot_forecasting_series(train, test, prd_tst, title=f"{self.file_tag} - Linear Regression",
                                    xlabel=self.index, ylabel=self.target, )
            savefig(
                f"images/timeSeries/transformation/aggregation/{self.file_tag}_{self.target}_aggregation_lr_{grans[i]}_series.png")
            show()

    def ts_aggregation_df(self, level: str):
        return ts_aggregation_by(self.data, gran_level=level, agg_func="sum")
