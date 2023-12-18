'''
file:       profilling_ts.py
'''
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, show, savefig, subplots, legend, subplot
from numpy import arange
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt


from auxiliar.dslabs_functions import HEIGHT, plot_line_chart, ts_aggregation_by, plot_ts_multivariate_chart, \
    set_chart_labels, plot_forecasting_eval, series_train_test_split, plot_forecasting_series


class Transformation:

    def __init__(self, filename, file_tag, target, index, data, series):
        self.target = target
        self.filename = filename
        self.file_tag = file_tag
        self.index = index
        self.data = data
        self.series = series

    def plot_time_series_line_chart(self):
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            self.series.index.to_list(),
            self.series.to_list(),
            xlabel=self.series.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} hourly {self.target}",
        )
        savefig(f"images/TS/Transformation/{self.file_tag}_ts_line_chart.png")
        show()

    # ------------------- #
    #   - Aggregation -   #
    # ------------------- #

    def plot_daily_aggregated_line_chart(self):
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        ss_agg: Series = ts_aggregation_by(self.series, gran_level="D", agg_func="sum")
        plot_line_chart(
            ss_agg.index.to_list(),
            ss_agg.to_list(),
            xlabel=ss_agg.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} daily {self.target}",
        )
        savefig(f"images/TS/Transformation/Aggregation/{self.file_tag}_daily_aggregated_line_chart.png")
        show()

    def forecasting_plots_after_aggregations(self):
        granularities = ["15T", "H", "W"]

        for gran in granularities:
            ss: Series = ts_aggregation_by(self.series, gran)

            # Prepare the data for training
            X = np.arange(len(ss)).reshape(-1, 1)
            y = ss.values.reshape(-1, 1)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train the Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Plot predictions
            figure(figsize=(3 * HEIGHT, HEIGHT / 2))
            plot_line_chart(X_test.flatten(), y_test.flatten(), label="Actual", color="blue")
            plot_line_chart(X_test.flatten(), y_pred.flatten(), label="Predicted", color="orange")
            set_chart_labels(title=f"{self.file_tag} Linear Regression - {gran} granularity", xlabel="Time",
                             ylabel=self.target)
            legend()
            savefig(f"images/TS/Transformation/Aggregation/{self.file_tag}_forecasting_plots_{gran}.png")
            show()

            # Evaluate and plot the Linear Regression
            trnX = np.arange(len(X_train)).reshape(-1, 1)
            tstX = np.arange(len(X_train), len(X)).reshape(-1, 1)

            prd_trn = Series(model.predict(trnX).flatten(), index=ss.index[:len(trnX)])
            prd_tst = Series(model.predict(tstX).flatten(), index=ss.index[len(trnX):])

            plot_forecasting_eval(ss.iloc[:len(trnX)], ss.iloc[len(trnX):], prd_trn, prd_tst,
                                  title=f"{self.file_tag} - Linear Regression Evaluation")
            savefig(f"images/TS/Transformation/Aggregation/{self.file_tag}_linear_regression_evaluation_{gran}.png")

    def forecasting_plots_after_aggregations_real(self):
        grans: list[str] = ["15T", "H", "D", "W"]
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
            # Plot predictions
            figure(figsize=(3 * HEIGHT, HEIGHT / 2))
            plot_line_chart(tstX.flatten(), tstY.flatten(), label="Actual", color="blue")
            plot_line_chart(tstX.flatten(), prd_tst.to_numpy().flatten(), label="Predicted", color="orange")

            set_chart_labels(title=f"{self.file_tag} Linear Regression - {grans[i]} granularity", xlabel="Time",
                             ylabel=self.target)
            legend()
            savefig(f"images/TS/Transformation/Aggregation/{self.file_tag}_forecasting_plots_{grans[i]}.png")
            show()

            # Evaluate and plot the Linear Regression
            prd_trn = Series(model.predict(trnX).flatten(), index=train.index)
            prd_tst = Series(model.predict(tstX).flatten(), index=test.index)

            plot_forecasting_eval(lr_series.iloc[:len(train)], lr_series.iloc[len(train):], prd_trn, prd_tst,
                                  title=f"{self.file_tag} - Linear Regression Evaluation")
            savefig(f"images/TS/Transformation/Aggregation/{self.file_tag}_linear_regression_evaluation_{grans[i]}.png")

    def forecasting_results_after_aggregation_real(self):
        # Define the model outside the loop
        model = LinearRegression()

        grans: list[str] = ["15T", "H", "D", "W"]
        for gran in grans:
            lr_series = ts_aggregation_by(self.data, gran_level=gran, agg_func="sum")
            train, test = series_train_test_split(lr_series, trn_pct=0.90)
            trnX = arange(len(train)).reshape(-1, 1)
            trnY = train.to_numpy()
            tstX = arange(len(train), len(lr_series)).reshape(-1, 1)
            tstY = test.to_numpy()
            model.fit(trnX, trnY)

            prd_trn: Series = Series(model.predict(trnX), index=train.index)
            prd_tst: Series = Series(model.predict(tstX), index=test.index)

            # Evaluate the model's performance
            mse = mean_squared_error(tstY, prd_tst)
            print(f"Mean Squared Error for {gran} granularity: {mse}")

        # Plot Forecasting results after aggregation
        ss_aggregated: Series = ts_aggregation_by(self.data, gran_level=grans[-1],
                                                  agg_func="sum")  # Using the last granularity
        X_aggregated = arange(len(ss_aggregated)).reshape(-1, 1)
        y_aggregated = ss_aggregated.to_numpy().reshape(-1, 1)

        y_pred_aggregated = model.predict(X_aggregated)  # Assuming you want to use the last trained model

        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            X_aggregated.flatten(),
            y_aggregated.flatten(),
            xlabel=ss_aggregated.index.name,
            ylabel=self.target,
            title=f"Set 2 - Forecasting results after aggregation ({grans[-1]})",
            label="Actual",
            color="blue"
        )
        plot_line_chart(
            X_aggregated.flatten(),
            y_pred_aggregated.flatten(),
            xlabel=ss_aggregated.index.name,
            ylabel=self.target,
            title=f"Set 2 - Forecasting results after aggregation ({grans[-1]})",
            label="Predicted",
            color="orange"
        )
        savefig(
            f"images/TS/Transformation/Aggregation/{self.file_tag}_forecasting_results_after_aggregation_{grans[-1]}.png")
        show()

    def forecasting_results_after_aggregation(self):
        granularities = ["15T", "H", "D", "W"]

        # Define the model outside the loop
        model = LinearRegression()

        for gran in granularities:
            ss: Series = ts_aggregation_by(self.series, gran)

            # Prepare the data for training
            X = np.arange(len(ss)).reshape(-1, 1)
            y = ss.values.reshape(-1, 1)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the Linear Regression model
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Evaluate the model's performance
            mse = mean_squared_error(y_test, y_pred)
            print(f"Mean Squared Error for {gran} granularity: {mse}")

        # Plot Forecasting results after aggregation
        ss_aggregated: Series = ts_aggregation_by(self.series, granularities[-1])  # Using the last granularity
        X_aggregated = np.arange(len(ss_aggregated)).reshape(-1, 1)
        y_aggregated = ss_aggregated.values.reshape(-1, 1)

        y_pred_aggregated = model.predict(X_aggregated)  # Assuming you want to use the last trained model

        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            X_aggregated.flatten(),
            y_aggregated.flatten(),
            xlabel=ss_aggregated.index.name,
            ylabel=self.target,
            title=f"Set 2 - Forecasting results after aggregation ({granularities[-1]})",
            name="Actual",
        )
        plot_line_chart(
            X_aggregated.flatten(),
            y_pred_aggregated.flatten(),
            xlabel=ss_aggregated.index.name,
            ylabel=self.target,
            title=f"Set 2 - Forecasting results after aggregation ({granularities[-1]})",
            name="Predicted",
        )
        savefig(
            f"images/TS/Transformation/Aggregation/{self.file_tag}_forecasting_results_after_aggregation_{granularities[-1]}.png")
        show()

    def ts_aggregation(self):
        figure(figsize=(3 * HEIGHT, 3 * HEIGHT / 2))

        # Weekly aggregation
        ax1 = subplot(3, 1, 1)
        ss: Series = ts_aggregation_by(self.series, gran_level="15T", agg_func="sum")
        plot_line_chart(
            ss.index.to_list(),
            ss.to_list(),
            xlabel=ss.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} 15 min {self.target}",
            ax=ax1
        )

        # Monthly aggregation
        ax2 = subplot(3, 1, 2)
        ss_agg_monthly: Series = ts_aggregation_by(self.series, gran_level="D", agg_func="sum")
        plot_line_chart(
            ss_agg_monthly.index.to_list(),
            ss_agg_monthly.to_list(),
            xlabel=ss_agg_monthly.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} daily {self.target}",
            ax=ax2
        )

        # Yearly aggregation
        ax3 = subplot(3, 1, 3)
        ss_agg_yearly: Series = ts_aggregation_by(self.series, gran_level="W", agg_func="sum")
        plot_line_chart(
            ss_agg_yearly.index.to_list(),
            ss_agg_yearly.to_list(),
            xlabel=ss_agg_yearly.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} weekly {self.target}",
            ax=ax3
        )
        savefig(f"images/TS/Transformation/Aggregation/{self.file_tag}_{self.target}_aggregation.png")
        show()

    def ts_aggregation_lr(self):
        grans: list[str] = ["15T", "H", "W"]
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
                f"images/TS/Transformation/Aggregation/{self.file_tag}_{self.target}_aggregation_lr_{grans[i]}.png")
            show()
            plot_forecasting_series(train, test, prd_tst, title=f"{self.file_tag} - Linear Regression",
                                    xlabel=self.index, ylabel=self.target, )
            savefig(
                f"images/TS/Transformation/Aggregation/{self.file_tag}_{self.target}_aggregation_lr_{grans[i]}_series.png")
            show()

    # ------------------- #
    #    - Smoothing -    #
    # ------------------- #

    def smoothing_different_winsizes(self):
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
        savefig(f"images/TS/Transformation/Smoothing/{self.file_tag}_winsizes_smoothing.png")
        show()

    def specific_smoothing(self, WIN_SIZE):
        ss_smooth: Series = self.series.rolling(window=WIN_SIZE).mean()

        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            ss_smooth.index.to_list(),
            ss_smooth.to_list(),
            xlabel=ss_smooth.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} {self.target} after smoothing (size={WIN_SIZE})",
        )
        savefig(f"images/TS/Transformation/Smoothing/{self.file_tag}_specific_smoothing.png")
        show()

    def forecasting_plots_after_smoothing(self, best_granularity):
        sizes = [25, 50, 75, 100]

        # Use the best aggregation granularity
        ss: Series = ts_aggregation_by(self.series, best_granularity)

        for size in sizes:
            # Apply smoothing
            ss_smoothed = ss.rolling(window=size).mean()

            # Prepare the data for training
            X = np.arange(len(ss_smoothed)).reshape(-1, 1)
            y = ss_smoothed.values.reshape(-1, 1)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train the Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Plot predictions
            figure(figsize=(3 * HEIGHT, HEIGHT / 2))
            plot_line_chart(X_test.flatten(), y_test.flatten(), label="Actual")
            plot_line_chart(X_test.flatten(), y_pred.flatten(), label="Predicted")
            set_chart_labels(title=f"{self.file_tag} Linear Regression - Smoothing Size {size}", xlabel="Time",
                             ylabel=self.target)
            legend()
            savefig(f"images/TS/Transformation/Smoothing/{self.file_tag}_forecasting_plots_smoothing_size_{size}.png")
            show()

            # Evaluate and plot the Linear Regression
            trnX = np.arange(len(X_train)).reshape(-1, 1)
            tstX = np.arange(len(X_train), len(X)).reshape(-1, 1)

            prd_trn = Series(model.predict(trnX).flatten(), index=ss_smoothed.index[:len(trnX)])
            prd_tst = Series(model.predict(tstX).flatten(), index=ss_smoothed.index[len(trnX):])

            plot_forecasting_eval(ss_smoothed.iloc[:len(trnX)], ss_smoothed.iloc[len(trnX):], prd_trn, prd_tst,
                                  title=f"{self.file_tag} - Linear Regression Evaluation - Smoothing Size {size}")
            savefig(f"images/TS/Transformation/Smoothing/{self.file_tag}_linear_regression_evaluation_smoothing_size_{size}.png")

    def forecasting_results_after_smoothing(self, best_window_size, best_smoothed_series):
        # Prepare os dados para treinamento
        X_smoothed = np.arange(len(best_smoothed_series)).reshape(-1, 1)
        y_smoothed = best_smoothed_series.values.reshape(-1, 1)

        # Divida os dados em conjuntos de treinamento e teste
        X_train_smoothed, X_test_smoothed, y_train_smoothed, y_test_smoothed = train_test_split(
            X_smoothed, y_smoothed, test_size=0.3, random_state=42)

        # Treine o modelo de Regressão Linear
        model_smoothed = LinearRegression()
        model_smoothed.fit(X_train_smoothed, y_train_smoothed)

        # Faça previsões no conjunto de teste
        y_pred_smoothed = model_smoothed.predict(X_test_smoothed)

        # Avalie o desempenho do modelo
        mse_smoothed = mean_squared_error(y_test_smoothed, y_pred_smoothed)
        print(f"Mean Squared Error for Smoothed Data (Best Window Size {best_window_size}): {mse_smoothed}")

        # Plot Forecasting results after smoothing
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            X_test_smoothed.flatten(),
            y_test_smoothed.flatten(),
            xlabel=best_smoothed_series.index.name,
            ylabel=self.target,
            title=f"Set 2 - Forecasting results after smoothing (Best Window Size {best_window_size})",
            name="Actual",
        )
        plot_line_chart(
            X_test_smoothed.flatten(),
            y_pred_smoothed.flatten(),
            xlabel=best_smoothed_series.index.name,
            ylabel=self.target,
            title=f"Set 2 - Forecasting results after smoothing (Best Window Size {best_window_size})",
            name="Predicted",
        )
        savefig(
            f"images/TS/Transformation/Smoothing/{self.file_tag}_forecasting_results_after_smoothing_best_winsize.png")
        show()


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
        savefig(f"images/TS/Transformation/Smoothing/{self.file_tag}_{self.target}_smoothing.png")

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
            savefig(f"images/TS/Transformation/Smoothing/{self.file_tag}_{self.target}_smoothing_lr_{sizes[i]}.png")
            plot_forecasting_series(train, test, prd_tst, title=f"{self.file_tag} - Linear Regression", xlabel=self.index, ylabel=self.target,)
            savefig(f"images/TS/Transformation/Smoothing/{self.file_tag}_{self.target}_smoothing_lr_{sizes[i]}_series.png")

    def smoothing_decision(self):
        best_window_size, _ = self.smoothing_different_winsizes()
        print(f"Smoothing Decision: Best Window Size is {best_window_size}")




    def differentiation(self):
        ss_diff: Series = self.series.diff()
        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_line_chart(
            ss_diff.index.to_list(),
            ss_diff.to_list(),
            title="Differentiation",
            xlabel=self.series.index.name,
            ylabel=self.target,
        )
        show()

    # --------------------- #
    #  - differentiation -  #
    # --------------------- #

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
        savefig(f"images/TS/Transformation/Differentiation/{self.file_tag}_{self.target}_differentiation.png")

    def ts_differentiation_lr(self):
        diffs: list[Series] = [self.data.diff(), self.data.diff().diff()]
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
            savefig(
                f"images/TS/Transformation/Differentiation/{self.file_tag}_{self.target}_differentiation_lr_diff{i + 1}.png")
            plot_forecasting_series(train, test, prd_tst, title=f"{self.file_tag} - Linear Regression",
                                    xlabel=self.index, ylabel=self.target, )
            savefig(
                f"images/TS/Transformation/Differentiation/{self.file_tag}_{self.target}_differentiation_lr_diff{i + 1}_series.png")

    def get_differentiated_series(self):
        ss_diff_1: Series = self.series.diff()
        ss_diff_2: Series = ss_diff_1.diff()
        return ss_diff_1, ss_diff_2

    # Função para visualizar a série temporal original e diferenciada
    def visualize_series_and_diff(self, diff_1, diff_2):
        fig, axs = subplots(3, 1, figsize=(3 * HEIGHT, HEIGHT * 3))
        fig.suptitle(f"{self.file_tag} {self.target} - Original and Differentiated Series")

        # Plot Original Series
        plot_line_chart(
            self.series.index.to_list(),
            self.series.to_list(),
            ax=axs[0],
            xlabel=self.series.index.name,
            ylabel=self.target,
            title="Original Series",
        )

        # Plot First Derivative
        plot_line_chart(
            diff_1.index.to_list(),
            diff_1.to_list(),
            ax=axs[1],
            xlabel=diff_1.index.name,
            ylabel=self.target,
            title="First Derivative",
        )

        # Plot Second Derivative
        plot_line_chart(
            diff_2.index.to_list(),
            diff_2.to_list(),
            ax=axs[2],
            xlabel=diff_2.index.name,
            ylabel=self.target,
            title="Second Derivative",
        )

        plt.tight_layout()
        savefig(f"images/TS/Transformation/Differentiation/{self.file_tag}_{self.target}_series_and_diff.png")
        show()

    # Função para comparar o desempenho dos modelos de regressão linear
    def compare_lr_performance(results_dict):
        for key, value in results_dict.items():
            print(f"Performance for differentiation {key}:")
            print(f"Mean Squared Error: {value['mse']}")
            print()

    def example_usage(self):

        # Chame a função de diferenciação
        self.ts_differentiation()

        # Obtenha as séries diferenciadas
        ss_diff_1, ss_diff_2 = self.get_differentiated_series()

        # Chame a função para visualizar as séries
        self.visualize_series_and_diff(ss_diff_1, ss_diff_2)

        # Aqui você pode calcular o MSE para cada série diferenciada e
        # armazenar os resultados em um dicionário chamado lr_results.
        lr_results = {}

        # Para a primeira diferenciação
        train_1, test_1 = series_train_test_split(Series(ss_diff_1).dropna(), trn_pct=0.90)

        trnX_1 = arange(len(train_1)).reshape(-1, 1)
        trnY_1 = train_1.to_numpy()
        tstX_1 = arange(len(train_1), len(ss_diff_1.dropna())).reshape(-1, 1)
        tstY_1 = test_1.to_numpy()

        model_1 = LinearRegression()
        model_1.fit(trnX_1, trnY_1)
        y_pred_1 = model_1.predict(tstX_1)

        mse_1 = mean_squared_error(tstY_1, y_pred_1)
        lr_results["First Derivative"] = {"mse": mse_1}

        # Para a segunda diferenciação
        train_2, test_2 = series_train_test_split(Series(ss_diff_2).dropna(), trn_pct=0.90)
        trnX_2 = arange(len(train_2)).reshape(-1, 1)
        trnY_2 = train_2.to_numpy()
        tstX_2 = arange(len(train_2), len(ss_diff_2.dropna())).reshape(-1, 1)
        tstY_2 = test_2.to_numpy()

        model_2 = LinearRegression()
        model_2.fit(trnX_2, trnY_2)
        y_pred_2 = model_2.predict(tstX_2)

        mse_2 = mean_squared_error(tstY_2, y_pred_2)
        lr_results["Second Derivative"] = {"mse": mse_2}

        # Chame a função para comparar o desempenho dos modelos de regressão linear
        self.compare_lr_performance(lr_results)
