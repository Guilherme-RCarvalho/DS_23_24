'''
file:       profilling_ts.py
'''
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, show, savefig, subplots, legend
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt


from auxiliar.dslabs_functions import HEIGHT, plot_line_chart, ts_aggregation_by, plot_ts_multivariate_chart, \
    set_chart_labels, plot_forecasting_eval


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
            plot_line_chart(X_test.flatten(), y_test.flatten(), label="Actual")
            plot_line_chart(X_test.flatten(), y_pred.flatten(), label="Predicted")
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

    def forecasting_results_after_aggregation(self):
        granularities = ["15T", "H", "W"]

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
            savefig(f"images/TS/Transformation/{self.file_tag}_forecasting_plots_smoothing_size_{size}.png")
            show()

            # Evaluate and plot the Linear Regression
            trnX = np.arange(len(X_train)).reshape(-1, 1)
            tstX = np.arange(len(X_train), len(X)).reshape(-1, 1)

            prd_trn = Series(model.predict(trnX).flatten(), index=ss_smoothed.index[:len(trnX)])
            prd_tst = Series(model.predict(tstX).flatten(), index=ss_smoothed.index[len(trnX):])

            plot_forecasting_eval(ss_smoothed.iloc[:len(trnX)], ss_smoothed.iloc[len(trnX):], prd_trn, prd_tst,
                                  title=f"{self.file_tag} - Linear Regression Evaluation - Smoothing Size {size}")
            savefig(f"images/TS/Transformation/{self.file_tag}_linear_regression_evaluation_smoothing_size_{size}.png")

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
            f"images/TS/Transformation/{self.file_tag}_forecasting_results_after_smoothing_best_winsize.png")
        show()

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
