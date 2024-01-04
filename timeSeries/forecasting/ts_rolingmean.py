from matplotlib.pyplot import savefig
from auxiliar.dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_forecasting_eval, plot_forecasting_series, plot_line_chart
from timeSeries.forecasting.rolingMean import RollingMeanRegressor
from pandas import Series

class ts_Rollingmean:
    def __init__(self, train, test, filetag, target, index) -> None:
        self.train = train
        self.test = test
        self.file_tag = filetag
        self.target = target
        self.index = index
        
    def rolling_mean_study(self, measure: str = "R2"):
        win_size = (3, 5, 10, 15, 20, 25, 30, 40, 50)
        #win_size = (12, 24, 48, 96, 192, 384, 768)
        flag = measure == "MAPE"
        best_model = None
        best_params: dict = {"name": "Rolling Mean", "metric": measure, "params": ()}
        best_performance: float = -100000

        yvalues = []
        for w in win_size:
            pred = RollingMeanRegressor(win=w)
            pred.fit(self.train)
            prd_tst = pred.predict(self.test)

            eval: float = FORECAST_MEASURES[measure](self.test, prd_tst)
            # print(w, eval)
            if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                best_performance: float = eval
                best_params["params"] = (w,)
                best_model = pred
            yvalues.append(eval)

        print(f"Rolling Mean best with win={best_params['params'][0]:.0f} -> {measure}={best_performance}")
        print(yvalues)
        plot_line_chart(
            win_size, yvalues, title=f"Rolling Mean ({measure})", xlabel="window size", ylabel=measure, percentage=flag
        )
        savefig(f"images/timeSeries/forecasting/rollingMean/{self.file_tag}_{measure}_rollingMean.png")
        params = best_params["params"]
        prd_trn: Series = best_model.predict(self.train)
        prd_tst: Series = best_model.predict(self.test)

        plot_forecasting_eval(self.train, self.test, prd_trn, prd_tst, title=f"{self.file_tag} - Rolling Mean (win={params[0]})")
        savefig(f"images/timeSeries/forecasting/rollingMean/{self.file_tag}_rollingmean_{measure}_win{params[0]}_eval.png")
        plot_forecasting_series(
            self.train,
            self.test,
            prd_tst,
            title=f"{self.file_tag} - Rolling Mean (win={params[0]})",
            xlabel=self.index,
            ylabel=self.target,
        )
        savefig(f"images/timeSeries/forecasting/rollingMean/{self.file_tag}_rollingmean_{measure}_forecast.png")
        return best_model, best_params
