from matplotlib.pyplot import figure, savefig, subplots
from statsmodels.tsa.arima.model import ARIMA
from auxiliar.dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_forecasting_eval, plot_forecasting_series, plot_multiline_chart, HEIGHT
from timeSeries.forecasting.rolingMean import RollingMeanRegressor
from pandas import Series

class ts_Arima:
    def __init__(self, train, test, filetag, target, index) -> None:
        self.train = train
        self.test = test
        self.file_tag = filetag
        self.target = target
        self.index = index
        
    def arima_study(self,  measure: str = "R2"):
        d_values = (0, 1, 2)
        p_params = (1, 2, 3, 5, 7, 10)
        q_params = (1, 3, 5, 7)

        flag = measure == "R2" or measure == "MAPE"
        best_model = None
        best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
        best_performance: float = -100000

        fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
        for i in range(len(d_values)):
            d: int = d_values[i]
            values = {}
            for q in q_params:
                yvalues = []
                for p in p_params:
                    arima = ARIMA(self.train, order=(p, d, q))
                    model = arima.fit()
                    prd_tst = model.forecast(steps=len(self.test), signal_only=False)
                    eval: float = FORECAST_MEASURES[measure](self.test, prd_tst)
                    print(f"ARIMA ({p}, {d}, {q})", eval)
                    if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                        best_performance: float = eval
                        best_params["params"] = (p, d, q)
                        best_model = model
                    yvalues.append(eval)
                values[q] = yvalues
            plot_multiline_chart(
                p_params, values, ax=axs[i], title=f"ARIMA d={d} ({measure})", xlabel="p", ylabel=measure, percentage=flag
            )
        print(
            f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
        )
        savefig(f"images/timeSeries/forecasting/arima/{self.file_tag}_arima_{measure}_study.png")
        params = best_params["params"]
        prd_trn = best_model.predict(start=0, end=len(self.train) - 1)
        prd_tst = best_model.forecast(steps=len(self.test))
        plot_forecasting_eval(
            self.train, self.test, prd_trn, prd_tst, title=f"{self.file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})"
        )
        savefig(f"images/timeSeries/forecasting/arima/{self.file_tag}_arima_{measure}_eval.png")
        plot_forecasting_series(
            self.train,
            self.test,
            prd_tst,
            title=f"{self.file_tag} - ARIMA ",
            xlabel=self.index,
            ylabel=self.target,
        )
        savefig(f"images/timeSeries/forecasting/arima/{self.file_tag}_arima_{measure}_forecast.png")
        return best_model, best_params
