from copy import deepcopy
from matplotlib.pyplot import figure, savefig, subplots
from pandas import Series, Index
from auxiliar.dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_forecasting_eval, plot_forecasting_series, plot_multiline_chart, HEIGHT
from timeSeries.forecasting.LSTMs import DS_LSTM, prepare_dataset_for_lstm

class ts_LSTM:
    def __init__(self, train, test, filetag, target, index, series2,t_size) -> None:
        self.train: Series = train
        self.test: Series = test
        self.file_tag = filetag
        self.target = target
        self.index = index
        self.series2 = series2
        self.t_size = t_size
    
    def lstm_study(self, nr_episodes: int = 1000, measure: str = "R2"):
        sequence_size = [2, 4, 8]
        nr_hidden_units = [25, 50, 100]
        #sequence_size = [2,4]
        #nr_hidden_units = [25,50]
        step: int = nr_episodes // 10
        episodes = [1] + list(range(0, nr_episodes + 1, step))[1:]
        flag = measure == "R2" or measure == "MAPE"
        best_model = None
        best_params: dict = {"name": "LSTM", "metric": measure, "params": ()}
        best_performance: float = -100000

        _, axs = subplots(1, len(sequence_size), figsize=(len(sequence_size) * HEIGHT, HEIGHT))

        for i in range(len(sequence_size)):
            length = sequence_size[i]
            tstX, tstY = prepare_dataset_for_lstm(self.test, seq_length=length)

            values = {}
            for hidden in nr_hidden_units:
                yvalues = []
                model = DS_LSTM(self.train, hidden_size=hidden)
                for n in range(0, nr_episodes + 1):
                    model.fit()
                    if n % step == 0:
                        prd_tst = model.predict(tstX)
                        eval: float = FORECAST_MEASURES[measure](self.test[length:], prd_tst)
                        print(f"seq length={length} hidden_units={hidden} nr_episodes={n}", eval)
                        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                            best_performance: float = eval
                            best_params["params"] = (length, hidden, n)
                            best_model = deepcopy(model)
                        yvalues.append(eval)
                values[hidden] = yvalues
            plot_multiline_chart(
                episodes,
                values,
                ax=axs[i],
                title=f"LSTM seq length={length} ({measure})",
                xlabel="nr episodes",
                ylabel=measure,
                percentage=flag,
            )
        print(best_params)
        print(
            f"LSTM best results achieved with length={best_params['params'][0]} hidden_units={best_params['params'][1]} and nr_episodes={best_params['params'][2]}) ==> measure={best_performance:.2f}"
        )
        savefig(f"images/timeSeries/forecasting/lstms/{self.file_tag}_lstms_{measure}_study.png")
        params = best_params["params"]
        best_length = params[0]
        trnX, trnY = prepare_dataset_for_lstm(self.train, seq_length=best_length)
        tstX, tstY = prepare_dataset_for_lstm(self.test, seq_length=best_length)
        
        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)
        
        plot_forecasting_eval(
            self.train[best_length:],
            self.test[best_length:],
            prd_trn,
            prd_tst,
            title=f"{self.file_tag} - LSTM (length={best_length}, hidden={params[1]}, epochs={params[2]})",
        )
        savefig(f"images/timeSeries/forecasting/lstms/{self.file_tag}_lstms_{measure}_eval.png")
        train, test = self.series2[:self.t_size], self.series2[self.t_size:]
        pred_series: Series = Series(prd_tst.numpy().ravel(), index=test.index[best_length:])
        plot_forecasting_series(
            train[best_length:],
            test[best_length:],
            pred_series,
            title=f"{self.file_tag} - LSTMs ",
            xlabel=self.index,
            ylabel=self.target,
        )
        savefig(f"images/timeSeries/forecasting/lstms/{self.file_tag}_lstms_{measure}_forecast.png")
        return best_model, best_params