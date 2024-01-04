'''
file:       set4.py
'''


from pandas import read_csv, DataFrame, Series

from TSeries.profiling_ts import Profiling
from TSeries.transformation import Transformation

# Load Data
filename = "data/forecast_traffic_single.csv"
file_tag = "traffic"
target = "Total"

data: DataFrame = read_csv(
    filename,
    index_col = "Timestamp",
    sep = ",",
    decimal = ".",
    parse_dates = True,
    infer_datetime_format = True,
)
data.sort_index(inplace=True)
index = "Timestamp"
series: Series = data[target]

# ------------------------------- #
#           Profiling             #
# ------------------------------- #

profiling = Profiling(filename, file_tag, target, index, data, series)

# profiling.data_before()


# ----------------- #
# -- Granularity -- #
# ----------------- #

# profiling.aggregation_study()
#
# # First granularity (most atomic)
# # mais atomico possivel -> intervalo mais pequeno no meu conjunto de dados (15 min)
# profiling.plot_most_atomic_granularity()
#
# # Second granularity
# profiling.aggregation_by_d()
#
# # Third granularity
# profiling.aggregation_by_w()



# ------------------ #
# -- Distribution -- #
# ------------------ #


# boxplots
# profiling.number_summary()
# profiling.plot_boxplots_for_granularities()
#
# # histograms
# # profiling.variables_distribution()
# profiling.plot_histograms_for_granularities()

# ---------------------- #
# -- Auto-correlation -- #
# ---------------------- #

# # lag-plots
# profiling.lag_plots()
#
# # correlogram
# profiling.autocorrelation_study(10, 1)


# ---------------------- #
# -- Components study -- #
# ---------------------- #

# dias nao estão sorted, apenas horas

# profiling.plot_time_series_components()


# ------------------ #
# -- Stationarity -- #
# ------------------ #

# Stationarity study

# profiling.stationary_study()
#
# profiling.stationary_study_2()

# Stationarity Dickey-Fuller Test
# profiling.eval_stationarity()

# ADF Statistic: -9.927
# p-value: 0.000  ----->> estranho???????
# Critical Values:
# 	1%: -3.433
# 	5%: -2.863
# 	10%: -2.567
# The series is stationary

# ------------------------------- #
#        - Transformation -       #
#         - Aggregation -         #
# ------------------------------- #

'''
Evaluate the performance of Linear Regression models trained 
over each one of the three granularities considered.
'''

transformation = Transformation(filename, file_tag, target, index, data, series)

# transformation.plot_time_series_line_chart()
# transformation.plot_daily_aggregated_line_chart()

# # Forecasting plots after aggregations
# # transformation.forecasting_plots_after_aggregations()
# transformation.forecasting_plots_after_aggregations_real()
# #
# # # Forecasting results after aggregation
# transformation.forecasting_results_after_aggregation()
# transformation.forecasting_results_after_aggregation_real()

# Mean Squared Error for 15T granularity: 3681.418251377907
# Mean Squared Error for H granularity: 3167.045709090381
# Mean Squared Error for W granularity: 1.5496155228041761

print("vou agora->>>>>>>>>>")
transformation.ts_aggregation()
transformation.ts_aggregation_lr()

# Aggregation decision
# Third - W granularity

best_granularity="W"

# ------------------------------- #
#        - Transformation -       #
#          - Smoothing -          #
# ------------------------------- #

'''
Evaluate the performance of Linear Regression models trained over smoothed 
time series, considering different windows size, over the time series aggregated 
to the best aggregation found.
'''

# transformation.smoothing_different_winsizes()
# WIN_SIZE = 10
# transformation.specific_smoothing(WIN_SIZE)

# Forecasting plots after smoothing
# transformation.forecasting_plots_after_smoothing(best_granularity)

# Forecasting results after smoothing
# transformation.forecasting_results_after_smoothing()

# # Smoothing decision (best window size)
# transformation.smoothing_decision()
# transformation.ts_smoothing()
# transformation.ts_smoothing_lr()

best_window_size = 100
#comentario teste
# ------------------------------- #
#        - Transformation -       #
#       - Differentiation -       #
# ------------------------------- #

'''
Evaluate the performance of Linear Regression models trained over the first and 
second differentiation of the best time series found up to the moment.
'''

# transformation.differentiation()

# Forecasting plots after differentiation

# Forecasting results after differentiation

# Differentiation decision

# transformation.ts_differentiation()
# transformation.ts_differentiation_lr()

# transformation.example_usage()

# ---------------------------------------
#             Forecasting
# ---------------------------------------
train, test = series_train_test_split(dif1, trn_pct=0.90)
# -------------------- #
# -- Simple Average -- #
# -------------------- #

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Simple Average")
savefig(f"images/timeseries/forecasting/simpleAverage/{file_tag}_simpleAvg_eval.png")
plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Simple Average",
    xlabel=index,
    ylabel=target,
)
savefig(f"images/timeseries/forecasting/simpleAverage/{file_tag}_simpleAvg_forecast.png")
# -------------------- #
#  Persistence Model - #
# -------------------- #
#Optimist
fr_mod = PersistenceOptimistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Optimist")
savefig(f"images/timeseries/forecasting/persistanceModel/{file_tag}_persistence_optim_eval.png")
plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Persistence Optimist",
    xlabel=index,
    ylabel=target,
)
savefig(f"images/timeseries/forecasting/persistanceModel/{file_tag}_persistence_optim_forecast.png")

#Realist
fr_mod = PersistenceRealistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Realist")
savefig(f"images/timeseries/forecasting/persistanceModel/{file_tag}_persistence_real_eval.png")
plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Persistence Realist",
    xlabel=index,
    ylabel=target,
)
savefig(f"images/timeseries/forecasting/persistanceModel/{file_tag}_persistence_real_forecast.png")

# -------------------- #
# --- Rolling Mean --- #
# -------------------- #

rollingMean = ts_Rollingmean(train, test, file_tag, target, index)
rollingMean.rolling_mean_study()
# -------------------- #
# ------- ARIMA ------ #
# -------------------- #

arima = ts_Arima(train, test, file_tag, target, index)
arima.arima_study()

# -------------------- #
# ------- LSTMs ------ #
# -------------------- #

series = dif1[[target]].values.astype("float32")
series2 = dif1[[target]]
train_size = int(len(series) * 0.90)
train, test = series[:train_size], series[train_size:]


lstm = ts_LSTM(train, test, file_tag, target, index,series2,train_size)
lstm.lstm_study(3000)
