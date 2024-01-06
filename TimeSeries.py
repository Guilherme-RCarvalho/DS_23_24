'''
file:       set4.py
'''


from pandas import read_csv, DataFrame, Series


from numpy import array, ndarray

from auxiliar.dslabs_functions import evaluate_approach, mvi_by_dropping, mvi_by_filling, plot_forecasting_eval, plot_forecasting_series, plot_multibar_chart, series_train_test_split
from matplotlib.pyplot import savefig, figure
from TSeries.profiling_ts import Profiling
from TSeries.transformation import Transformation
from timeSeries.forecasting.ts_arima import ts_Arima
from timeSeries.forecasting.ts_lstms import ts_LSTM
from timeSeries.forecasting.ts_persistance_optimist import PersistenceOptimistRegressor
from timeSeries.forecasting.ts_persistance_realist import PersistenceRealistRegressor
from timeSeries.forecasting.ts_rolingmean import ts_Rollingmean
from timeSeries.forecasting.ts_simpleaverage import SimpleAvgRegressor
from timeSeries.profiling.ts_dimensionality import ts_Dimensionality
from timeSeries.profiling.ts_distribution import ts_Distribtion
from timeSeries.profiling.ts_granularity import ts_Granularity
from timeSeries.profiling.ts_stationarity import ts_Stationarity
from timeSeries.transformation.ts_aggregation import ts_Aggregation
from timeSeries.transformation.ts_differentiation import ts_Differentiation
from timeSeries.transformation.ts_smoothing import ts_Smoothing
from timeSeries.transformation.ts_scaling import ts_Scaling



# Load Data
filename = "data/forecast_traffic_single.csv"
file_tag = "traffic"
target = "Total"
index = "Timestamp"

data: DataFrame = read_csv(
    filename,
    index_col = "Timestamp",
    sep = ",",
    decimal = ".",
    parse_dates = True,
    infer_datetime_format = True,
)
data.sort_index(inplace=True)
series: Series = data[target]

# ---------------------------------------
#             Profiling
# ---------------------------------------
# -------------------- #
# -- Dimensionality -- #
# -------------------- #

# dim = ts_Dimensionality(filename, file_tag, data, target)
# dim.univariate_ts()

# -------------------- #
# --- Granularity ---- #
# -------------------- #

# granularity = ts_Granularity(filename, file_tag, data, target)
# granularity.ts_aggregation()

# -------------------- #
# --- Distribution --- #
# -------------------- #

dist = ts_Distribtion(filename, file_tag, data, target, index)
# dist.ts_distribution_boxplots()
# dist.ts_distribution_histograms()
dist.ts_autocorrelation_lag()
dist.ts_autocorrelation_correlogram()

# -------------------- #
# --- Stationarity --- #
# -------------------- #

# stationarity = ts_Stationarity(filename, file_tag, data, target, index)
# stationarity.ts_components()
# stationarity.ts_stationarity_study()
# stationarity.eval_stationarity()

# ADF Statistic: -2.156
# p-value: 0.223
# Critical Values:
#         1%: -3.464
#         5%: -2.876
#         10%: -2.575
# The series is not stationary (p-value > 0.05 : the series is non-stationary)
'''
# ---------------------------------------
#             Transformation
# ---------------------------------------

# -------------------- #
# ----- Scaling ------ #
# -------------------- #

scaling = ts_Scaling(filename, file_tag, data, target)
scaling.ts_scale()

df = scaling.scale_all_dataframe(data)

# -------------------- #
# --- Aggregation ---- #
# -------------------- #

aggregation = ts_Aggregation(filename, file_tag, df, target, index)
# aggregation.ts_aggregation()
# aggregation.ts_aggregation_lr()

df = aggregation.ts_aggregation_df("H")

# -------------------- #
# ---- Smoothing ----- #
# -------------------- #

smoothing = ts_Smoothing(filename, file_tag, df, target, index)
smoothing.ts_smoothing()
smoothing.ts_smoothing_lr()

series = smoothing.ts_smoothing_size(100)
#size 50?

# -------------------- #
# - Differentiation -- #
# -------------------- #

differentiation = ts_Differentiation(filename, file_tag, df, target, index,series)
dif1,dif2 = differentiation.ts_differentiation()
differentiation.ts_differentiation_lr()

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

# 50
# {'name': 'LSTM', 'metric': 'R2', 'params': (4, 25, 3000)}
# LSTM best results achieved with length=4 hidden_units=25 and nr_episodes=3000) ==> measure=0.67

# 100
# {'name': 'LSTM', 'metric': 'R2', 'params': (4, 100, 2100)}
# LSTM best results achieved with length=4 hidden_units=100 and nr_episodes=2100) ==> measure=0.85
'''
