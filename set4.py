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

transformation.plot_time_series_line_chart()
transformation.plot_daily_aggregated_line_chart()


# Forecasting plots after aggregations
transformation.forecasting_plots_after_aggregations()

# Forecasting results after aggregation
transformation.forecasting_results_after_aggregation()

# Mean Squared Error for 15T granularity: 3681.418251377907
# Mean Squared Error for H granularity: 3167.045709090381
# Mean Squared Error for W granularity: 1.5496155228041761

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

transformation.smoothing_different_winsizes()
WIN_SIZE = 10
transformation.specific_smoothing(WIN_SIZE)

# Forecasting plots after smoothing
# transformation.forecasting_plots_after_smoothing(best_granularity)

# Forecasting results after smoothing
# transformation.forecasting_results_after_smoothing()
#
# # Smoothing decision (best window size)
# transformation.smoothing_decision()


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
