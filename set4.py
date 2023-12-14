'''
file:       set4.py
'''


from pandas import read_csv, DataFrame, Series

from TSeries.profiling_ts import Profiling

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

# profiling.aggregation_by()
#
# profiling.aggregation_study()
#
# profiling.aggregation_by_w()


# First granularity (most atomic)

# Second granularity

# Third granularity


# ------------------ #
# -- Distribution -- #
# ------------------ #

# # boxplots
# profiling.number_summary()
#
# # histograms
# profiling.variables_distribution()

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

# # nao funciona
# profiling.plot_time_series_components()


# ------------------ #
# -- Stationarity -- #
# ------------------ #

# Stationarity study

# profiling.stationary_study()
#
# profiling.stationary_study_2()

# Stationarity Dickey-Fuller Test
profiling.eval_stationarity()

# ADF Statistic: -9.927
# p-value: 0.000
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

# Forecasting plots after aggregations

# Forecasting results after aggregation

# Aggregation decision


# ------------------------------- #
#        - Transformation -       #
#          - Smoothing -          #
# ------------------------------- #

'''
Evaluate the performance of Linear Regression models trained over smoothed 
time series, considering different windows size, over the time series aggregated 
to the best aggregation found.
'''

# Forecasting plots after smoothing

# Forecasting results after smoothing

# Smoothing decision (best window size)


# ------------------------------- #
#        - Transformation -       #
#       - Differentiation -       #
# ------------------------------- #

'''
Evaluate the performance of Linear Regression models trained over the first and 
second differentiation of the best time series found up to the moment.
'''

# Forecasting plots after differentiation

# Forecasting results after differentiation

# Differentiation decision
