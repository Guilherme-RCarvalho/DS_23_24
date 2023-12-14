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

series: Series = data[target]

# ------------------------------- #
#           Profiling             #
# ------------------------------- #

profiling = Profiling(filename, file_tag, target, data, series)

profiling.data_before()

profiling.aggregation_by()

profiling.aggregation_study()

# ----------------- #
# -- Granularity -- #
# ----------------- #

# First granularity (most atomic)

# Second granularity

# Third granularity


# ------------------ #
# -- Distribution -- #
# ------------------ #

# boxplots

# histograms

# ---------------------- #
# -- Auto-correlation -- #
# ---------------------- #

# lag-plots

# correlogram



# ---------------------- #
# -- Components study -- #
# ---------------------- #




# ------------------ #
# -- Stationarity -- #
# ------------------ #

# Stationarity study

# Stationarity Dickey-Fuller Test



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
