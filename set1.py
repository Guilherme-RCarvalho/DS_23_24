'''
file:       set1.py
'''

#Imports
from pandas import Series, read_csv, DataFrame
from numpy import array, ndarray
from sklearn.model_selection import train_test_split
from auxiliar.dslabs_functions import evaluate_approach, mvi_by_dropping, mvi_by_filling, plot_forecasting_eval, plot_forecasting_series, plot_multibar_chart, series_train_test_split
from classification.NBayes import NBayes
from classification.decision_tree import DecisionTree
from classification.gradient_boosting import GradientBoosting
from classification.knn import Knn
from classification.mlp import MultiLayerPerceptron
from classification.random_forest import RandomForests
from preparation.balancing import Balancing
from preparation.missingValues import MissingValues
from preparation.scaling import Scaling
from preparation.outliers import Outliers
from profiling.dimensionality import Dimensionality
from profiling.sparsity import Sparsity
from profiling.granularity import Granularity
from profiling.distribution import Distribution
from matplotlib.pyplot import savefig, figure
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
 
# ---------------------------------------
#             DATASET 1  
# ---------------------------------------

# Load Data
filename = "data/class_pos_covid.csv"
file_tag = "pos_covid"
target   = "CovidPos" 
data: DataFrame = read_csv(filename, na_values="") 
NR_STDEV:   int   = 8

# ---------------------------------------
#             aux functions  
# ---------------------------------------
def evaluate(eval: dict, file_tag: str, measure: str):
    figure()
    plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} {measure} evaluation", percentage=True
    )
    savefig(f"images/{file_tag}_{measure}_eval.png")
    return
# ---------------------------------------
#             Profiling  
# ---------------------------------------

'''
# -------------------- #
# -- Dimensionality -- #
# -------------------- #
dim = Dimensionality(filename, file_tag, data)

print('Set1 Data Shape: ', data.shape) # Dimension: (380932, 40)
print()


# Number of records and variables
dim.num_records_variables()

# Missing Values
dim.missing_values()

# Variables Type
print('Set1 Data Types:\n', data.dtypes)
print()


# -> 'numeric': ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI'], 
# -> 'binary':  ['Sex', 'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 
# 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 
# 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 
# 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear', 'CovidPos'], 
# -> 'date': [], 
# -> 'symbolic': ['State', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage', 
# 'RaceEthnicityCategory', 'AgeCategory', 'TetanusLast10Tdap']


# Number of variables per type
dim.num_variables_per_type()
print()



# ----------------- #
# -- Granularity -- #
# ----------------- #
granularity = Granularity(filename, file_tag, data)
#
##Demographic granularity
granularity.analyse_property_granularity("demography", ["Sex", "RaceEthnicityCategory", "AgeCategory", "State"])

#Health Indicators #TODO change Days interval?
granularity.analyse_property_granularity("health_indicators", ["GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "LastCheckupTime"])

#TODO Sub-Levels of information? SleepHours, SmokerStatus+ECigars




# -------------- #
# -- Sparsity -- #
# -------------- #
sparsity = Sparsity(filename, file_tag, data)

# Scatter plots to project the data along two of its defining variables
#sparsity.analyse_property_sparsity("CovidPos")
#print()

# Correlation Analysis
#Variable encoding before?? Applied to ordinal / binary variables
encoded_data = sparsity.encode_pos_covid()
print(encoded_data)

print(data[''])

print('Encoded Data Shape: ', encoded_data.shape) #aumentou dummification acho
sparsity.correlation_analysis(encoded_data)


# ------------------ #
# -- Distribution -- #
# ------------------ #

#summary5: DataFrame = data.describe(include="all")
#print(summary5)

distribution = Distribution(filename, file_tag, data)

#print(distribution.get_numeric_variables())
#print(distribution.get_symbolic_variables())

# Global Boxplots
#distribution.global_boxplot()

# Single Variable Boxplots
#distribution.boxplots()

# Outliers Study
distribution.num_standart_outliers_per_variable() #2, 1.5

### default ainda
NR_STDEV:   int   = 8
IQR_FACTOR: float = 9 # a partir de 9 tudo a zero
distribution.num_outliers_per_variable(NR_STDEV, IQR_FACTOR) #testar varios valores??


# Histograms 
#print('histogram numeric')
distribution.histogram_numeric()
print('histogram numeric with distributions')
#distribution.helper_numeric_with_distributions()
#distribution.histograms_symbolic_variables()

# Class Distribution
distribution.class_distribution("CovidPos")

'''


# ---------------------------------------
#             Preparation  
# ---------------------------------------

'''
# -------------------- #
# ----- Encoding ----- #
# -------------------- #
sparsity = Sparsity(filename, file_tag, data)
encoded_data = sparsity.encode_pos_covid()


# -------------------- #
# -- Missing Values -- #
# -------------------- #
dataMissingValues = MissingValues(filename,file_tag,encoded_data)

#dataMissingValues.read_mv()


# FIRST APPROACH  - drop vars <0.93, records<0.8, fill mean  -> Lost 2 variables [TetanusLast10Tdap,PneumoVaxEver]
# Aplicar a funções
#encoded_data_MV1 = mvi_by_dropping(encoded_data, min_pct_per_variable=0.93, min_pct_per_record=0.9)
#encoded_data_MV1 = mvi_by_filling(encoded_data_MV1, strategy="frequent")
#print(f"mv1's remaining:{encoded_data_MV1.isna().sum()}")

# Second approach - 0 drop, fill knn (takes too long) -> frequent
encoded_data_MV2 = mvi_by_filling(encoded_data, strategy="frequent")
#print(f"mv2's remaining:{encoded_data_MV2.isna().sum()}")

#Eval MV1
#train_data_1, test_data_1 = train_test_split(encoded_data_MV1, test_size=0.3, random_state=42)
#evalMV1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
#print(evalMV1)
#evaluate(evalMV1, file_tag, "MV1")
#
##Eval MV2
#train_data_2, test_data_2 = train_test_split(encoded_data_MV2, test_size=0.3, random_state=42)
#evalMV2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
#print(evalMV2)
#evaluate(evalMV1, file_tag, "MV2")
#{'accuracy': [0.6963492522406497, 0.7009663486389641], 'recall': [0.14569616066865096, 0.08123628273353177], 'precision': [0.4369702434625789, 0.43059760956175297], 'auc': [0.5342476846864521, 0.5185298624350151], 'f1': [0.2185294581858357, 0.136685552407932]}
#{'accuracy': [0.6963073153657683, 0.7029838991949597], 'recall': [0.16870120866865598, 0.0797239051149893], 'precision': [0.4399119704472216, 0.43675693527080584], 'auc': [0.5404188138950101, 0.5188331373472066], 'f1': [0.24387799564270152, 0.13483547013993322]} 





# -------------------- #
# ----- Outliers ----- #
# -------------------- #

dataOutliers = Outliers(filename, file_tag, encoded_data_MV2, NR_STDEV)

#dataOutliers.drop_outliers()
outliers1 = dataOutliers.median_outliers()
outliers2 = dataOutliers.truncate_outliers()

#Eval Outliers1
train_data_1, test_data_1 = train_test_split(outliers1, test_size=0.3, random_state=42)
eval1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
print(eval1)
evaluate(eval1, file_tag, "Outliers_median")
#Eval Outliers2
train_data_2, test_data_2 = train_test_split(outliers2, test_size=0.3, random_state=42)
eval2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
print(eval2)
evaluate(eval1, file_tag, "Outliers_truncate")

{'accuracy': [0.6963073153657683, 0.7029488974448722], 'recall': [0.16870120866865598, 0.07966362238900443], 'precision': [0.4399119704472216, 0.4364266842800528], 'auc': [0.5404188138950101, 0.5187906659840907], 'f1': [0.24387799564270152, 0.13473351515305995]}
{'accuracy': [0.6963073153657683, 0.7029838991949597], 'recall': [0.16870120866865598, 0.0797239051149893], 'precision': [0.4399119704472216, 0.43675693527080584], 'auc': [0.5404188138950101, 0.5188331373472066], 'f1': [0.24387799564270152, 0.13483547013993322]}


# -------------------- #
# ----- Scaling ------ #
# -------------------- #
outliers = DataFrame = read_csv("data\pos_covid_truncate_outliers.csv",na_values="")
dataScaling = Scaling(filename, file_tag, outliers, target)


scaling1 = dataScaling.standard_scaler()
scaling2 = dataScaling.min_max_scaler()
#dataScaling.scaler_plot(scaling1,scaling2)

#Eval Scaling1
train_data_1, test_data_1 = train_test_split(scaling1, test_size=0.3, random_state=42)
eval1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
print(eval1)
evaluate(eval1, file_tag, "Scaling_standard")
#Eval Scaling2
train_data_2, test_data_2 = train_test_split(scaling2, test_size=0.3, random_state=42)
eval2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
print(eval2)
evaluate(eval2, file_tag, "Scaling_min-max")

{'accuracy': [0.7096867343367168, 0.7042527126356318], 'recall': [0.0, 0.06700424993218193], 'precision': [0.0, 0.43872113676731794], 'auc': [0.5, 0.5159688647907584], 'f1': [0.0, 0.1162535299654848]}
{'accuracy': [0.7074641232061603, 0.7047602380119006], 'recall': [0.07236941254483528, 0.06516562678964342], 'precision': [0.47488132911392406, 0.44239819930427665], 'auc': [0.5198166311087369, 0.5157831882268255], 'f1': [0.125598305129077, 0.11359815048339639]}

# -------------------- #
# ----  Balancing ---- #
# -------------------- #
#TODO RUN THIS ONLY ONE TIME TO HAVE NEW DATASET!
#scaling = DataFrame = read_csv("data\pos_covid_scaled_minmax.csv",na_values="")
#strategy: array = scaling[target].to_list()
#trainData, testData = train_test_split(scaling, test_size=0.3, random_state=42, stratify=strategy)
#trainData.to_csv(f"data/{file_tag}_train.csv", index=False)
#testData.to_csv(f"data/{file_tag}_test.csv", index=False)

trainData = DataFrame = read_csv("data\pos_covid_train.csv")
testData = DataFrame = read_csv("data\pos_covid_test.csv")
dataBalancing = Balancing(filename, file_tag, trainData, target)

#dataBalancing.check_balance()

undersampling = dataBalancing.undersampling()
#oversampling = dataBalancing.oversampling()
#smote = dataBalancing.SMOTE()
undersampling.to_csv(f"data/{file_tag}_train.csv", index=False)

#Eval undersampling
eval1 = evaluate_approach(undersampling, testData.copy(), target, "accuracy")
print(eval1)
evaluate(eval1, file_tag, "Balancing_undersampling")
#Eval oversampling
eval2 = evaluate_approach(oversampling, testData.copy(), target, "accuracy")
print(eval2)
evaluate(eval2, file_tag, "Balancing_oversampling")
#Eval smote
eval3 = evaluate_approach(smote, testData.copy(), target, "accuracy")
print(eval3)
evaluate(eval3, file_tag, "Balancing_SMOTE")

#{'accuracy': [0.5847917395869794, 0.6007000350017501], 'recall': [0.5995719926455074, 0.4569430629653073], 'precision': [0.3679819449839984, 0.35441262419637637], 'auc': [0.5891587692164815, 0.5582250547801889], 'f1': [0.4560607102735172, 0.3991994944175269]}
#{'accuracy': [0.5846692334616731, 0.6171946097304866], 'recall': [0.6004762335352805, 0.30732133707086234], 'precision': [0.3680331048752101, 0.3293068923196176], 'auc': [0.5893396296586554, 0.5256382772552072], 'f1': [0.45636138727264397, 0.3179344860381359]}
#{'accuracy': [0.5851855092754638, 0.5815453272663633], 'recall': [0.5952014950116045, 0.40503963589233505], 'precision': [0.36757752875916755, 0.32365125240847786], 'auc': [0.5881448704112435, 0.5293942862149122], 'f1': [0.4544816396045984, 0.359800259715919]}
'''
# ---------------------------------------
#             Classification  
# ---------------------------------------

#Buscar Dados (treino, teste e validação)    

#trainData, validationData = train_test_split(trainData, test_size=0.3, stratify=strategy)

'''

# -------------------- #
# ------ NBayes ------ #
# -------------------- #

classificationNB = NBayes(file_tag,"data/pos_covid_train.csv","data/pos_covid_test.csv",target)

classificationNB.full_metrics_NB_study()


# -------------------- #
# ------- KNN -------- #
# -------------------- #
K_MAX = 25
lag   = 2 # step size

classificationKnn = Knn(file_tag,"data/pos_covid_train.csv","data/pos_covid_test.csv",target)

# Testar Parâmetros:
# --- K = [1 ... 25] 
classificationKnn.full_metrics_knn_study(K_MAX, lag)

# -------------------- #
# -- Decision Trees -- #
# -------------------- #
D_MAX = 50
lag   = 2 

classificationDT = DecisionTree(file_tag,"data/pos_covid_train.csv","data/pos_covid_test.csv",target)

# Testar Parâmetros:
# --- D = [2, 4, 6, 8, 10] 
classificationDT.full_metrics_decision_tree_study(D_MAX, lag)



# -------------------- #
# ------- MLP -------- #
# -------------------- #
classificationMLP = MultiLayerPerceptron(file_tag,"data/pos_covid_train.csv","data/pos_covid_test.csv",target)

# Testar Parâmetros:
N_MAX_INTER = 2500 # default
lag = 100

classificationMLP.full_metrics_mlp_study(N_MAX_INTER, lag) # only accuracy


# -------------------- #
# - GradientBoosting - #
# -------------------- #
classificationGB = GradientBoosting(file_tag,"data/pos_covid_train.csv","data/pos_covid_test.csv",target)

# Testar Parâmetros:
NR_MAX_TREES_GB = 2500 # default
lag = 500

classificationGB.full_metrics_gradient_boosting_study(NR_MAX_TREES_GB, lag) # only accuracy


# -------------------- #
# --- RandomForest --- #
# -------------------- #
classificationRF = RandomForests(file_tag,"data/pos_covid_train.csv","data/pos_covid_test.csv",target)

# Testar Parâmetros:
NR_MAX_TREES_RF = 2500 # default
lag = 500

classificationRF.full_metrics_random_forests_study(NR_MAX_TREES_RF, lag) # only accuracy
'''
# ---------------------------------------#
#             Time-Series                #
# ---------------------------------------#


filename = "data/forecast_covid_single.csv"
file_tag = "covid"
target   = "deaths" 
index = "date"
data: DataFrame = read_csv(
    filename,
    index_col=index,
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

# ---------------------------------------
#             Profiling  
# ---------------------------------------
# -------------------- #
# -- Dimensionality -- #
# -------------------- #

#dim = ts_Dimensionality(filename, file_tag, data, target)
#dim.univariate_ts()

# -------------------- #
# --- Granularity ---- #
# -------------------- #

#granularity = ts_Granularity(filename, file_tag, data, target)
#granularity.ts_aggregation()

# -------------------- #
# --- Distribution --- #
# -------------------- #

#dist = ts_Distribtion(filename, file_tag, data, target, index)
#dist.ts_distribution_boxplots()
#dist.ts_distribution_histograms()
#dist.ts_autocorrelation_lag()
#dist.ts_autocorrelation_correlogram()

# -------------------- #
# --- Stationarity --- #
# -------------------- #

#stationarity = ts_Stationarity(filename, file_tag, data, target, index)
#stationarity.ts_components()
#stationarity.ts_stationarity_study()
#stationarity.eval_stationarity()
'''
ADF Statistic: -2.156
p-value: 0.223
Critical Values:
        1%: -3.464
        5%: -2.876
        10%: -2.575
The series is not stationary (p-value > 0.05 : the series is non-stationary)
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
aggregation.ts_aggregation()
aggregation.ts_aggregation_lr()

df = aggregation.ts_aggregation_df("W")

# -------------------- #
# ---- Smoothing ----- #
# -------------------- #

smoothing = ts_Smoothing(filename, file_tag, df, target, index)
smoothing.ts_smoothing()
smoothing.ts_smoothing_lr()

series = smoothing.ts_smoothing_size(50)
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
'''
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
'''
rollingMean = ts_Rollingmean(train, test, file_tag, target, index)
rollingMean.rolling_mean_study()
'''
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
'''