'''
file:       set1.py
'''

#Imports
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from auxiliar.dslabs_functions import evaluate_approach, mvi_by_dropping, mvi_by_filling
from preparation.balancing import Balancing
from preparation.missingValues import MissingValues
from preparation.scaling import Scaling
from profiling.dimensionality import Dimensionality
from profiling.sparsity import Sparsity
from profiling.granularity import Granularity
from profiling.distribution import Distribution

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


# -------------------- #
# ----- Encoding ----- #
# -------------------- #
sparsity = Sparsity(filename, file_tag, data)
encoded_data = sparsity.encode_pos_covid()


# -------------------- #
# -- Missing Values -- #
# -------------------- #
dataMissingValues = MissingValues(filename,file_tag,encoded_data)

dataMissingValues.read_mv()


# FIRST APPROACH  - drop vars <0.93, records<0.8, fill mean  -> Lost 2 variables [TetanusLast10Tdap,PneumoVaxEver]
# Aplicar a funções
encoded_data_MV1 = mvi_by_dropping(encoded_data, min_pct_per_variable=0.93, min_pct_per_record=0.9)
encoded_data_MV1 = mvi_by_filling(encoded_data_MV1, strategy="frequent")
#print(f"mv1's remaining:{encoded_data_MV1.isna().sum()}")

# Second approach - 0 drop, fill knn
encoded_data_MV2 = mvi_by_filling(encoded_data, strategy="knn")
#print(f"mv2's remaining:{encoded_data_MV2.isna().sum()}")

#Eval MV1
train_data_1, test_data_1 = train_test_split(encoded_data_MV1, test_size=0.3, random_state=42)
evalMV1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
print(evalMV1)

#Eval MV2
train_data_2, test_data_2 = train_test_split(encoded_data_MV2, test_size=0.3, random_state=42)
evalMV2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
print(evalMV2)


# Verificar contagem de NaNs após o drop
#print("NaN counts after dropping:")
#print(encoded_data_MV1.isna().sum())


# SECOND APPROACH  



# -------------------- #
# ----- Outliers ----- #
# -------------------- #
'''
dataOutliers = Outliers(filename, file_tag, best_encoded, NR_STDEV)

dataOutliers.drop_outliers()
outliers1 = dataOutliers.median_outliers()
outliers2 = dataOutliers.truncate_outliers()

#Eval Outliers1
train_data_1, test_data_1 = train_test_split(outliers1, test_size=0.3, random_state=42)
eval1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
print(eval1)

#Eval Outliers2
train_data_2, test_data_2 = train_test_split(outliers2, test_size=0.3, random_state=42)
eval2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
print(eval2)

# -------------------- #
# ----- Scaling ------ #
# -------------------- #
dataScaling = Scaling(filename, file_tag, best_outliers)


scaling1 = dataScaling.standard_scaler(target)
scaling2 = dataScaling.min_max_scaler(target)


#Eval Scaling1
train_data_1, test_data_1 = train_test_split(scaling1, test_size=0.3, random_state=42)
eval1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
print(eval1)

#Eval Scaling2
train_data_2, test_data_2 = train_test_split(scaling2, test_size=0.3, random_state=42)
eval2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
print(eval2)
# -------------------- #
# ----  Balancing ---- #
# -------------------- #

dataBalancing = Balancing(filename, file_tag, best_scaling, target)

dataBalancing.check_balance()
undersampling = dataBalancing.undersampling()
oversampling = dataBalancing.oversampling()
smote = dataBalancing.SMOTE()

#Eval undersampling
train_data_1, test_data_1 = train_test_split(undersampling, test_size=0.3, random_state=42)
eval1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
print(eval1)

#Eval oversampling
train_data_2, test_data_2 = train_test_split(oversampling, test_size=0.3, random_state=42)
eval2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
print(eval2)

#Eval smote
train_data_3, test_data_3 = train_test_split(smote, test_size=0.3, random_state=42)
eval3 = evaluate_approach(train_data_3, test_data_3, target, "accuracy")
print(eval3)


'''