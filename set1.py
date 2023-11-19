'''
file:       set1.py
'''

#Imports
from pandas import read_csv, DataFrame
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
data: DataFrame = read_csv(filename, na_values="") 



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

#Demographic granularity
granularity.analyse_property_granularity("demography", ["Sex", "RaceEthnicityCategory", "AgeCategory", "State"])

#Health Indicators #TODO change Days interval?
granularity.analyse_property_granularity("health indicators", ["GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "LastCheckupTime"])

#TODO Sub-Levels of information? SleepHours, SmokerStatus+ECigars




# -------------- #
# -- Sparsity -- #
# -------------- #
sparsity = Sparsity(filename, file_tag, data)

# Scatter plots to project the data along two of its defining variables
sparsity.analyse_property_sparsity("CovidPos")
print()

# Correlation Analysis
#Variable encoding before?? Applied to ordinal / binary variables
#sparsity.encode_pos_covid()
sparsity.correlation_analysis()



# ------------------ #
# -- Distribution -- #
# ------------------ #
# ainda ranhoso

summary5: DataFrame = data.describe(include="all")
print(summary5)

distribution = Distribution(filename, file_tag, data)

#print(distribution.get_numeric_variables())
#print(distribution.get_symbolic_variables())

# Global Boxplots
distribution.global_boxplot()

# Single Variable Boxplots
distribution.boxplots()

# Outliers Study
distribution.num_standart_outliers_per_variable()

### default ainda
NR_STDEV:   int   = 2
IQR_FACTOR: float = 1.5
distribution.num_outliers_per_variable(NR_STDEV, IQR_FACTOR) #testar varios valores??


# Histograms - nao esta a carregar need help
#distribution.histograms_numeric_with_distributions()
#distribution.histograms_symbolic_variables()

# Class Distribution
distribution.class_distribution("CovidPos")












