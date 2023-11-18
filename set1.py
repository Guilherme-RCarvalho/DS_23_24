'''
file:       set1.py
'''

#Imports
from pandas import read_csv, DataFrame
from profiling.dimensionality import Dimensionality
from profiling.sparsity import Sparsity
from profiling.granularity import Granularity

# ---------------------------------------
#             DATASET 1  
# ---------------------------------------

# Load Data
filename = "data/class_pos_covid.csv"
file_tag = "pos_covid"
data: DataFrame = read_csv(filename, na_values="") 



# -- Dimensionality --
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

# Number of variables per type
dim.num_variables_per_type()
print()



# -- Granularity --
granularity = Granularity(filename, file_tag, data)

#Demographic granularity
granularity.analyse_property_granularity("demography", ["Sex", "RaceEthnicityCategory", "AgeCategory", "State"])

#Health Indicators #TODO change Days interval?
granularity.analyse_property_granularity("health indicators", ["GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "LastCheckupTime"])

#TODO Sub-Levels of information? SleepHours, SmokerStatus+ECigars

# -- Sparsity -- 
sparsity = Sparsity(filename, file_tag, data)

sparsity.analyse_property_sparsity("CovidPos")
print()

# -- Distribution --

#summary5: DataFrame = data.describe(include="all")
#print(summary5)













