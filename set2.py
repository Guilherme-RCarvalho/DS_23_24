'''
file:       set2.py
'''

#Imports
from pandas import read_csv, DataFrame
from profiling.dimensionality import Dimensionality
from profiling.distribution import Distribution
from profiling.sparsity import Sparsity
from profiling.granularity import Granularity

# --------------------------------------- #
#                DATASET 2                #
# --------------------------------------- #

# Load Data
filename = "data/class_credit_score.csv"
file_tag = "credit"
data: DataFrame = read_csv(filename, na_values="", index_col="ID")

# -- Dimensionality --
dim = Dimensionality(filename, file_tag, data)

print('Set2 Data Shape: ', data.shape) # Dimension: (100000, 27)
print('Set2 Columns Names: ', data.columns)

# Index(['Customer_ID', 'Month', 'Name', 'Age', 'SSN', 'Occupation',
#        'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
#        'Num_Credit_Card', 'Interest_Rate', 'NumofLoan', 'Type_of_Loan',
#        'Delay_from_due_date', 'NumofDelayedPayment', 'ChangedCreditLimit',
#        'NumCreditInquiries', 'CreditMix', 'OutstandingDebt',
#        'CreditUtilizationRatio', 'Credit_History_Age', 'Payment_of_Min_Amount',
#        'TotalEMIpermonth', 'Amountinvestedmonthly', 'Payment_Behaviour',
#        'MonthlyBalance', 'Credit_Score'],
#       dtype='object')

# Number of records and variables
dim.num_records_variables()

# Missing Values
dim.missing_values()

# Variables Type
print('Set2 Data Types:\n', data.dtypes, '\n')

# Number of variables per type
dim.num_variables_per_type()

# -- Distribution --

summary5: DataFrame = data.describe(include="all")
print(summary5, '\n')

distribution = Distribution(filename, file_tag, data)

print('Numeric Variables: ', distribution.get_numeric_variables(), '\n')
print('Symbolic Variables: ', distribution.get_symbolic_variables(), '\n')

# Global Boxplots
#distribution.global_boxplot()

# Single Variable Boxplots
distribution.boxplots()

# Outliers Study
distribution.num_standart_outliers_per_variable()

# -- Granularity --
granularity = Granularity(filename, file_tag, data)

# -- Sparsity -- 
sparsity = Sparsity(filename, file_tag, data)

