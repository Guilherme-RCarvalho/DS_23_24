'''
file:       set2.py
'''
import numpy as np
import pandas as pd
# Imports
from pandas import read_csv, DataFrame
from profiling.dimensionality import Dimensionality
from profiling.distribution import Distribution
from profiling.sparsity import Sparsity
from profiling.granularity import Granularity

# ------------------------------------------------- #
#                     DATASET 2                     #
# ------------------------------------------------- #

# Load Data
filename = "data/class_credit_score.csv"
file_tag = "credit"
data: DataFrame = read_csv(filename, na_values="")
# data = data.drop(['Annual_Income', 'TotalEMIpermonth'], axis=1)
data = data.drop(['ID', 'SSN'], axis=1)

# -------------------- #
# --  Data Cleaning -- #
# -------------------- #

# # Substituir valores não finitos (inf) por NaN
# data.replace([np.inf, -np.inf], np.nan, inplace=True)
# # # Substituir células NaN por um valor vazio
# # data = data.applymap(lambda x: "" if pd.notna(x) else x)
#
# Lista de colunas a serem convertidas para int
cols_to_convert_to_int = [
    'Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
    'NumofLoan', 'Delay_from_due_date', 'NumofDelayedPayment',
    'ChangedCreditLimit', 'NumCreditInquiries'
]

# cols_to_convert_to_str = [
#     'Name', 'Month', 'Occupation', 'Type_of_Loan', 'CreditMix', 'Credit_Score'
# ]
#
# Substituir "_" por nada (remover) nas colunas relevantes
data[cols_to_convert_to_int] = data[cols_to_convert_to_int].replace('_', '', regex=True)
#
#
# # # conversão das colunas para int
# # data[cols_to_convert_to_int] = data[cols_to_convert_to_int].astype(int)
#
# # # conversão das colunas para int
# # data[cols_to_convert_to_int] = pd.to_numeric(data[cols_to_convert_to_int], errors='raise')
#
# Converter para int apenas as células não vazias
data[cols_to_convert_to_int] = data[cols_to_convert_to_int].applymap(lambda x: pd.to_numeric(x, errors='coerce'))

# Limpar células na coluna 'Age' que estão abaixo de 0 ou acima de 100
data['Age'] = data['Age'].apply(lambda x: None if x < 0 or x > 100 else x)

# Lista de valores válidos para a coluna 'Payment_of_Min_Amount'
valid_values = ['Yes', 'No']

# Limpar células na coluna 'Payment_of_Min_Amount' que não são 'Yes' nem 'No'
data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].apply(lambda x: None if x not in valid_values else x)


# Função para converter "Years and Months" para meses
def years_and_months_to_months(years_and_months):
    if pd.isnull(years_and_months):
        return None

    # Dividir a string em anos e meses
    parts = years_and_months.split()

    # Verificar se a string tem o formato esperado
    if len(parts) == 5 and parts[1] == 'Years' and parts[4] == 'Months':
        # Converter anos e meses para meses
        years = int(parts[0])
        months = int(parts[3])
        result = years * 12 + months
        return result
    else:
        return None


# Aplicar a função à coluna "Credit_History_Age"
data['Credit_History_Age'] = data['Credit_History_Age'].apply(years_and_months_to_months)

# Renomear a coluna para "Credit_History_Age_in_Months"
data = data.rename(columns={'Credit_History_Age': 'Credit_History_Age_in_Months'})

# Limpar células na coluna 'NumofLoan' que estão abaixo de 0
data['NumofLoan'] = data['NumofLoan'].apply(lambda x: None if x < 0 else x)

# Limpar células na coluna 'ChangedCreditLimit' que estão abaixo de 0
data['ChangedCreditLimit'] = data['ChangedCreditLimit'].apply(lambda x: None if x < 0 else x)
#
# # conversão das colunas para string
# data[cols_to_convert_to_str] = data[cols_to_convert_to_str].astype(str)
#
# # Converter para string apenas as células não vazias
# data[cols_to_convert_to_str] = data[cols_to_convert_to_str].applymap(lambda x: str(x) if pd.notna(x) else x)
#
# Salvar o dataset modificado
data.to_csv("cleaned_data.csv", index=False)

# -------------------- #
# -- Dimensionality -- #
# -------------------- #

# Nr records x Nr variables - Done (images2submit/credit_records_variables.png)
# Nr Variable types         - Done (images2submit/credit_variable_types.png)
# Nr missing values         - Done (images2submit/credit_mv.png)

dim = Dimensionality(filename, file_tag, data)

print('Set2 Data Shape: ', data.shape)  # Dimension: (100000, 27)
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
#dim.num_records_variables()

# Missing Values
#dim.missing_values()

# Variables Type
print('Set2 Data Types:\n', data.dtypes, '\n')

# Number of variables per type
#dim.num_variables_per_type()

# ------------------ #
# -- Distribution -- #
# ------------------ #

# Global boxplot                        - Juntar as 3
# Boxplots for individual numeric vars  - Done (images2submit/credit_single_boxplots.png)
# Outliers                              - Juntar os 2
# Histograms for numeric                - Done
# Distributions for numeric             -
# Histograms for symbolic               - Done
# Class Distribution                    - Done (images2submit/credit_class_distribution.png)

summary5: DataFrame = data.describe(include="all")
print(summary5, '\n')

distribution = Distribution(filename, file_tag, data)

print('Numeric Variables: ', distribution.get_numeric_variables(), '\n')
print('Symbolic Variables: ', distribution.get_symbolic_variables(), '\n')

valor_monthly_balance = data.loc[5545, 'MonthlyBalance']
print("Valor da coluna 'MonthlyBalance' para a linha 5545:", valor_monthly_balance)

linha_com_valor_especifico = data[data['MonthlyBalance'] == -3.33333E+26]

# Exibir a linha encontrada
print("Linha com o valor específico na coluna 'MonthlyBalance':")
print(linha_com_valor_especifico)

# data['MonthlyBalance'] = data['MonthlyBalance'].replace(-3.33333E+26, "")

# Global Boxplot
#distribution.global_boxplot2()

# Single Variable Boxplots
#distribution.boxplots()

# Outliers Study
#distribution.num_standart_outliers_per_variable()

NR_STDEV: int = 2
IQR_FACTOR: float = 1.5
#distribution.num_outliers_per_variable(NR_STDEV, IQR_FACTOR)

# Histograms - numeric
distribution.histogram_numeric()

# Histograms - symbolic
#distribution.histograms_symbolic_variables()

# Distribution for numeric
#distribution.histograms_numeric_with_distributions()


# Class Distribution
#distribution.class_distribution(target='Credit_Score')

# ----------------- #
# -- Granularity -- #
# ----------------- #

# Symbolic variables  -

#granularity = Granularity(filename, file_tag, data)

# -------------- #
# -- Sparsity -- #
# -------------- #

# Scatter-plots (all x all - including class)   -
# Correlation (all x all - including class)     -

sparsity = Sparsity(filename, file_tag, data)

# Scatter plots to project the data along two of its defining variables
#sparsity.analyse_property_sparsity("Credit_Score")
print()

# print(data["Payment_Behaviour"].unique())
#
# encoded_data = sparsity.encode_credit()
# print(encoded_data)
# print('Encoded Data Shape: ', encoded_data.shape)
#sparsity.correlation_analysis(encoded_data)
sparsity.correlation_analysis_num(data)
