'''
file:       set2.py
'''
import os

import numpy as np
import pandas as pd
# Imports
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split

from auxiliar.dslabs_functions import dummify, evaluate_approach, mvi_by_dropping, mvi_by_filling, count_outliers
from profiling.dimensionality import Dimensionality
from profiling.distribution import Distribution
from profiling.sparsity import Sparsity
from profiling.granularity import Granularity
from preparation.variables_encoding import Encoding
from preparation.mv_and_outliers import MV_out
from preparation.data_balancing import Balancing
from preparation.scaling_transf import Scaling
from preparation.methodology import Methodology


# ------------------------------------------------- #
#                     DATASET 2                     #
# ------------------------------------------------- #


# Load Data
filename = "data/class_credit_score.csv"
file_tag = "credit"
data: DataFrame = read_csv(filename, na_values="")
data = data.drop(['ID', 'SSN', 'Name'], axis=1)
print("Fiz agora o primeiro drop")
mv_out = MV_out("data/class_credit_score.csv", file_tag, data, 2)
data = mv_out.data_correction()
# Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
data['Occupation'] = data.groupby('Customer_ID')['Occupation'].fillna(method='ffill')

# Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
data['Occupation'] = data.groupby('Customer_ID')['Occupation'].fillna(method='bfill')



# ------------------------------------------ #
#                Data Profiling              #
# ------------------------------------------ #

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
dim.missing_values()

# Variables Type
print('Set2 Data Types 2:\n', data.dtypes, '\n')

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

# summary5: DataFrame = data.describe(include="all")
# print(summary5, '\n')
#
# distribution = Distribution(filename, file_tag, data)
#
# print('Numeric Variables: ', distribution.get_numeric_variables(), '\n')
# print('Symbolic Variables: ', distribution.get_symbolic_variables(), '\n')
#
# valor_monthly_balance = data.loc[5545, 'MonthlyBalance']
# print("Valor da coluna 'MonthlyBalance' para a linha 5545:", valor_monthly_balance)
#
# linha_com_valor_especifico = data[data['MonthlyBalance'] == -3.33333E+26]
#
# # Exibir a linha encontrada
# print("Linha com o valor específico na coluna 'MonthlyBalance':")
# print(linha_com_valor_especifico)

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
# distribution.histogram_numeric()

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

granularity = Granularity(filename, file_tag, data)

#granularity.analyse_property_granularity("credit_history",["Credit_Score", "CreditMix" , "Payment_Behaviour"])

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
#sparsity.correlation_analysis_num(data)


# ------------------------------------------ #
#             Data Preparation               #
# ------------------------------------------ #

# -------------- #
# -- Encoding -- #
# -------------- #
filename = "cleaned_data.csv"
encoding = Encoding(filename, file_tag, data)


# Cria uma lista em cada registo com os seus loan types
def create_loan_type_columns():
    print("vou fazer a correção da data")
    print("fiz a correção da data")
    # Separar os valores dentro de cada linha usando vírgulas e a palavra "and"
    # Preeenche os NA com string vazia
    data['Type_of_Loan'] = data['Type_of_Loan'].fillna('').str.replace('and ', '').str.replace(' ', '').str.split(',')

    existing_loan_types = list()

    # Iterar sobre os tipos de empréstimos e criar uma nova coluna para cada um
    for loan_list in data['Type_of_Loan']:
        for loan_type in loan_list:
            # Verificar se o novo valor já existe na lista
            if loan_type not in existing_loan_types:
                # Adicionar o novo valor à lista
                existing_loan_types.append(loan_type)

    for loan_type in existing_loan_types:
        if loan_type != '':
            data[loan_type] = data['Type_of_Loan'].apply(lambda x: 1 if loan_type in x else 0)

print("vou às colunas")
create_loan_type_columns()
print("sai das colunas")
data.drop('Type_of_Loan', axis=1, inplace=True)
print("type of loan apagado")

#encoded_data1 = encoding.create_loan_type_columns()
print("vou dar encode")
encoded_data = encoding.encode_credit()
print("encode feito")
target = "Credit_Score"
encoded_data.to_csv("encoded.csv", index=False)
print("encode to csv")


# --------------------- #
# --------- MV -------- #
# --------------------- #
filename = "encoded.csv"
mv_out = MV_out(filename, file_tag, encoded_data, 2)
methodology = Methodology(filename, file_tag, encoded_data)


#mv_out.read_mv()

# Primeira approach
print("EXEMPLO DA PROF")
df1: DataFrame = mvi_by_filling(encoded_data, strategy="frequent")
# Salvar o dataset modificado
df1.to_csv("DF_1.csv", index=False)

# Dividir os dados em conjuntos de treino e teste
train_data, test_data = train_test_split(df1, test_size=0.3, random_state=42)

methodology.evaluate_approach(train_data, test_data, 1, target=target)

# Segunda approach
print("NOSSO")
# df2: DataFrame = mv_out.our_mv(encoded_data)
# # Salvar o dataset modificado
# df2.to_csv("DF_2.csv", index=False)
#
# # Dividir os dados em conjuntos de treino e teste
# train_data, test_data = train_test_split(df2, test_size=0.3, random_state=42)
#
# methodology.evaluate_approach(train_data, test_data, 2, target=target)


#fill_values_id_based()

# --------------------- #
# ----- Outliers ------ #
# --------------------- #

dataOutliers = MV_out(filename, file_tag, df1, 2)

#dataOutliers.drop_outliers()

outliers1 = dataOutliers.median_outliers()
outliers2 = dataOutliers.truncate_outliers()

#Eval Outliers1
train_data_1, test_data_1 = train_test_split(outliers1, test_size=0.3, random_state=42)
eval1 = methodology.evaluate_approach(train_data_1, test_data_1, 3, target)
print(eval1)

#Eval Outliers2
train_data_2, test_data_2 = train_test_split(outliers2, test_size=0.3, random_state=42)
eval2 = methodology.evaluate_approach(train_data_2, test_data_2, 4, target)
print(eval2)


# ---------------------------- #
# -- Scaling Transformation -- #
# ---------------------------- #
dataScaling = Scaling(filename, file_tag, outliers1, target)


scaling1 = dataScaling.standard_scaler()
scaling2 = dataScaling.min_max_scaler()
dataScaling.scaler_plot(scaling1,scaling2)

#Eval Scaling1
train_data_1, test_data_1 = train_test_split(scaling1, test_size=0.3, random_state=42)
print("AAAAAAAAA " + str(test_data_1.dtypes))
print(test_data_1)
eval1 = methodology.evaluate_approach(train_data_1, test_data_1, 5, target)
print(eval1)

#Eval Scaling2
train_data_2, test_data_2 = train_test_split(scaling2, test_size=0.3, random_state=42)
eval2 = methodology.evaluate_approach(train_data_2, test_data_2, 6, target)
print(eval2)


# -------------------- #
# -- Data Balancing -- #
# -------------------- #
# dataBalancing = Balancing(filename, file_tag, best_scaling, target)
#
# dataBalancing.check_balance()
# undersampling = dataBalancing.undersampling()
# oversampling = dataBalancing.oversampling()
# smote = dataBalancing.SMOTE()
#
# #Eval undersampling
# train_data_1, test_data_1 = train_test_split(undersampling, test_size=0.3, random_state=42)
# eval1 = methodology.evaluate_approach(train_data_1, test_data_1, 7, target, "accuracy")
# print(eval1)
#
# #Eval oversampling
# train_data_2, test_data_2 = train_test_split(oversampling, test_size=0.3, random_state=42)
# eval2 = methodology.evaluate_approach(train_data_2, test_data_2, 8, target, "accuracy")
# print(eval2)
#
# #Eval smote
# train_data_3, test_data_3 = train_test_split(smote, test_size=0.3, random_state=42)
# eval3 = methodology.evaluate_approach(train_data_3, test_data_3, 9, target, "accuracy")
# print(eval3)



# ----------------- #
# -- Methodology -- #
# ----------------- #

methodology = Methodology(filename, file_tag, encoded_data)

# Dividir os dados em conjuntos de treino e teste
train_data, test_data = train_test_split(encoded_data, test_size=0.3, random_state=42)

#methodology.evaluate_approach(train_data, test_data, target=target)

# guardar os conjuntos de treino e teste em CSV
train_data.to_csv("credit_train.csv", index=False)
test_data.to_csv("credit_test.csv", index=False)
