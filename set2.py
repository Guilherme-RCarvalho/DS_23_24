'''
file:       set2.py
'''

# Imports
import os
import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame
from numpy import array, ndarray
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from auxiliar.dslabs_functions import dummify, evaluate_approach, mvi_by_dropping, mvi_by_filling, count_outliers, \
    plot_evaluation_results
from classification.KNN import KNN
from classification.NBayes import NBayes
from classification.decisionTrees import decisionTrees
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
print("Drop of ID, SSN and Name")
mv_out = MV_out("data/class_credit_score.csv", file_tag, data, 2)
data = mv_out.data_correction()
print("Data Cleaned")


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
# dim.missing_values()

# Variables Type
print('Set2 Data Types 2:\n', data.dtypes, '\n')

# Number of variables per type
# dim.num_variables_per_type()

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

# data['MonthlyBalance'] = data['MonthlyBalance'].replace(-3.33333E+26, "")

# Global Boxplot
# distribution.global_boxplot2()

# Single Variable Boxplots
# distribution.boxplots()

# Outliers Study
# distribution.num_standart_outliers_per_variable()

NR_STDEV: int = 2
IQR_FACTOR: float = 1.5
# distribution.num_outliers_per_variable(NR_STDEV, IQR_FACTOR)

# Histograms - numeric
# distribution.histogram_numeric()

# Histograms - symbolic
# distribution.histograms_symbolic_variables()

# Distribution for numeric
# distribution.histograms_numeric_with_distributions()

# Class Distribution
# distribution.class_distribution(target='Credit_Score')

# ----------------- #
# -- Granularity -- #
# ----------------- #

# Symbolic variables

granularity = Granularity(filename, file_tag, data)

#granularity.analyse_property_granularity("credit_history",["Credit_Score", "CreditMix" , "Payment_Behaviour"])

# -------------- #
# -- Sparsity -- #
# -------------- #

# Scatter-plots (all x all - including class)   -
# Correlation (all x all - including class)     -

sparsity = Sparsity(filename, file_tag, data)

# Scatter plots to project the data along two of its defining variables
# sparsity.analyse_property_sparsity("Credit_Score")

# encoded_data = sparsity.encode_credit()
# print('Encoded Data Shape: ', encoded_data.shape)
# sparsity.correlation_analysis(encoded_data)
# sparsity.correlation_analysis_num(data)


# ------------------------------------------ #
#             Data Preparation               #
# ------------------------------------------ #

# -------------- #
# -- Encoding -- #
# -------------- #
filename = "cleaned_data.csv"
encoding = Encoding(filename, file_tag, data)

def create_loan_type_columns():
    # Preencher os valores ausentes na coluna 'Type_of_Loan' com uma string vazia
    data['Type_of_Loan'] = data['Type_of_Loan'].fillna('').astype(str).str.replace('and ', '').str.replace(' ', '').str.split(',')

    # Adicionar uma coluna 'Without_Loan' e atribuir 1 aos clientes com NumofLoan igual a 0 e célula vazia em 'Type_of_Loan'
    data['Without_Loan'] = data.apply(lambda row: 1 if row['NumofLoan'] == 0 and all(pd.isna(item) or (isinstance(item, str) and item == '') for item in row['Type_of_Loan']) else 0, axis=1)

    existing_loan_types = list()

    # Iterar sobre os tipos de empréstimos e criar uma nova coluna para cada um
    for loan_list in data['Type_of_Loan']:
        for loan_type in loan_list:
            # Verificar se o novo valor já existe na lista
            if loan_type not in existing_loan_types:
                # Adicionar o novo valor à lista
                existing_loan_types.append(loan_type)

    # Iterar sobre os tipos únicos de empréstimos e criar colunas binárias
    for loan_type in existing_loan_types:
        if loan_type != '':
            data[loan_type] = data['Type_of_Loan'].apply(lambda x: 1 if loan_type in x else 0)

create_loan_type_columns()
data.drop('Type_of_Loan', axis=1, inplace=True)

encoded_data = encoding.encode_credit()
print("Encode done")
target = "Credit_Score"
encoded_data.to_csv("encoded.csv", index=False)
print("Encode to .csv")


# --------------------- #
# --------- MV -------- #
# --------------------- #
filename = "encoded.csv"
mv_out = MV_out(filename, file_tag, encoded_data, 2)
methodology = Methodology(filename, file_tag, encoded_data)

# # First approach
# print("Prof Filling")
# df1: DataFrame = mvi_by_filling(encoded_data, strategy="frequent")
# df1.to_csv("DF_1_PROF.csv", index=False)
#
# def count_missing_values(data):
#     missing_values = data.isnull().sum()
#     return missing_values
#
# # Chama a função para obter o número de valores ausentes em cada coluna
# resultado1 = count_missing_values(df1)
# print("Result MV PROF:\n", resultado1)
#
# # Divide os dados em conjuntos de treino e teste
# train_data, test_data = train_test_split(df1, test_size=0.3, random_state=42)
# methodology.evaluate_approach(train_data, test_data, 1, target=target)
#
# # Second approach
# print("Our filling")
# df2: DataFrame = mv_out.our_mv(encoded_data)
# df2.to_csv("DF_2_NOSSO.csv", index=False)
#
# resultado2 = count_missing_values(df2)
# print("Result MV Our:\n", resultado2)
#
# # Divide os dados em conjuntos de treino e teste
# train_data, test_data = train_test_split(df2, test_size=0.3, random_state=42)
# methodology.evaluate_approach(train_data, test_data, 2, target=target)

#best_approach = df2

# --------------------- #
# ----- Outliers ------ #
# --------------------- #

best_approach: DataFrame = read_csv("DF_2_NOSSO.csv", na_values="")
filename = "DF_2_NOSSO.csv"
dataOutliers = MV_out(filename, file_tag, best_approach, 2)

outliers1 = dataOutliers.median_outliers()
# outliers2 = dataOutliers.truncate_outliers()
# outliers3 = dataOutliers.median_outliers_by_customer()
# outliers4 = dataOutliers.truncate_outliers_by_customer()
# outliers5 = dataOutliers.drop_outliers()

# def corrigir_outliers(data, threshold=2):
#     data_corrigido = data.copy()
#
#     for customer_id in tqdm(data['Customer_ID'].unique(), desc="Processando clientes", unit="cliente"):
#         cliente_data = data_corrigido[data_corrigido['Customer_ID'] == customer_id]
#
#         for coluna in tqdm(cliente_data.select_dtypes(include=np.number).columns, desc="Processando colunas", unit="coluna", leave=False):
#             z_scores = (cliente_data[coluna] - cliente_data[coluna].mean()) / cliente_data[coluna].std()
#             outliers = abs(z_scores) > threshold
#             cliente_data.loc[outliers, coluna] = cliente_data[coluna].mean()
#
#         data_corrigido.loc[data_corrigido['Customer_ID'] == customer_id] = cliente_data
#
#     return data_corrigido

# # Aplicar a função para corrigir outliers para todas as colunas numéricas para todos os clientes
# df_corrigido = corrigir_outliers(best_approach)
#
# # Salvar o DataFrame corrigido em um arquivo CSV
# df_corrigido.to_csv('dados_corrigidos.csv', index=False)

# # Eval Outliers1
# train_data_1, test_data_1 = train_test_split(outliers1, test_size=0.3, random_state=42)
# eval1 = methodology.evaluate_approach(train_data_1, test_data_1, 3, target)

# print("EVAL 1:", eval1)
# # Eval Outliers2
# train_data_2, test_data_2 = train_test_split(outliers2, test_size=0.3, random_state=42)
# eval2 = methodology.evaluate_approach(train_data_2, test_data_2, 4, target)
#
# print("EVAL 2:", eval2)

# # Eval Outliers3
# train_data_3, test_data_3 = train_test_split(outliers3, test_size=0.3, random_state=42)
# eval3 = methodology.evaluate_approach(train_data_3, test_data_3, 5, target)
# print("EVAL 3:", eval3)
#
# # Eval Outliers4
# train_data_4, test_data_4 = train_test_split(outliers4, test_size=0.3, random_state=42)
# eval4 = methodology.evaluate_approach(train_data_4, test_data_4, 6, target)
# print("EVAL 4:", eval4)

# # Eval Outliers5
# train_data_5, test_data_5 = train_test_split(df_corrigido, test_size=0.3, random_state=42)
# eval5 = methodology.evaluate_approach(train_data_5, test_data_5, 7, target)
# print("EVAL 5:", eval5)

# melhor escolha -> outliers1


# ---------------------------- #
# -- Scaling Transformation -- #
# ---------------------------- #
dataScaling = Scaling(filename, file_tag, outliers1, target)

scaling1 = dataScaling.standard_scaler()
# scaling2 = dataScaling.min_max_scaler()
# dataScaling.scaler_plot(scaling1, scaling2)

# #E val Scaling1
# train_data_1, test_data_1 = train_test_split(scaling1, test_size=0.3, random_state=42)
# eval1 = methodology.evaluate_approach(train_data_1, test_data_1, 8, target)
# print(eval1)
#
# # Eval Scaling2
# train_data_2, test_data_2 = train_test_split(scaling2, test_size=0.3, random_state=42)
# eval2 = methodology.evaluate_approach(train_data_2, test_data_2, 9, target)
# print(eval2)

best_scaling = scaling1

# -------------------- #
# -- Data Balancing -- #
# -------------------- #
dataBalancing = Balancing(filename, file_tag, best_scaling, target)

# dataBalancing.check_balance()
# undersampling = dataBalancing.undersampling()
oversampling = dataBalancing.oversampling()
# smote = dataBalancing.SMOTE()

# # Eval undersampling
# train_data_1, test_data_1 = train_test_split(undersampling, test_size=0.3, random_state=42)
# eval1 = methodology.evaluate_approach(train_data_1, test_data_1, 10, target = "Credit_Score")
# print(eval1)

# # Eval oversampling
# train_data_2, test_data_2 = train_test_split(oversampling, test_size=0.3, random_state=42)
# eval2 = methodology.evaluate_approach(train_data_2, test_data_2, 11, target = "Credit_Score")
# print(eval2)

# # Eval smote
# train_data_3, test_data_3 = train_test_split(smote, test_size=0.3, random_state=42)
# eval3 = methodology.evaluate_approach(train_data_3, test_data_3, 12, target = "Credit_Score")
# print(eval3)


# {'accuracy': [0.7420541410425886, 0.8372894993965171], 'recall': [0.7224064966258721, 0.8107057074230813], 'precision': [0.753969201384744, 0.8576960309777347], 'auc': [0.7421528786271689, 0.8374230940072893], 'f1': [0.7378504672897197, 0.833539130946081]}
# {'accuracy': [0.7446364020468522, 0.9259893901694756], 'recall': [0.7182675613532917, 0.8908544882924312], 'precision': [0.7584481220889902, 0.9582576216434484], 'auc': [0.7446487870173533, 0.926005892401347], 'f1': [0.7378111970693851, 0.9233275782408871]}
# {'accuracy': [0.7463264635463124, 0.9200272287686024], 'recall': [0.7217264700356941, 0.8764324628968627], 'precision': [0.7589016741567485, 0.9600761434377734], 'auc': [0.74631607406055, 0.9200088170889756], 'f1': [0.7398473796971665, 0.9163495298190478]}

#oversample escolhido

# --------------------------------------- #
#             Classification              #
# --------------------------------------- #

classificationData = DataFrame = read_csv("data\credit_balance_oversample.csv", na_values="")
classificationData.drop(classificationData.columns[0], axis=1, inplace=True)
strategy: array = classificationData[target].to_list()
trainData, testData = train_test_split(classificationData, test_size=0.3, stratify=strategy)
trainData.to_csv(f"data/{file_tag}_train.csv", index=False)
testData.to_csv(f"data/{file_tag}_test.csv", index=False)

# -------------------- #
# ------ NBayes ------ #
# -------------------- #

# classificationNB = NBayes(file_tag,"data/credit_train.csv","data/credit_test.csv",target)
#
# # Chama a função naive_Bayes_study para obter o melhor modelo e parâmetros
# best_model, best_params = classificationNB.naive_Bayes_study()
#
# # Chama a função performance_analysis com os resultados obtidos
# classificationNB.performance_analysis(best_model, best_params)

# -------------------- #
# -------- KNN ------- #
# -------------------- #

classificationKNN = KNN(file_tag,"data/credit_train.csv","data/credit_test.csv",target)

# Chama a função full_metrics_KNN_study para obter o melhor modelo e parâmetros
best_model, best_params = classificationKNN.full_metrics_KNN_study()

# Chama a função performance_analysis com os resultados obtidos
classificationKNN.performance_analysis(best_model, best_params)

classificationKNN.overfitting_study(best_params, "accuracy")

# -------------------- #
# -------- DT -------- #
# -------------------- #

decisionTrees = decisionTrees(file_tag,"data/credit_train.csv","data/credit_test.csv",target)

best_model, best_params = decisionTrees.full_metrics_DT_study()

decisionTrees.performance_analysis(best_model, best_params)

# TODO resto decision trees
