'''
file:       set2.py
'''

# Imports
import os

import joblib
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
from classification.gradient_boosting import GradientBoosting
from classification.mlp import MultiLayerPerceptron
from classification.random_forest import RandomForests
from preparation import methodology
from preparation.methodology import Methodology
from profiling.dimensionality import Dimensionality
from profiling.distribution import Distribution
from profiling.sparsity import Sparsity
from profiling.granularity import Granularity
from preparation.variables_encoding import Encoding
from preparation.mv_and_outliers import MV_out
from preparation.data_balancing import Balancing
from preparation.scaling_transf import Scaling


# ------------------------------------------------- #
#                     DATASET 2                     #
# ------------------------------------------------- #

# Load Data
filename = "data/class_credit_score.csv"
file_tag = "credit"
target = "Credit_Score"
data: DataFrame = read_csv(filename, na_values="")




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
#
# print('Set2 Data Shape: ', data.shape)  # Dimension: (100000, 28)
# print('Set2 Columns Names: ', data.columns)

# Index(['ID', 'Customer_ID', 'Month', 'Name', 'Age', 'SSN', 'Occupation',
#        'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
#        'Num_Credit_Card', 'Interest_Rate', 'NumofLoan', 'Type_of_Loan',
#        'Delay_from_due_date', 'NumofDelayedPayment', 'ChangedCreditLimit',
#        'NumCreditInquiries', 'CreditMix', 'OutstandingDebt',
#        'CreditUtilizationRatio', 'Credit_History_Age', 'Payment_of_Min_Amount',
#        'TotalEMIpermonth', 'Amountinvestedmonthly', 'Payment_Behaviour',
#        'MonthlyBalance', 'Credit_Score'],
#       dtype='object')

# # Number of records and variables
# dim.num_records_variables()

# # Missing Values
# dim.missing_values()

# Variables Type
print('Set2 Data Types 2:\n', data.dtypes, '\n')

# Set2 Data Types 2:
# ID                         object
# Customer_ID                object
# Month                      object
# Name                       object
# Age                        object
# SSN                        object
# Occupation                 object
# Annual_Income             float64
# Monthly_Inhand_Salary     float64
# Num_Bank_Accounts           int64
# Num_Credit_Card             int64
# Interest_Rate               int64
# NumofLoan                   int64
# Type_of_Loan               object
# Delay_from_due_date         int64
# NumofDelayedPayment       float64
# ChangedCreditLimit        float64
# NumCreditInquiries        float64
# CreditMix                  object
# OutstandingDebt           float64
# CreditUtilizationRatio    float64
# Credit_History_Age         object
# Payment_of_Min_Amount      object
# TotalEMIpermonth          float64
# Amountinvestedmonthly     float64
# Payment_Behaviour          object
# MonthlyBalance            float64
# Credit_Score               object
# dtype: object


# # Number of variables per type
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

# data = data.drop(['ID', 'SSN', 'Name'], axis=1)
# #data = data.drop(['Annual_Income', 'TotalEMIpermonth'], axis=1)
# print("Drop of ID, SSN and Name")
# mv_out = MV_out("data/class_credit_score.csv", file_tag, data, 2)
# data = mv_out.data_correction()
data['MonthlyBalance'] = data['MonthlyBalance'].replace(-3.33333E+26, )


summary5: DataFrame = data.describe(include="all")
print(summary5, '\n')



distribution = Distribution(filename, file_tag, data)

# print('Numeric Variables: ', distribution.get_numeric_variables(), '\n')
# print('Symbolic Variables: ', distribution.get_symbolic_variables(), '\n')


# # Global Boxplot
# distribution.global_boxplot2()
#
# # Single Variable Boxplots
# distribution.boxplots()
#
# # Outliers Study
# distribution.num_standart_outliers_per_variable()

NR_STDEV: int = 2
IQR_FACTOR: float = 2.5
# distribution.num_outliers_per_variable(NR_STDEV, IQR_FACTOR)

# # Histograms - numeric
# distribution.histogram_numeric()

# Histograms - symbolic
# distribution.histograms_symbolic_variables()

# # Distribution for numeric
# distribution.histograms_numeric_with_distributions()

# # Class Distribution
# distribution.class_distribution(target='Credit_Score')

# ----------------- #
# -- Granularity -- #
# ----------------- #

# Symbolic variables

granularity = Granularity(filename, file_tag, data)

# granularity.analyse_property_granularity("credit_history",["Credit_Score", "CreditMix", "Payment_Behaviour"])

# -------------- #
# -- Sparsity -- #
# -------------- #

# Scatter-plots (all x all - including class)   -
# Correlation (all x all - including class)     -

data = data.drop(['ID', 'SSN', 'Name'], axis=1)
print("Drop of ID, SSN and Name")
mv_out = MV_out("data/class_credit_score.csv", file_tag, data, 2)
data = mv_out.data_correction()
print("Data Cleaned")
# #

sparsity = Sparsity(filename, file_tag, data)

# Scatter plots to project the data along two of its defining variables
# sparsity.analyse_property_sparsity("Credit_Score")
#
# encoded_data = sparsity.encode_credit()
# print('Encoded Data Shape: ', encoded_data.shape)
# sparsity.correlation_analysis(encoded_data)



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

sparsity.correlation_analysis(encoded_data)
# sparsity.analyse_property_sparsity("Credit_Score")

# --------------------- #
# --------- MV -------- #
# --------------------- #
# filename = "encoded.csv"
# mv_out = MV_out(filename, file_tag, encoded_data, 2)
# methodology = Methodology(filename, file_tag, encoded_data)

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
# methodology.evaluate_approach(train_data, test_data, "mvi_by_filling_first_approach", "Prof MV filling", target=target)
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
# methodology.evaluate_approach(train_data, test_data, "our_mv_second_approach", "Our MV filling", target=target)

# best_approach = df2

# --------------------- #
# ----- Outliers ------ #
# --------------------- #

# best_approach: DataFrame = read_csv("DF_2_NOSSO.csv", na_values="")
# filename = "DF_2_NOSSO.csv"
# dataOutliers = MV_out(filename, file_tag, best_approach, 2)
#
# outliers1 = dataOutliers.median_outliers()
# outliers2 = dataOutliers.truncate_outliers()
# # outliers3 = dataOutliers.median_outliers_by_customer()
# # outliers4 = dataOutliers.truncate_outliers_by_customer()
# outliers5 = dataOutliers.drop_outliers()
#
# # Eval Outliers1
# train_data_1, test_data_1 = train_test_split(outliers1, test_size=0.3, random_state=42)
# eval1 = methodology.evaluate_approach(train_data_1, test_data_1, "median_outliers1", "Median Outliers", target)
# print("EVAL 1:", eval1)
#
# # Eval Outliers2
# train_data_2, test_data_2 = train_test_split(outliers2, test_size=0.3, random_state=42)
# eval2 = methodology.evaluate_approach(train_data_2, test_data_2, "truncate_outliers2", "Truncate Outliers", target)
# print("EVAL 2:", eval2)

# # Eval Outliers3
# train_data_3, test_data_3 = train_test_split(outliers3, test_size=0.3, random_state=42)
# eval3 = methodology.evaluate_approach(train_data_3, test_data_3, "median_outliers3_by_customer", target)
# print("EVAL 3:", eval3)
#
# # Eval Outliers4
# train_data_4, test_data_4 = train_test_split(outliers4, test_size=0.3, random_state=42)
# eval4 = methodology.evaluate_approach(train_data_4, test_data_4, "truncate_outliers4_by_customer", target)
# print("EVAL 4:", eval4)

# # Eval Outliers5
# train_data_5, test_data_5 = train_test_split(outliers5, test_size=0.3, random_state=42)
# eval5 = methodology.evaluate_approach(train_data_5, test_data_5, "drop_outliers5", "Drop Outliers", target)
# print("EVAL 5:", eval5)

# melhor escolha -> outliers1


# ---------------------------- #
# -- Scaling Transformation -- #
# ---------------------------- #
# dataScaling = Scaling(filename, file_tag, outliers1, target)
#
# scaling1 = dataScaling.standard_scaler()
# scaling2 = dataScaling.min_max_scaler()
# dataScaling.scaler_plot(scaling1, scaling2)

# # Eval Scaling1
# train_data_1, test_data_1 = train_test_split(scaling1, test_size=0.3, random_state=42)
# eval1 = methodology.evaluate_approach(train_data_1, test_data_1, "standard_scaler", "Standard Scaler", target)
# print(eval1)
#
# # Eval Scaling2
# train_data_2, test_data_2 = train_test_split(scaling2, test_size=0.3, random_state=42)
# eval2 = methodology.evaluate_approach(train_data_2, test_data_2, "min_max_scaler", "Min Max Scaler", target)
# print(eval2)

# best_scaling = scaling1

# {'accuracy': [0.7556, 0.8563], 'recall': [0.7568766218447747, 0.8990799716914366], 'precision': [0.8804127106086383, 0.8976823063877897], 'auc': [0.7547017975776968, 0.8262009739206756], 'f1': [0.8139841688654353, 0.8983805954317233]}
# {'accuracy': [0.7429333333333333, 0.8567], 'recall': [0.798490209955178, 0.8978060863411181], 'precision': [0.831033636140437, 0.8992061241848597], 'auc': [0.7038447642620864, 0.8277786820121263], 'f1': [0.8144369586140519, 0.8985055598838444]}

# -------------------- #
# -- Data Balancing -- #
# -------------------- #
# dataBalancing = Balancing(filename, file_tag, best_scaling, target)

# strategy: array = best_scaling[target].to_list()
# trainData, testData = train_test_split(best_scaling, test_size=0.3, random_state=42, stratify=strategy)
# trainData.to_csv(f"data/{file_tag}_train.csv", index=False)
# testData.to_csv(f"data/{file_tag}_test.csv", index=False)
#
# dataBalancing = Balancing(filename, file_tag, trainData, target)

# dataBalancing.check_balance()
# undersampling = dataBalancing.undersampling()
# oversampling = dataBalancing.oversampling()
# smote = dataBalancing.SMOTE()


# # Eval under sampling
# train_data_1, test_data_1 = train_test_split(undersampling, test_size=0.3, random_state=42)
# eval1 = methodology.evaluate_approach(train_data_1, test_data_1, "under_sampling", "Undersampling", target)
# print(eval1)
#
# # Eval oversampling
# train_data_2, test_data_2 = train_test_split(oversampling, test_size=0.3, random_state=42)
# eval2 = methodology.evaluate_approach(train_data_2, test_data_2, "oversampling", "Oversampling", target)
# print(eval2)
#
# # Eval smote
# train_data_3, test_data_3 = train_test_split(smote, test_size=0.3, random_state=42)
# eval3 = methodology.evaluate_approach(train_data_3, test_data_3, "SMOTE", "SMOTE", target)
# print(eval3)


# {'accuracy': [0.7463875205254515, 0.8052545155993431], 'recall': [0.7292177037399967, 0.7731504164625184], 'precision': [0.7572930800542741, 0.8280566730802869], 'auc': [0.7464810658373089, 0.8054294264911239], 'f1': [0.742990265413096, 0.7996621621621622]}
# {'accuracy': [0.7504778511786996, 0.9216994735253681], 'recall': [0.7246298788694482, 0.884185733512786], 'precision': [0.7627682935467875, 0.9553551952301316], 'auc': [0.7503906028262086, 0.9215728480410665], 'f1': [0.7432101321737931, 0.9183937371125013]}
# {'accuracy': [0.753529392039167, 0.914489789074813], 'recall': [0.7354009687836384, 0.8688105489773951], 'precision': [0.7618483412322274, 0.9556015983424597], 'auc': [0.753473032362669, 0.9143477763273015], 'f1': [0.7483910721621251, 0.9101416590316442]}

# oversample escolhido
# oversampling.to_csv(f"data/{file_tag}_train.csv", index=False)

# --------------------------------------- #
#             Classification              #
# --------------------------------------- #

# uso o de teste antes do balancing
# só o de treino é que vem do balancing


# -------------------- #
# ------ NBayes ------ #
# -------------------- #

# classificationNB = NBayes(file_tag, "data/credit_train.csv", "data/credit_test.csv", target)

# classificationNB.full_metrics_NB_study()

# accuracy -> Best Model:  BernoulliNB()  | Metric: accuracy: 0.7443077789775128
# recall -> Best Model:  BernoulliNB()  | Metric: recall:  0.7245669217407633
# precision -> Best Model:  BernoulliNB() | Metric: precision: 0.7567005201882586
# auc -> Best Model:  BernoulliNB() | Metric: auc: 0.7436035866860712
# f1 -> Best Model:  BernoulliNB() | Metric: f1: 0.7366601027808463

# NB_metric = "accuracy"
#
# # Chama a função naive_Bayes_study para obter o melhor modelo e parâmetros
# best_model, best_params = classificationNB.specific_metric_naive_Bayes_study(NB_metric)
#
# # Chama a função performance_analysis com os resultados obtidos
# classificationNB.performance_analysis(best_model, best_params)

# -------------------- #
# -------- KNN ------- #
# -------------------- #

# classificationKNN = KNN(file_tag, "data/credit_train.csv", "data/credit_test.csv", target)
#
# # classificationKNN.full_metrics_KNN_study()
#
# KNN_metric = "accuracy"
#
# # Chama a função full_metrics_KNN_study para obter o melhor modelo e parâmetros
# best_model, best_params = classificationKNN.specific_metric_KNN_study(KNN_metric)
#
# # Chama a função performance_analysis com os resultados obtidos
# classificationKNN.performance_analysis(best_model, best_params)
#
# classificationKNN.overfitting_study(best_params, KNN_metric)

# -------------------- #
# -------- DT -------- #
# -------------------- #

decisionTrees = decisionTrees(file_tag, "data/credit_train.csv", "data/credit_test.csv", target)

#decisionTrees.full_metrics_DT_study()

# d_max = 25 & lag = 2
# accuracy -> DT best with gini and d=24
# recall -> DT best with entropy and d=24
# precision -> DT best with gini and d=24
# auc -> DT best with gini and d=24
# f1 -> DT best with gini and d=24

# dt_metric = "accuracy"
#
# best_model, best_params = decisionTrees.specific_metric_DT_study(dt_metric)
#
# # decisionTrees.performance_analysis(best_model, best_params)
#
# decisionTrees.variables_importance(best_model, dt_metric)
#
# decisionTrees.simple_tree(best_model)
#
# decisionTrees.tree_importance(best_model, dt_metric)
#
# decisionTrees.overfitting_study(dt_metric, best_params)


# -------------------- #
# ------- MLP -------- #
# -------------------- #
# classificationMLP = MultiLayerPerceptron(file_tag, "data/credit_train.csv", "data/credit_test.csv", target)
#
# # Testar Parâmetros:
# N_MAX_INTER_MLP = 2500 # default
# lag_mlp = 100
# mlp_metric = "accuracy"
#
# #classificationMLP.full_metrics_mlp_study(N_MAX_INTER_MLP, lag_mlp)
#
# best_model_mlp, best_params_mlp = classificationMLP.specific_metric_mlp_study(N_MAX_INTER_MLP, lag_mlp, mlp_metric)
#
# a, b = best_model_mlp, best_params_mlp
#
# classificationMLP.performance_analysis(best_model_mlp, best_params_mlp)
#
# classificationMLP.mlp_overfitting_study(best_params_mlp, mlp_metric)


# --------------------- #
# - Gradient Boosting - #
# --------------------- #
# classificationGB = GradientBoosting(file_tag, "data/credit_train.csv", "data/credit_test.csv", target)
#
# # Testar Parâmetros:
# NR_MAX_TREES_GB = 2500 # default
# lag_gb = 500
# gb_metric = "accuracy"
#
# #classificationGB.full_metrics_gradient_boosting_study(NR_MAX_TREES_GB, lag_gb)
#
# best_model_gb, best_params_gb = classificationGB.specific_metric_gradient_boosting_study(NR_MAX_TREES_GB, lag_gb, gb_metric)
#
# # guardar os resultados
# joblib.dump((best_model_gb, best_params_gb), 'resultados_gb.joblib')
#
# classificationGB.performance_analysis(best_model_gb, best_params_gb)
# classificationGB.gb_overfitting_study(best_params_gb)
# classificationGB.visualize_variable_importance(best_model_gb, gb_metric)


# --------------------- #
# --- Random Forest --- #
# --------------------- #
# classificationRF = RandomForests(file_tag, "data/credit_train.csv" , "data/credit_test.csv", target)
#
# # Testar Parâmetros:
# NR_MAX_TREES_RF = 2500 # default
# lag_rf = 500
# rf_metric = "accuracy"
#
# #classificationRF.full_metrics_random_forests_study(NR_MAX_TREES_RF, lag_rf)
#
# best_model_rf, best_params_rf = classificationRF.specific_metric_random_forests_study(NR_MAX_TREES_RF, lag_rf, rf_metric)
#
# # guardar os resultados
# joblib.dump((best_model_rf, best_params_rf), 'resultados_rf.joblib')
#
# classificationRF.performance_analysis(best_model_rf, best_params_rf)
# classificationRF.rf_overfitting_study(best_params_rf)
# classificationRF.visualize_variable_importance(best_model_rf, rf_metric)


