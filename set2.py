'''
file:       set2.py
'''
import os

import numpy as np
import pandas as pd
# Imports
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split

from auxiliar.dslabs_functions import mvi_by_filling
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




print('Set2 Data Types 1:\n', data.dtypes, '\n')

#data = data.drop(['ID', 'SSN', 'Name', "Occupation", "Type_of_Loan", "MonthlyBalance", "Customer_ID"], axis=1)
data = data.drop(['ID', 'SSN', 'Name'], axis=1)

# Dúvidas:
# Name - pode ir com o catano? ou dá para diferenciar entre homem e mulher?
# Occupation - temos de tratar dos missing values de acordo com o Customer ID , dummify?
# Type_of_Loan - separar os vários e colocar cada um como uma coluna em que apresentam valores 0 e 1
# MonthlyBalance - pode ser negativo?


# -------------------- #
# --  Data Cleaning -- #
# -------------------- #



# print("EXEMPLO DA PROF")
# df: DataFrame = mvi_by_filling(data, strategy="frequent")
# df.head(10)
#
# # Salvar o dataset modificado
# df.to_csv("DF.csv", index=False)

dim = Dimensionality(filename, file_tag, data)
# Missing Values
dim.missing_values()

def count_nan(data):
    nan_counts = data.isnull().sum()
    return nan_counts

# Exemplo de uso
nan_counts = count_nan(data)
print("1:", nan_counts)

# Substituir valores não finitos (inf) por NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Exemplo de uso
nan_counts = count_nan(data)
print("2:", nan_counts)

# Lista de colunas a serem convertidas para int
cols_to_convert_to_int = [
    'Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
    'NumofLoan', 'Delay_from_due_date', 'NumofDelayedPayment',
    'ChangedCreditLimit', 'NumCreditInquiries'
]

# Substituir "_" por nada (remover) nas colunas relevantes
data[cols_to_convert_to_int] = data[cols_to_convert_to_int].replace('_', '', regex=True)

# # Converter para int apenas as células não vazias
# data[cols_to_convert_to_int] = data[cols_to_convert_to_int].map(lambda x: pd.to_numeric(x, errors='coerce'))

# Converter para int apenas as células não vazias
data[cols_to_convert_to_int] = data[cols_to_convert_to_int].apply(pd.to_numeric, errors='coerce', downcast='integer')

# Lista de valores válidos para a coluna 'Payment_of_Min_Amount'
valid_values = ['Yes', 'No']
# Limpar células na coluna 'Payment_of_Min_Amount' que não são 'Yes' nem 'No'
data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].apply(lambda x: None if x not in valid_values else x)

data['Age'] = data['Age'].astype(int)
# Limpar células na coluna 'Age' que estão abaixo de 0 ou acima de 100
data['Age'] = data['Age'].apply(lambda x: None if x < 0 or x > 100 else x)
# Limpar células na coluna 'Num_Bank_Accounts' que estão abaixo de 0
data['Num_Bank_Accounts'] = data['Num_Bank_Accounts'].apply(lambda x: None if x < 0 else x)
# Limpar células na coluna 'Num_Credit_Card' que estão abaixo de 0
data['Num_Credit_Card'] = data['Num_Credit_Card'].apply(lambda x: None if x < 0 else x)
# Limpar células na coluna 'Interest_Rate' que estão abaixo de 0
data['Interest_Rate'] = data['Interest_Rate'].apply(lambda x: None if x < 0 else x)
# Limpar células na coluna 'NumofDelayedPayment' que estão abaixo de 0
data['NumofDelayedPayment'] = data['NumofDelayedPayment'].apply(lambda x: None if x < 0 else x)
# Limpar células na coluna 'NumofLoan' que estão abaixo de 0
data['NumofLoan'] = data['NumofLoan'].apply(lambda x: None if x < 0 else x)
# Limpar células na coluna 'NumoNumCreditInquiriesfLoan' que estão abaixo de 0
data['NumCreditInquiries'] = data['NumCreditInquiries'].apply(lambda x: None if x < 0 else x)
# Limpar células na coluna 'Delay_from_due_date' que estão abaixo de 0
data['Delay_from_due_date'] = data['Delay_from_due_date'].apply(lambda x: None if x < 0 else x)
# Limpar células na coluna 'ChangedCreditLimit' que estão abaixo de 0
data['ChangedCreditLimit'] = data['ChangedCreditLimit'].apply(lambda x: None if x < 0 else x)

# Remove 'CUS_0x' e converte os valores hexadecimais para decimal
data['Customer_ID'] = data['Customer_ID'].str.replace('CUS_0x', '').apply(lambda x: int(x, 16))

# # Remova as linhas onde 'Monthly_Inhand_Salary' é NaN
# data_filtered = data.dropna(subset=['Monthly_Inhand_Salary'])
#
# # Agrupe por 'Customer_ID' e conte os valores únicos de 'Monthly_Inhand_Salary'
# unique_values_per_customer = data_filtered.groupby('Customer_ID')['Monthly_Inhand_Salary'].nunique()
#
# # Filtre os Customer_IDs com mais de um valor único
# customers_with_multiple_salaries = unique_values_per_customer[unique_values_per_customer > 1]
#
# # Imprima o número total de Customer_IDs analisados
# total_customers_analyzed = unique_values_per_customer.shape[0]
# print(f"Número total de Customer_IDs analisados: {total_customers_analyzed}")
#
# # Imprima o número de Customer_IDs com mais de um valor único
# num_customers_with_multiple_salaries = customers_with_multiple_salaries.shape[0]
# print(f"Número de Customer_IDs com mais de um 'Monthly_Inhand_Salary': {num_customers_with_multiple_salaries}")
#
# # Imprima os Customer_IDs com mais de um valor único
# print("Customer_IDs com mais de um 'Monthly_Inhand_Salary':")
# print(customers_with_multiple_salaries.index.tolist())
#
# # Obtenha os Customer_IDs com mais de um valor único
# customer_ids_to_check = customers_with_multiple_salaries.head(3).index.tolist()
#
# # Imprima os diferentes valores de 'Monthly_Inhand_Salary' para os três primeiros Customer_IDs
# for customer_id in customer_ids_to_check:
#     unique_salaries = data_filtered[data_filtered['Customer_ID'] == customer_id]['Monthly_Inhand_Salary'].unique()
#     print(f"Customer_ID: {customer_id}, Valores Únicos de 'Monthly_Inhand_Salary': {unique_salaries}")
#
# # Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
# data['Occupation'] = data.groupby('Customer_ID')['Occupation'].fillna(method='ffill')
#
# # Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
# data['Occupation'] = data.groupby('Customer_ID')['Occupation'].fillna(method='bfill')
#
# # Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
# data['CreditMix'] = data.groupby('Customer_ID')['CreditMix'].fillna(method='ffill')
#
# # Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
# data['CreditMix'] = data.groupby('Customer_ID')['CreditMix'].fillna(method='bfill')
#
# # Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
# data['Monthly_Inhand_Salary'] = data.groupby('Customer_ID')['Monthly_Inhand_Salary'].fillna(method='ffill')
#
# # Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
# data['Monthly_Inhand_Salary'] = data.groupby('Customer_ID')['Monthly_Inhand_Salary'].fillna(method='bfill')
#
# # Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
# data['Age'] = data.groupby('Customer_ID')['Age'].fillna(method='ffill')
#
# # Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
# data['Age'] = data.groupby('Customer_ID')['Age'].fillna(method='bfill')

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

# # Preencher os valores ausentes com base nos vizinhos dentro de cada grupo
# data['Credit_History_Age_in_Months'] = data.groupby('Customer_ID')['Credit_History_Age_in_Months'].transform(lambda x: x.ffill().bfill())

# # Preencher os valores ausentes com base no último valor não nulo
# data['Credit_History_Age_in_Months'] = data.groupby('Customer_ID')['Credit_History_Age_in_Months'].fillna(method='ffill')
# data['Credit_History_Age_in_Months'] = data.groupby('Customer_ID')['Credit_History_Age_in_Months'].fillna(method='bfill')
#
# # Preencher os valores ausentes com uma sequência contínua
# data['Credit_History_Age_in_Months'] = data.groupby('Customer_ID')['Credit_History_Age_in_Months'].transform(lambda x: range(int(x.min()), int(x.max()) + 1))

# Supondo que 'data' seja o DataFrame que você está usando
customer_ids_with_missing_values = data[data['Credit_History_Age_in_Months'].isnull()]['Customer_ID'].unique()

print("Customer IDs with missing values in 'Credit_History_Age_in_Months':")
print(customer_ids_with_missing_values)

def preencher_credit_history_age2(df):
    for customer_id, group in df.groupby('Customer_ID'):
        indices = group.index
        for i, idx in enumerate(indices):
            if pd.isna(df.at[idx, 'Credit_History_Age_in_Months']):
                if i == 0:
                    df.at[idx, 'Credit_History_Age_in_Months'] = group.at[indices[i + 1], 'Credit_History_Age_in_Months'] - 1
                elif i == len(indices) - 1:
                    df.at[idx, 'Credit_History_Age_in_Months'] = group.at[indices[i - 1], 'Credit_History_Age_in_Months'] + 1
                else:
                    upper_value = group.at[indices[i - 1], 'Credit_History_Age_in_Months']
                    lower_value = group.at[indices[i + 1], 'Credit_History_Age_in_Months']
                    df.at[idx, 'Credit_History_Age_in_Months'] = np.mean([upper_value, lower_value])

def preencher_credit_history_age1(df):
    for customer_id, group in df.groupby('Customer_ID'):
        indices = group.index
        missing_indices = group.index[group['Credit_History_Age_in_Months'].isnull()]

        for i, idx in enumerate(missing_indices):
            if pd.notna(df.at[idx, 'Credit_History_Age_in_Months']):
                # Se a célula já foi preenchida em uma execução anterior, ignorar
                continue

            if i == 0:
                # Preencher o primeiro valor ausente
                prev_idx = next((i for i in reversed(indices) if i < idx), None)
                next_idx = missing_indices[i + 1] if len(missing_indices) > i + 1 else next((i for i in indices if i > idx), None)

            elif i == len(missing_indices) - 1:
                # Preencher o último valor ausente
                prev_idx = missing_indices[i - 1]
                next_idx = next((i for i in indices if i > idx), None)

            else:
                # Preencher valores ausentes no meio
                prev_idx = missing_indices[i - 1]
                next_idx = missing_indices[i + 1]

            if prev_idx is not None and next_idx is not None:
                prev_value = df.at[prev_idx, 'Credit_History_Age_in_Months']
                if pd.notna(prev_value):
                    prev_value = int(prev_value)
                else:
                    # Lida com NaN atribuindo um valor padrão ou tomando alguma ação apropriada
                    prev_value = 0  # Pode substituir por outro valor padrão ou tratamento necessário

                # Tratar caso em que next_idx pode ser None ou conter NaN
                next_value = df.at[next_idx, 'Credit_History_Age_in_Months']
                if pd.notna(next_value):
                    next_value = int(next_value)
                else:
                    # Se next_value for NaN, tentar encontrar próximo valor não NaN
                    next_value = next((int(df.at[i, 'Credit_History_Age_in_Months']) for i in indices if pd.notna(df.at[i, 'Credit_History_Age_in_Months']) and i > next_idx), None)

                if prev_value is not None and next_value is not None:
                    step = 1 if prev_value < next_value else -1
                    values_to_fill = np.arange(prev_value + step, next_value, step, dtype=int)

                    # Atualizar valores na DataFrame
                    if len(values_to_fill) == 1:
                        df.loc[idx, 'Credit_History_Age_in_Months'] = values_to_fill[0]
                    else:
                        # Se houver mais de um valor a ser preenchido, calcular a média
                        df.loc[idx, 'Credit_History_Age_in_Months'] = np.mean(values_to_fill).astype(int)

preencher_credit_history_age1(data)
preencher_credit_history_age2(data)

# Supondo que 'data' seja o DataFrame que você está usando
customer_ids_with_missing_values = data[data['Credit_History_Age_in_Months'].isnull()]['Customer_ID'].unique()

print("Customer IDs with missing values in 'Credit_History_Age_in_Months':")
print(customer_ids_with_missing_values)

# Salvar o dataset modificado
data.to_csv("cleaned_data.csv", index=False)

# Exemplo de uso
nan_counts = count_nan(data)
print("3:", nan_counts)

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
encoding = Encoding(filename, file_tag, data)

encoded_data = encoding.encode_credit()
target = "Credit_Score"
encoded_data.to_csv("encoded.csv", index=False)

print('Set2 Data Types 3:\n', encoded_data.dtypes, '\n')

# --------------------- #
# -- MV and Outliers -- #
# --------------------- #
mv = MV_out(filename, file_tag, data)

# -------------------- #
# -- Data Balancing -- #
# -------------------- #
balancing = Balancing(filename, file_tag, data)

# ---------------------------- #
# -- Scaling Transformation -- #
# ---------------------------- #
scaling = Scaling(filename, file_tag, data)

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


