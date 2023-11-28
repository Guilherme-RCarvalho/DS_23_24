from pandas import read_csv, DataFrame, Series
from auxiliar.dslabs_functions import (
    NR_STDEV,
    get_variable_types,
    determine_outlier_thresholds_for_var,
)
from matplotlib.pyplot import figure, show, savefig
from auxiliar.dslabs_functions import plot_bar_chart
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class MV_out:

    def __init__(self, filename, file_tag, data, n_std):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.n_std = n_std
        self.numeric_vars: list[str] = get_variable_types(self.data)["numeric"]
        self.summary5: DataFrame = self.data[self.numeric_vars].describe()
        
    def drop_outliers(self):
        if self.numeric_vars is not None:
            df: DataFrame = self.data.copy(deep=True)
            for var in self.numeric_vars:
                top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
                    self.summary5[var], threshold=self.n_std
                )
                outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
                df.drop(outliers.index, axis=0, inplace=True)
            df.to_csv(f"data/{self.file_tag}_drop_outliers.csv", index=True)
            print(f"Data after dropping outliers: {df.shape}")
        else:
            print("There are no numeric variables")
            
    def median_outliers(self):
        if [] != self.numeric_vars:
            df: DataFrame = self.data.copy(deep=True)
            for var in self.numeric_vars:
                top, bottom = determine_outlier_thresholds_for_var(
                    self.summary5[var], threshold=self.n_std
                )
                median: float = df[var].median()
                df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)
            df.to_csv(f"data/{self.file_tag}_replacing_outliers.csv", index=True)
            print("Data after replacing outliers:", df.shape)
            print(df.describe())
        else:
            print("There are no numeric variables")
        return df

    def truncate_outliers(self):
        if [] != self.numeric_vars:
            df: DataFrame = self.data.copy(deep=True)
            for var in self.numeric_vars:
                top, bottom = determine_outlier_thresholds_for_var(self.summary5[var], threshold=self.n_std)
                df[var] = df[var].apply(
                    lambda x: top if x > top else bottom if x < bottom else x
                )
            df.to_csv(f"data/{self.file_tag}_truncate_outliers.csv", index=True)
            print("Data after truncating outliers:", df.shape)
            print(df.describe())
        else:
            print("There are no numeric variables")
        return df

    def count_nan(self):
        data=self.data
        nan_counts = data.isnull().sum()
        return nan_counts

    def read_mv(self):
        print(f"Dataset nr records={self.data.shape[0]}", f"nr variables={self.data.shape[1]}") #TODO Comment after tests
        mv: dict[str, int] = {}
        figure()
        for var in self.data:
            nr: int = self.data[var].isna().sum()
            if nr > 0:
                mv[var] = nr
        plot_bar_chart(
            list(mv.keys()),
            list(mv.values()),
            title="Missing values per variable",
            xlabel="variables",
            ylabel="nr missing values",
        )
        plt.tight_layout()
        savefig(f"images/{self.file_tag}_MissingValues_per_variable.png")
        show()



    def data_correction(self):
        data = self.data
        # Substituir valores não finitos (inf) por NaN
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Lista de colunas a serem convertidas para int
        cols_to_convert_to_int = [
            'Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
            'NumofLoan', 'Delay_from_due_date', 'NumofDelayedPayment',
            'ChangedCreditLimit', 'NumCreditInquiries'
        ]

        # Substituir "_" por nada (remover) nas colunas relevantes
        data[cols_to_convert_to_int] = data[cols_to_convert_to_int].replace('_', '', regex=True)

        # Converter para int apenas as células não vazias
        data[cols_to_convert_to_int] = data[cols_to_convert_to_int].apply(pd.to_numeric, errors='coerce',
                                                                          downcast='integer')

        # Lista de valores válidos para a coluna 'Payment_of_Min_Amount'
        valid_values = ['Yes', 'No']

        # Limpar células na coluna 'Payment_of_Min_Amount' que não são 'Yes' nem 'No'
        data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].apply(
            lambda x: None if x not in valid_values else x)
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




        # Salvar o dataset modificado
        data.to_csv("cleaned_data.csv", index=False)

        return data

    def our_mv(self, encoded_data):
        print("Estou no our mv")
        data = encoded_data
        def preencher_credit_history_age2(df):

            for customer_id, group in df.groupby('Customer_ID'):
                indices = group.index
                for i, idx in enumerate(indices):
                    if pd.isna(df.at[idx, 'Credit_History_Age_in_Months']):
                        if i == 0:
                            df.at[idx, 'Credit_History_Age_in_Months'] = group.at[
                                                                             indices[
                                                                                 i + 1], 'Credit_History_Age_in_Months'] - 1
                        elif i == len(indices) - 1:
                            df.at[idx, 'Credit_History_Age_in_Months'] = group.at[
                                                                             indices[
                                                                                 i - 1], 'Credit_History_Age_in_Months'] + 1
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
                        next_idx = missing_indices[i + 1] if len(missing_indices) > i + 1 else next(
                            (i for i in indices if i > idx), None)

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
                            next_value = next((int(df.at[i, 'Credit_History_Age_in_Months']) for i in indices if
                                               pd.notna(df.at[i, 'Credit_History_Age_in_Months']) and i > next_idx),
                                              None)

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


        # # Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
        # data['Occupation'] = data.groupby('Customer_ID')['Occupation'].fillna(method='ffill')
        #
        # # Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
        # data['Occupation'] = data.groupby('Customer_ID')['Occupation'].fillna(method='bfill')

        # Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
        data['CreditMix'] = data.groupby('Customer_ID')['CreditMix'].fillna(method='ffill')

        # Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
        data['CreditMix'] = data.groupby('Customer_ID')['CreditMix'].fillna(method='bfill')

        # Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
        data['Monthly_Inhand_Salary'] = data.groupby('Customer_ID')['Monthly_Inhand_Salary'].fillna(method='ffill')

        # Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
        data['Monthly_Inhand_Salary'] = data.groupby('Customer_ID')['Monthly_Inhand_Salary'].fillna(method='bfill')

        # Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
        data['Age'] = data.groupby('Customer_ID')['Age'].fillna(method='ffill')

        # Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
        data['Age'] = data.groupby('Customer_ID')['Age'].fillna(method='bfill')
        print("no fim deste manicomio")

        # def fill_values_id_based():
        #     variables = ['Occupation', 'CreditMix', 'Monthly_Inhand_Salary', 'Age']
        #     for var_type in variables:
        #         # Preencher os valores ausentes com base no último valor não nulo dentro de cada grupo de 'Customer_ID'
        #         data[var_type] = data.groupby('Customer_ID')[var_type].fillna(method='ffill')
        #
        #         # Se ainda houver algum NaN após o preenchimento para o último valor não nulo, preenche com o primeiro valor não nulo
        #         data[var_type] = data.groupby('Customer_ID')[var_type].fillna(method='bfill')
        return data




