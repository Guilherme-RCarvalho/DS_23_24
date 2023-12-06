'''
file:       sparsity.py
'''

# Scatter-plots      - Done: ?
# Correlation        - Done: correlation_analysis()

# Imports
from numpy import ndarray
from matplotlib.pyplot import figure, savefig, show, subplots
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from auxiliar.dslabs_functions import HEIGHT, plot_bar_chart, plot_multi_scatters_chart
from seaborn import heatmap
from auxiliar.dslabs_functions import get_variable_types, encode_cyclic_variables, dummify

class Sparsity:

    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
    
    def analyse_property_sparsity(self, target_variable):
        clean_data = self.data.dropna()
        vars: list = clean_data.columns.to_list()
        if [] != vars:

            n: int = len(vars) - 1
            axs: ndarray
            _, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
            for i in range(len(vars)):
                var1: str = vars[i]
                for j in range(i + 1, len(vars)):
                    var2: str = vars[j]
                    plot_multi_scatters_chart(clean_data, var1, var2, target_variable, ax=axs[i, j - 1])
            savefig(f"images/{self.file_tag}_sparsity_study.png")
        else:
            print("Sparsity class: there are no variables.")

    def correlation_analysis(self, encoded_data):        

        '''
        variables_types: dict[str, list] = get_variable_types(encoded_data)
        numeric: list[str] = variables_types["numeric"]
        corr_mtx: DataFrame = encoded_data[numeric].corr().abs()
        '''
        
        corr_mtx: DataFrame = encoded_data.corr().abs()

        figure()
        heatmap(
            abs(corr_mtx),
            #xticklabels=numeric,
            #yticklabels=numeric,
            annot=False,
            cmap="Blues",
            #vmin=0,
            #vmax=1,
        )
        savefig(f"images/{self.file_tag}_correlation_analysis.png")
        show()

    

    # Ordinal Encoding, Cyclic Variables (nao temos?), Dummification or One-hot Encoding:
        # Ordinal Encoding: O numeric mantem-se, só muda o binary e symbolic (nao ha date), ter em atencao a ordem!
        # Non-ordinal symbolic variables that don't have a meaningful order are typically not included
        # in this ordinal encoding process.
    
    def encode_pos_covid(self):
        encoded_data = self.data.copy()        

        encoding = {
            # Symbolic (Ordinal)
            "GeneralHealth": {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4},
            "LastCheckupTime": {"Within past year (anytime less than 12 months ago)": 0, "Over a year ago": 1, "Within past 2 years (1 year but less than 2 years ago)": 2, "Within past 5 years (2 years but less than 5 years ago)": 3, "5 or more years ago": 4 , "Never": 5},
            "SmokerStatus": {"Never smoked": 0, "Former smoker": 1, "Current smoker - now smokes some days": 2, "Current smoker - now smokes every day": 3},            
            "AgeCategory": {"Age 18 to 24": 0, "Age 25 to 29": 1, "Age 30 to 34": 2, "Age 35 to 39": 3, "Age 40 to 44": 4, "Age 45 to 49": 5, "Age 50 to 54": 6, "Age 55 to 59": 7, "Age 60 to 64": 8, "Age 65 to 69": 9, "Age 70 to 74": 10, "Age 75 to 79": 11, "Age 80 or older": 12},
            "TetanusLast10Tdap": {"No, did not receive any tetanus shot in the past 10 years": 0, "Yes, received tetanus shot but not sure what type": 1, "Yes, received tetanus shot, but not Tdap": 2, "Yes, received Tdap": 3},
            "RemovedTeeth": {"None of them": 0, "1 to 5": 1, "6 or more, but not all": 2,"All": 3},
            "ECigaretteUsage": {"Never used e-cigarettes in my entire life": 0, "Not at all (right now)": 1, "Use them some days": 2,"Use them every day": 3},
            "HadDiabetes": {"No": 0, "No, pre-diabetes or borderline diabetes": 1, "Yes": 2, "Yes, but only during pregnancy (female)": 3},
            #ordered by gdp/capita
            "State" : {'Alabama': 54753, 'Alaska': 86722, 'Arizona': 62365, 'Arkansas': 54259, 'California': 92190, 'Colorado': 82954, 'Connecticut': 88760, 'Delaware': 85977, 'District of Columbia': 242853, 'Florida': 62446, 'Georgia': 69253, 'Hawaii': 68207, 'Idaho': 56496, 'Illinois': 82126, 'Indiana': 66698, 'Iowa': 72221, 'Kansas': 71729, 'Kentucky': 57653, 'Louisiana': 61313, 'Maine': 61008, 'Maryland': 76279, 'Massachusetts': 98609, 'Michigan': 61859, 'Minnesota': 78100, 'Mississippi': 47190, 'Missouri': 63126, 'Montana': 57945, 'Nebraska': 82207, 'Nevada': 67962, 'New Hampshire': 75565, 'New Jersey': 80490, 'New Mexico': 57792, 'New York': 104344, 'North Carolina': 68243, 'North Dakota': 94021, 'Ohio': 69978, 'Oklahoma': 59894, 'Oregon': 70548, 'Pennsylvania': 71160, 'Rhode Island': 65362, 'South Carolina': 56066, 'South Dakota': 74268, 'Tennessee': 67473, 'Texas': 78456, 'Utah': 73424, 'Vermont': 62771, 'Virginia': 74789, 'Washington': 93193, 'West Virginia': 53852, 'Wisconsin': 68192, 'Wyoming': 81586, 'Guam': 35904, 'Puerto Rico': 35208, 'Virgin Islands': 39552},
            #based on similarities between ethnicities
            "RaceEthnicityCategory" : {'White only, Non-Hispanic': 1, 'Other race only, Non-Hispanic': 2, 'Multiracial, Non-Hispanic': 3, 'Hispanic': 4, 'Black only, Non-Hispanic': 5},
            # Binary - mudar c a da prof so yes no Yes No wtv
            "Sex": {"Male": 0, "Female": 1}, 
            "PhysicalActivities": {"No": 0, "Yes": 1},
            "HadHeartAttack": {"No": 0, "Yes": 1},
            "HadAngina": {"No": 0, "Yes": 1},
            "HadStroke": {"No": 0, "Yes": 1},
            "HadAsthma": {"No": 0, "Yes": 1},
            "HadSkinCancer": {"No": 0, "Yes": 1},
            "HadCOPD": {"No": 0, "Yes": 1},
            "HadDepressiveDisorder": {"No": 0, "Yes": 1},
            "HadKidneyDisease": {"No": 0, "Yes": 1},
            "HadArthritis": {"No": 0, "Yes": 1},
            "DeafOrHardOfHearing": {"No": 0, "Yes": 1},
            "BlindOrVisionDifficulty": {"No": 0, "Yes": 1},
            "DifficultyConcentrating": {"No": 0, "Yes": 1},
            "DifficultyWalking": {"No": 0, "Yes": 1},
            "DifficultyDressingBathing": {"No": 0, "Yes": 1},
            "DifficultyErrands": {"No": 0, "Yes": 1},
            "ChestScan": {"No": 0, "Yes": 1},
            "AlcoholDrinkers": {"No": 0, "Yes": 1},
            "HIVTesting": {"No": 0, "Yes": 1},
            "FluVaxLast12": {"No": 0, "Yes": 1},
            "PneumoVaxEver": {"No": 0, "Yes": 1},
            "HighRiskLastYear": {"No": 0, "Yes": 1},
            "CovidPos": {"No": 0, "Yes": 1},                         
        }

        # Encoding ordinal variables
        encoded_data.replace(encoding, inplace=True)
            
        # Dummification (One-hot encoding) (non-ordenal)
        # The dummifying 'State' will create separate binary columns for each unique state, increasing the dimensionality of your dataset.             
        #vars: list[str] = ["RaceEthnicityCategory"]
        #df_encoded = dummify(encoded_data, vars)
                
        return encoded_data

    def encode_credit(self):
        encoded_data = self.data.copy().dropna()

        encoding = {
            # Symbolic (Ordinal)
            "CreditMix": {"Bad": 0, "Standard": 1, "Good": 2},
            "Payment_Behaviour": {"Low_spent_Small_value_payments": 0, "Low_spent_Medium_value_payments": 1,
                                      "Low_spent_Large_value_payments": 2, "High_spent_Small_value_payments": 3,
                                      "High_spent_Medium_value_payments": 4, "High_spent_Large_value_payments": 5},


            # Binary - mudar c a da prof so yes no Yes No wtv
            "Payment_of_Min_Amount": {"No": 0, "Yes": 1},
            "Credit_Score": {"Poor": 0, "Good": 1},
        }

        # Encoding ordinal variables
        encoded_data.replace(encoding, inplace=True)

        # Dummification (One-hot encoding) (non-ordenal)
        # The dummifying 'State' will create separate binary columns for each unique state, increasing the dimensionality of your dataset. - help
        vars: list[str] = ["Customer_ID", "Name", "Occupation", "Type_of_Loan", "Month"]
        df_encoded = dummify(encoded_data, vars)

        return df_encoded

    def correlation_analysis_num(self, encoded_data):
        variables_types: dict[str, list] = get_variable_types(encoded_data)
        numeric: list[str] = variables_types["numeric"]
        corr_mtx: DataFrame = encoded_data[numeric].corr().abs()


        corr_mtx: DataFrame = encoded_data.corr().abs()

        figure()
        heatmap(
            abs(corr_mtx),
            xticklabels=numeric,
            yticklabels=numeric,
            annot=False,
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        savefig(f"images/{self.file_tag}_correlation_analysis.png")
        show()
