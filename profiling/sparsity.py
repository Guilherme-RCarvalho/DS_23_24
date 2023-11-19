'''
file:       sparsity.py
'''

# Scatter-plots      - Done: ?
# Correlation        - Done: correlation_analysis()

# Imports
from numpy import ndarray
from matplotlib.pyplot import figure, savefig, show, subplots
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

    def correlation_analysis(self):
        variables_types: dict[str, list] = get_variable_types(self.data)
        numeric: list[str] = variables_types["numeric"]
        corr_mtx: DataFrame = self.data[numeric].corr().abs()

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



    # VERIFIQUEM Q DEVE TAR MAL
    # Horroroso tudo hardcoded mas chatgpt <3 

    # Ordinal Encoding, Cyclic Variables (nao temos?), Dummification or One-hot Encoding:
        # Ordinal Encoding: O numeric mantem-se, só muda o binary e symbolic (nao ha date), ter em atencao a ordem!
        # Non-ordinal symbolic variables that don't have a meaningful order are typically not included
        # in this ordinal encoding process.
            ### -> No - 0, Yes - 1 (binary)

    def encode_pos_covid(self):
        encoded_data = self.data.copy()
        
        encoding = {
            # Symbolic (Ordinal)
            "GeneralHealth": {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4},
            "LastCheckupTime": {"Never": 0, "Within past year (anytime less than 12 months ago)": 1, "Over a year ago": 2},
            "SmokerStatus": {"Never smoked": 0, "Former smoker": 1, "Current smoker - now smokes some days": 2, "Current smoker - now smokes every day": 3},            
            "AgeCategory": {"Age 40 to 44": 0, "Age 55 to 59": 1, "Age 65 to 69": 2, "Age 70 to 74": 3, "Age 75 to 79": 4, "Age 80 or older": 5},
            "TetanusLast10Tdap": {"No": 0, "Yes, received tetanus shot but not sure what type": 1, "Yes, received Tdap": 2},
            
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
            "HadDiabetes": {"No": 0, "Yes": 1},
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

        # Dummification (One-hot encoding) (non-ordenal) ???????????????
        # The dummifying 'State' will create separate binary columns for each unique state, 
        # increasing the dimensionality of your dataset. - help
        vars_to_dummify = ["State", "RaceEthnicityCategory"]
        df_encoded = dummify(encoded_data, vars_to_dummify)
        
        df_encoded = encoded_data.replace(encoding, inplace=False)
        return df_encoded



    def encode_credit(self):
        pass