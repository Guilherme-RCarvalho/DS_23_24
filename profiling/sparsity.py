'''
file:       sparsity.py
'''

# Scatter-plots      - Done: ?
# Correlation        - Done: ?

# Imports
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, subplots
from pandas import Series
from auxiliar.dslabs_functions import HEIGHT, plot_bar_chart, plot_multi_scatters_chart


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
