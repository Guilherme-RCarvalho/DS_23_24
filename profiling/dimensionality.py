'''
file:       dimensionality.py
'''

# Nr records x Nr variables - Done: num_records_variables()
# Nr Variable types         - Done: num_variables_per_type()
# Nr missing values         - Done: missing_values()

#Imports
from matplotlib.pyplot import figure, savefig, show
from auxiliar.dslabs_functions import plot_bar_chart, get_variable_types

class Dimensionality:

    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data

    def num_records_variables(self):
        figure(figsize=(4, 2))
        values: dict[str, int] = {"nr records": self.data.shape[0], "nr variables": self.data.shape[1]}
        plot_bar_chart(
            list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
        )
        savefig(f"images/profiling/{self.file_tag}_records_variables.png")

    def missing_values(self):
        mv: dict[str, int] = {}
        for var in self.data.columns:
            nr: int = self.data[var].isna().sum()
            if nr > 0:
                mv[var] = nr

        figure()
        plot_bar_chart(
            list(mv.keys()),
            list(mv.values()),
            title="Nr of missing values per variable",
            xlabel="variables",
            ylabel="nr missing values",
        )
        savefig(f"images/profiling/{self.file_tag}_mv.png")


    def num_variables_per_type(self):
        variable_types: dict[str, list] = get_variable_types(self.data)
        print(variable_types, '\n')
        counts: dict[str, int] = {}
        for tp in variable_types.keys():
            counts[tp] = len(variable_types[tp])

        figure(figsize=(4, 2))
        plot_bar_chart(
            list(counts.keys()), list(counts.values()), title="Nr of variables per type"
        )
        savefig(f"images/profiling/{self.file_tag}_variable_types.png")
    
