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
        savefig(f"images/{self.file_tag}_records_variables.png")        

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
        savefig(f"images/{self.file_tag}_mv.png")


    def num_variables_per_type(self):
        variable_types: dict[str, list] = get_variable_types(self.data)
        print(variable_types)
        counts: dict[str, int] = {}
        for tp in variable_types.keys():
            counts[tp] = len(variable_types[tp])

        figure(figsize=(4, 2))
        plot_bar_chart(
            list(counts.keys()), list(counts.values()), title="Nr of variables per type"
        )
        savefig(f"images/{self.file_tag}_variable_types.png")        
    



















'''

# Load Data

# Dúvida: Qual é o "index_col"? Nenhuma coluna é identificador único, sem default fica a 0
# The index should ideally be a variable with unique values for each row, serving as an unique identifier 

filename = "data/class_pos_covid.csv"
file_tag = "pos_covid"
data: DataFrame = read_csv(filename, na_values="") 


# Dimensionality
print(data.shape) # (380932, 40)


# Number of records and variables
figure(figsize=(4, 2))
values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
)
savefig(f"images/{file_tag}_records_variables.png")
show()


# Missing Values
mv: dict[str, int] = {}
for var in data.columns:
    nr: int = data[var].isna().sum()
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
savefig(f"images/{file_tag}_mv.png")
show()


# Variables Type
print(data.dtypes)


# Number of variables per type
variable_types: dict[str, list] = get_variable_types(data)
print(variable_types)
counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)
savefig(f"images/{file_tag}_variable_types.png")
show()

'''