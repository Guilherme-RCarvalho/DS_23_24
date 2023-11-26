'''
file:       missingValues.py
'''
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, show, savefig
from auxiliar.dslabs_functions import plot_bar_chart
import matplotlib.pyplot as plt


class MissingValues:

    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data = data
        
    
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