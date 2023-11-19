'''
file:       distribution.py
'''

# Global boxplot            - Done: global_boxplot() dá bronca
# Single variables boxplots - Done: boxplots()
# Outliers study            - Done: num_outliers_per_variable()
# Histograms (num and symb) - Done: todos??
# Distribution for numeric  -
# Class Distribution        - Done: class_distribution()


# Imports
from numpy import ndarray, log
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, savefig, show, subplots, xticks
from pandas import Series
from auxiliar.dslabs_functions import HEIGHT, count_outliers, define_grid, get_variable_types, plot_bar_chart, plot_multibar_chart, plot_multiline_chart
from scipy.stats import norm, expon, lognorm

class Distribution():

    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data

    def get_numeric_variables(self):
        variables_types: dict[str, list] = get_variable_types(self.data)
        numeric: list[str] = variables_types["numeric"]
        return numeric
    
    def get_symbolic_variables(self):
        variables_types: dict[str, list] = get_variable_types(self.data)
        symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]
        return symbolic
    
    def global_boxplot(self): 
        numeric = self.get_numeric_variables()       
        if [] != numeric:
            self.data[numeric].boxplot(rot=45)
            savefig(f"images/{self.file_tag}_global_boxplot.png")
            show()
        else:
            print("There are no numeric variables.")

    def boxplots(self):
        numeric = self.get_numeric_variables()       
        if [] != numeric:
            rows: int
            cols: int
            rows, cols = define_grid(len(numeric))
            fig: Figure
            axs: ndarray
            fig, axs = subplots(
                rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
            )
            i, j = 0, 0
            for n in range(len(numeric)):
                axs[i, j].set_title("Boxplot for %s" % numeric[n])
                axs[i, j].boxplot(self.data[numeric[n]].dropna().values)
                i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
            savefig(f"images/{self.file_tag}_single_boxplots.png")
            show()
        else:
            print("There are no numeric variables.")

    # Depois pode-se juntar estas duas funcoes
    def num_standart_outliers_per_variable(self):
        numeric = self.get_numeric_variables() 
        if [] != numeric:
            outliers: dict[str, int] = count_outliers(self.data, numeric)

            figure(figsize=(12, HEIGHT))
            plot_multibar_chart(
                numeric,
                outliers,
                title="Nr of standard outliers per variable",
                xlabel="variables",
                ylabel="nr outliers",
                percentage=False,
            )
            xticks(rotation=45, ha="right")
            savefig(f"images/{self.file_tag}_outliers_standard.png")
            show()
        else:
            print("There are no numeric variables.")

    def num_outliers_per_variable(self, nrstdev, iqrfactor):
        numeric = self.get_numeric_variables()
        if [] != numeric:
            outliers: dict[str, int] = count_outliers(self.data, numeric, nrstdev, iqrfactor)
            figure(figsize=(12, HEIGHT))
            plot_multibar_chart(
                numeric,
                outliers,
                title="Nr of outliers per variable",
                xlabel="variables",
                ylabel="nr outliers",
                percentage=False,
            )
            xticks(rotation=45, ha="right")
            savefig(f"images/{self.file_tag}_outliers.png")
            show()
        else:
            print("There are no numeric variables.")

    
    def compute_known_distributions(self, x_values: list) -> dict:
        distributions = dict()
        # Gaussian
        mean, sigma = norm.fit(x_values)
        distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
        # Exponential
        loc, scale = expon.fit(x_values)
        distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)
        # LogNorm
        sigma, loc, scale = lognorm.fit(x_values)
        distributions["LogNor(%.1f,%.2f)" % (log(scale), sigma)] = lognorm.pdf(
            x_values, sigma, loc, scale
        )
        return distributions

    def histogram_with_distributions(self, ax: Axes, series: Series, var: str):
        values: list = series.sort_values().to_list()
        ax.hist(values, 20, density=True)
        distributions: dict = self.compute_known_distributions(values)
        plot_multiline_chart(
            values,
            distributions,
            ax=ax,
            title="Best fit for %s" % var,
            xlabel=var,
            ylabel="",
        )

    def histograms_numeric_with_distributions(self):
        numeric = self.get_numeric_variables()
        if [] != numeric:
            rows, cols = define_grid(len(numeric))
            _, axs = subplots(
                rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
            )
            i, j = 0, 0
            for n in range(len(numeric)):
                self.histogram_with_distributions(axs[i, j], self.data[numeric[n]].dropna(), numeric[n])
                i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
            savefig(f"images/{self.file_tag}_histogram_numeric_distribution.png")
            show()
        else:
            print("There are no numeric variables.")


    def histograms_symbolic_variables(self):
        symbolic = self.get_symbolic_variables()
        if [] != symbolic:
            rows, cols = define_grid(len(symbolic))
            _, axs = subplots(
                rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
            )
            i, j = 0, 0
            for n in range(len(symbolic)):
                counts: Series = self.data[symbolic[n]].value_counts()
                plot_bar_chart(
                    counts.index.to_list(),
                    counts.to_list(),
                    ax=axs[i, j],
                    title="Histogram for %s" % symbolic[n],
                    xlabel=symbolic[n],
                    ylabel="nr records",
                    percentage=False,
                )
                i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
            savefig(f"images/{self.file_tag}_histograms_symbolic.png")
            show()
        else:
            print("There are no symbolic variables.")


    def class_distribution(self, target):
        values: Series = self.data[target].value_counts()
        print(values)

        figure(figsize=(4, 2))
        plot_bar_chart(
            values.index.to_list(),
            values.to_list(),
            title=f"Target distribution (target={target})",
        )
        savefig(f"images/{self.file_tag}_class_distribution.png")
        show()
