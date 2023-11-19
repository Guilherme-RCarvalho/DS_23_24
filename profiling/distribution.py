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
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, savefig, show, subplots, xticks
from pandas import Series
from auxiliar.dslabs_functions import HEIGHT, count_outliers, define_grid, get_variable_types, plot_bar_chart, \
    plot_multibar_chart, plot_multiline_chart
from scipy.stats import norm, expon, lognorm
from auxiliar.dslabs_functions import set_chart_labels


class Distribution():

    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data = data

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

    def global_boxplot2(self):
        numeric = self.get_numeric_variables()
        print("1")
        print(self.data[numeric].shape[1])
        print("2")
        print(self.data[numeric].columns)
        print("3")
        print(self.data[numeric].head())
        if [] != numeric:
            fig, ax = subplots(figsize=(12, HEIGHT))
            self.data[numeric].boxplot(ax=ax, rot=45)
            ax.set_title(f"Global Boxplot for Numeric Variables")
            ax.set_ylabel("Values")
            ax.set_xlabel("Variables")
            savefig(f"images/{self.file_tag}_global_boxplot.png")
            show()
        else:
            print("There are no numeric variables.")


    def global_boxplot3(self):
        numeric = self.get_numeric_variables()
        # Remova temporariamente a linha com o valor problemático
        problematic_row_index = self.data[self.data['MonthlyBalance'] == -3.33333E+26].index
        self.data_temp = self.data.drop(problematic_row_index)
        # Crie o boxplot
        if [] != numeric:
            fig, ax = subplots(figsize=(12, HEIGHT))
            self.data_temp[numeric].boxplot(ax=ax, rot=45)
            savefig(f"images/{self.file_tag}_global_boxplot3.png")
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
    
    def histogram_numeric(self):
        numeric = self.get_numeric_variables()
        if [] != numeric:
            rows, cols = define_grid(len(numeric))
            _, axs = subplots(
                rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
            )
            i: int
            j: int
            i, j = 0, 0
            for n in range(len(numeric)):
                set_chart_labels(
                    axs[i, j],
                    title=f"Histogram for {numeric[n]}",
                    xlabel=numeric[n],
                    ylabel="nr records",
                )
                axs[i, j].hist(self.data[numeric[n]].dropna().values, "auto")
                i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
            plt.tight_layout()
            savefig(f"images/{self.file_tag}_single_histograms_numeric.png")
        else:
            print("There are no numeric variables.")

    def histograms_numeric_with_distributions(self):
        numeric = self.get_numeric_variables()
        if [] != numeric:
            rows, cols = define_grid(len(numeric))
            _, axs = subplots(
                rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
            )
            i: int
            j: int
            i, j = 0, 0
            for n in range(len(numeric)):
                print(f'histogram for numeric: {n}\n{self.data[numeric[n]]}')
                self.histogram_with_distributions(axs[i, j], self.data[numeric[n]].dropna(), numeric[n])
                i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
            plt.tight_layout()
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
            plt.tight_layout()
            savefig(f"images/{self.file_tag}_histograms_symbolic.png",bbox_inches='tight')
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
