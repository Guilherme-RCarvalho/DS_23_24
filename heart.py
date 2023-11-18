from pandas import DataFrame, read_csv, Series
from matplotlib.pyplot import figure, savefig, show, subplots
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dslabs_functions import plot_bar_chart, get_variable_types, HEIGHT, plot_multi_scatters_chart
from numpy import ndarray

filename = "class_pos_covid.csv"
file_tag = "health_domain"
data: DataFrame = read_csv(filename, na_values="")
'''
#Data Dimensionality
print(data.shape)
figure(figsize=(4, 2))
values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
)
savefig(f"figures/{file_tag}_records_variables.png")

#Data distribuition
summary: DataFrame = data.describe(include="all")
var = "TetanusLast10Tdap"
print(summary[var])
print(data[var].unique())
'''
#Data granularity
def analyse_property_granularity(
    data: DataFrame, property: str, vars: list[str]
) -> ndarray:
    cols: int = len(vars)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {property}")
    for i in range(cols):
        counts: Series[int] = data[vars[i]].value_counts()
        counts = counts.sort_index()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[0, i],
            title=vars[i],
            xlabel=vars[i],
            ylabel="nr records",
            percentage=False,
        )
    #FIXME plt.tight_layout()? not good solution
    return axs
#Demographic granularity
analyse_property_granularity(data, "demography", ["Sex", "RaceEthnicityCategory", "AgeCategory", "State"])
savefig(f"images/{file_tag}_granularity_demography.png")

#Health Indicators #TODO change Days interval?
analyse_property_granularity(data, "health indicators", ["GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "LastCheckupTime"])
savefig(f"images/{file_tag}_granularity_health_indicators.png")

#TODO Sub-Levels of information? SleepHours, SmokerStatus+ECigars


#Data Sparsity
data = data.dropna()
vars: list = data.columns.to_list()
if [] != vars:
    target = "CovidPos"

    n: int = len(vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])
    savefig(f"images/{file_tag}_sparsity_study.png")
else:
    print("Sparsity class: there are no variables.")
	