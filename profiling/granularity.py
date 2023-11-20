'''
file:       granularity.py
'''

# Symbolic variables      - Done: ?

# Imports
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, subplots, show
from pandas import Series
from auxiliar.dslabs_functions import HEIGHT, plot_bar_chart, plot_multi_scatters_chart


class Granularity:
       
    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data

    def analyse_property_granularity(self,
        property: str, vars: list[str]
    ) -> ndarray:
        cols: int = len(vars)
        fig: Figure
        axs: ndarray
        fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
        fig.suptitle(f"Granularity study for {property}")
        for i in range(cols):
            counts: Series[int] = self.data[vars[i]].value_counts()
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
        # FIXME plt.tight_layout()? not good solution
        plt.tight_layout()
        savefig(f"images/{self.file_tag}_{property}_granularity.png")
        show()
