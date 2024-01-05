from sklearn.preprocessing import StandardScaler
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, savefig, show
import matplotlib.pyplot as plt
from auxiliar.dslabs_functions import plot_line_chart, HEIGHT

class ts_Scaling:

    def __init__(self, filename, file_tag, data, target):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data
        self.target   = target

    def scale_all_dataframe(self, data: DataFrame) -> DataFrame:
        vars: list[str] = data.columns.to_list()
        transf: StandardScaler = StandardScaler().fit(data)
        df = DataFrame(transf.transform(data), index=data.index)
        df.columns = vars
        return df
    
    def ts_scale(self):
        df: DataFrame = self.scale_all_dataframe(self.data)
        ss: Series = df[self.target]
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            ss.index.to_list(),
            ss.to_list(),
            xlabel=ss.index.name,
            ylabel=self.target,
            title=f"{self.file_tag} {self.target} after scaling",
        )
        plt.tight_layout()
        savefig(f"images/timeSeries/transformation/scaling/{self.file_tag}_{self.target}_after_scaling.png")
        show()