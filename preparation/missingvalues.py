
from numpy import ndarray
from pandas import DataFrame, concat
from sklearn.impute import KNNImputer, SimpleImputer

from auxiliar.dslabs_functions import get_variable_types


class MissingValues:

    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data


    # Another option might be to only drop variables or records that show valid values above a given threshold - 
    # through the use of the thresh parameter.
    def mvi_by_dropping(self, min_pct_per_var: float = 0.1, min_pct_per_rec: float = 0.0) -> DataFrame:        
        # Deleting variables
        df: DataFrame = self.data.dropna(
            axis=1, thresh=self.data.shape[0] * min_pct_per_var, inplace=False
        )
        
        # Deleting records
        df.dropna(axis=0, thresh= self.data.shape[1] * min_pct_per_rec, inplace=True)

        return df


    def mvi_by_filling(self, strategy: str = "frequent") -> DataFrame:
        df: DataFrame
        variables: dict = get_variable_types(self.data)
        stg_num, v_num = "mean", -1
        stg_sym, v_sym = "most_frequent", "NA"
        stg_bool, v_bool = "most_frequent", False
        if strategy != "knn":
            lst_dfs: list = []
            if strategy == "constant":
                stg_num, stg_sym, stg_bool = "constant", "constant", "constant"
            if len(variables["numeric"]) > 0:
                imp = SimpleImputer(strategy=stg_num, fill_value=v_num, copy=True)
                tmp_nr = DataFrame(
                    imp.fit_transform(self.data[variables["numeric"]]),
                    columns=variables["numeric"],
                )
                lst_dfs.append(tmp_nr)
            if len(variables["symbolic"]) > 0:
                imp = SimpleImputer(strategy=stg_sym, fill_value=v_sym, copy=True)
                tmp_sb = DataFrame(
                    imp.fit_transform(self.data[variables["symbolic"]]),
                    columns=variables["symbolic"],
                )
                lst_dfs.append(tmp_sb)
            if len(variables["binary"]) > 0:
                imp = SimpleImputer(strategy=stg_bool, fill_value=v_bool, copy=True)
                tmp_bool = DataFrame(
                    imp.fit_transform(self.data[variables["binary"]]),
                    columns=variables["binary"],
                )
                lst_dfs.append(tmp_bool)
            df = concat(lst_dfs, axis=1)
        else:
            imp = KNNImputer(n_neighbors=5)
            imp.fit(self.data)
            ar: ndarray = imp.transform(self.data)
            df = DataFrame(ar, columns=self.data.columns, index=self.data.index)
        return df

