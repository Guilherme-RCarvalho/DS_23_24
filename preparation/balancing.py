'''
file:       balancing.py
'''

from pandas import concat, DataFrame, Series
from numpy import ndarray
from matplotlib.pyplot import figure, show
from auxiliar.dslabs_functions import plot_bar_chart
from imblearn.over_sampling import SMOTE

class Balancing:

    def __init__(self, filename, file_tag, data, target):
        self.filename = filename
        self.file_tag = file_tag
        self.original = data
        self.target   = target
        self.target_count: Series = self.original[self.target].value_counts()
        self.positive_class = self.target_count.idxmin()
        self.negative_class = self.target_count.idxmax()
        self.df_positives: Series = self.original[self.original[self.target] == self.positive_class]
        self.df_negatives: Series = self.original[self.original[self.target] == self.negative_class]
        self.random_state = 42


    def check_balance(self):
        
        print("Minority class=", self.positive_class, ":", self.target_count[self.positive_class])
        print("Majority class=", self.negative_class, ":", self.target_count[self.negative_class])
        print(
            "Proportion:",
            round(self.target_count[self.positive_class] / self.target_count[self.negative_class], 2),
            ": 1",
        )
        values: dict[str, list] = {
            "Original": [self.target_count[self.positive_class], self.target_count[self.negative_class]]
        }

        figure()
        plot_bar_chart(
            self.target_count.index.to_list(), self.target_count.to_list(), title="Class balance"
        )
        show()

    #dataset1 uses undersampling
    def undersampling(self): 
        df_neg_sample: DataFrame = DataFrame(self.df_negatives.sample(len(self.df_positives)))
        df_under: DataFrame = concat([self.df_positives, df_neg_sample], axis=0)
        print("Minority class=", self.positive_class, ":", len(self.df_positives))
        print("Majority class=", self.negative_class, ":", len(df_neg_sample))
        print("Proportion:", round(len(self.df_positives) / len(df_neg_sample), 2), ": 1")
        df_under.to_csv(f"data/{self.file_tag}_balance_undersample.csv", index=False)
        return df_under

    def oversampling(self):
        df_pos_sample: DataFrame = DataFrame(self.df_positives.sample(len(self.df_negatives), replace=True))
        df_over: DataFrame = concat([df_pos_sample, self.df_negatives], axis=0)
        print("Minority class=", self.positive_class, ":", len(df_pos_sample))
        print("Majority class=", self.negative_class, ":", len(self.df_negatives))
        print("Proportion:", round(len(df_pos_sample) / len(self.df_negatives), 2), ": 1")
        df_over.to_csv(f"data/{self.file_tag}_balance_oversample.csv", index=False)
        return df_over

    def SMOTE(self):
        smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=self.random_state)
        y = self.original.pop(self.target).values
        X: ndarray = self.original.values
        smote_X, smote_y = smote.fit_resample(X, y)
        df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
        df_smote.columns = list(self.original.columns) + [self.target]
        smote_target_count: Series = Series(smote_y).value_counts()
        print("Minority class=", self.positive_class, ":", smote_target_count[self.positive_class])
        print("Majority class=", self.negative_class, ":", smote_target_count[self.negative_class])
        print(
            "Proportion:",
            round(smote_target_count[self.positive_class] / smote_target_count[self.negative_class], 2),
            ": 1",
        )
        df_smote.to_csv(f"data/{self.file_tag}_balance_smote.csv", index=False)
        return df_smote