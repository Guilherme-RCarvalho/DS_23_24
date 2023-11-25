'''
file:       methodology.py
'''

from numpy import ndarray
from pandas import DataFrame
from matplotlib.pyplot import savefig, show, figure
from auxiliar.dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN, evaluate_approach


class Methodology:

    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data


    def evaluate_approach(self, train, test, target):
        figure()
        eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
        plot_multibar_chart(
            ["NB", "KNN"], eval, title=f"{self.file_tag} evaluation", percentage=True
        )
        savefig(f"images/{self.file_tag}_eval.png")
        show()




