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


    def evaluate_approach(self, train, test, target, metric: str = "accuracy"):
        trnY = train.pop(target).values
        trnX: ndarray = train.values
        tstY = test.pop(target).values
        tstX: ndarray = test.values
        eval: dict[str, list] = {}

        eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
        eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
        if eval_NB != {} and eval_KNN != {}:
            for met in CLASS_EVAL_METRICS:
                eval[met] = [eval_NB[met], eval_KNN[met]]

        figure()
        eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
        plot_multibar_chart(
            ["NB", "KNN"], eval, title=f"{self.file_tag} evaluation", percentage=True
        )
        savefig(f"images/{self.file_tag}_eval.png")
        show()




