'''
file:       KNN.py
'''
from auxiliar.dslabs_functions import read_train_test_from_files, CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart, \
    plot_evaluation_results, plot_multiline_chart
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig, show
import matplotlib.pyplot as plt
from typing import Literal
from numpy import array, ndarray
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure, savefig, show


class KNN:

    def __init__(self, file_tag, trainFilename: str, testFilename: str, target):
        self.file_tag = file_tag
        self.target = target
        self.trnX, self.tstX, self.trnY, self.tstY, self.labels, self.vars = read_train_test_from_files(trainFilename,
                                                                                                        testFilename,
                                                                                                        target)

    def knn_study(
            self, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, k_max, lag: int = 2, metric='accuracy'
    ) -> tuple[KNeighborsClassifier | None, dict]:
        dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = ['manhattan', 'euclidean', 'chebyshev']
        print("estou aqui 1")
        kvalues: list[int] = [i for i in range(1, k_max + 1, lag)]
        best_model: KNeighborsClassifier | None = None
        best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
        best_performance: float = 0.0

        values: dict[str, list] = {}
        for d in dist:
            y_tst_values: list = []
            for k in kvalues:
                clf = KNeighborsClassifier(n_neighbors=k, metric=d)
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance: float = eval
                    best_params['params'] = (k, d)
                    best_model = clf
                print(f'KNN {d} k={k}')
            values[d] = y_tst_values
            print("estou aqui 2")
        print(f'KNN best with k={best_params["params"][0]} and {best_params["params"][1]}')
        plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)
        print("estou aqui3")
        return best_model, best_params

    def full_metrics_KNN_study(self):
        print("estou aqui 4")
        eval_metric = 'accuracy'
        figure()
        best_model, params = self.knn_study(self.trnX, self.trnY, self.tstX, self.tstY, k_max=25, metric=eval_metric)
        savefig(f'images/{self.file_tag}_knn_{eval_metric}_study.png')
        show()
        return best_model, params

    def performance_analysis(self, best_model, params):
        prd_trn: array = best_model.predict(self.trnX)
        prd_tst: array = best_model.predict(self.tstX)
        figure()
        plot_evaluation_results(params, self.trnY, prd_trn, self.tstY, prd_tst, self.labels)
        savefig(f'images/{self.file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png')
        show()