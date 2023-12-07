from typing import Literal
from numpy import array
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure, savefig, show
from auxiliar.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart
from auxiliar.dslabs_functions import read_train_test_from_files, plot_evaluation_results
from matplotlib.pyplot import figure, savefig, show
import matplotlib.pyplot as plt

class Knn:

    def __init__(self, file_tag, trainFilename: str, testFilename: str, target):
        self.file_tag = file_tag
        self.target   = target
        self.trnX, self.tstX, self.trnY, self.tstY, self.labels, self.vars = read_train_test_from_files(trainFilename, testFilename, target)

    def knn_study(self, k_max: int=19, lag: int=2, metric='accuracy') -> tuple[KNeighborsClassifier | None, dict]:

        dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = ['manhattan', 'euclidean', 'chebyshev']

        kvalues: list[int] = [i for i in range(1, k_max+1, lag)]
        best_model: KNeighborsClassifier | None = None
        best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
        best_performance: float = 0.0

        values: dict[str, list] = {}
        for d in dist:
            print(f"new distance: {d}")
            y_tst_values: list = []
            for k in kvalues:
                clf = KNeighborsClassifier(n_neighbors=k, metric=d, n_jobs=-1)
                clf.fit(self.trnX, self.trnY)
                prdY: array = clf.predict(self.tstX)
                eval: float = CLASS_EVAL_METRICS[metric](self.tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance: float = eval
                    best_params['params'] = (k, d)
                    best_model = clf
                print(f'KNN {d} k={k}')
            values[d] = y_tst_values
        print(f'KNN best with k={best_params["params"][0]} and {best_params["params"][1]}')
        plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)

        return best_model, best_params

    def full_metrics_knn_study(self, k_max, lag):
        #eval_metrics = ["accuracy","recall","precision","auc","f1"]
        eval_metrics = ["accuracy","recall"]
        for eval_metric in eval_metrics:
            figure()
            print(f"new metric: {eval_metric}")
            best_model, params = self.knn_study(k_max, lag, eval_metric)
            plt.tight_layout()
            savefig(f"images/classification/{self.file_tag}_knn-{k_max}_{eval_metric}_study.png")
            self.performance_analysis(best_model,params)
            if eval_metric == "accuracy":
                self.knn_overfitting_study(params)

    def performance_analysis(self, best_model, params):
        prd_trn: array = best_model.predict(self.trnX)
        prd_tst: array = best_model.predict(self.tstX)
        figure()
        plot_evaluation_results(params, self.trnY, prd_trn, self.tstY, prd_tst, self.labels)
        savefig(f'images/classification/{self.file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')


    def knn_overfitting_study(self, params, K_MAX=25):
        distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
        kvalues: list[int] = [i for i in range(1, K_MAX, 2)]
        y_tst_values: list = []
        y_trn_values: list = []
        acc_metric: str = "accuracy"

        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
            clf.fit(self.trnX, self.trnY)
            prd_tst_Y: array = clf.predict(self.tstX)
            prd_trn_Y: array = clf.predict(self.trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](self.tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](self.trnY, prd_trn_Y))

        figure()
        plot_multiline_chart(
            kvalues,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"KNN overfitting study for {distance}",
            xlabel="K",
            ylabel=str(acc_metric),
            percentage=True,
        )
        savefig(f"images/{self.file_tag}_knn_overfitting.png")

