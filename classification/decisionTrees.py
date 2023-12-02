'''
file:       decisionTrees.py
'''
from typing import Literal
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig, show
from sklearn.tree import DecisionTreeClassifier
from auxiliar.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, read_train_test_from_files
from auxiliar.dslabs_functions import plot_evaluation_results, plot_multiline_chart


class decisionTrees:

    def __init__(self, file_tag, trainFilename: str, testFilename: str, target):
        self.file_tag = file_tag
        self.target = target
        self.trnX, self.tstX, self.trnY, self.tstY, self.labels, self.vars = read_train_test_from_files(trainFilename,
                                                                                                        testFilename,
                                                                                                        target)

    def trees_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, d_max: int=10, lag:int=2, metric='accuracy'
        ) -> tuple:
        criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
        depths: list[int] = [i for i in range(2, d_max+1, lag)]

        best_model: DecisionTreeClassifier | None = None
        best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
        best_performance: float = 0.0

        values: dict = {}
        for c in criteria:
            y_tst_values: list[float] = []
            for d in depths:
                clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params['params'] = (c, d)
                    best_model = clf
                # print(f'DT {c} and d={d}')
            values[c] = y_tst_values
        print(f'DT best with {best_params["params"][0]} and d={best_params["params"][1]}')
        plot_multiline_chart(depths, values, title=f'DT Models ({metric})', xlabel='d', ylabel=metric, percentage=True)

        return best_model, best_params
    
    def full_metrics_DT_study(self):
        eval_metric = 'accuracy'
        figure()
        best_model, params = trees_study(self.trnX, self.trnY, self.tstX, self.tstY, d_max=25, metric=eval_metric)
        savefig(f'images/{self.file_tag}_dt_{eval_metric}_study.png')
        show()
        return best_model, params
    
    def performance_analysis(self, best_model, params):
        prd_trn: array = best_model.predict(self.trnX)
        prd_tst: array = best_model.predict(self.tstX)
        figure()
        plot_evaluation_results(params, self.trnY, prd_trn, self.tstY, prd_tst, self.labels)
        savefig(f'images/classification/{self.file_tag}_dt_{params["name"]}_best_{params["metric"]}_eval.png')
        show()