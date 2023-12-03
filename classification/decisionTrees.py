'''
file:       decisionTrees.py
'''
from typing import Literal
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig, show
from sklearn.tree import DecisionTreeClassifier
from auxiliar.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, read_train_test_from_files
from auxiliar.dslabs_functions import plot_evaluation_results, plot_multiline_chart
from sklearn.tree import export_graphviz
from matplotlib.pyplot import imread, imshow, axis
from subprocess import call
from sklearn.tree import plot_tree
from numpy import argsort
from auxiliar.dslabs_functions import plot_horizontal_bar_chart


class decisionTrees:

    def __init__(self, file_tag, trainFilename: str, testFilename: str, target):
        self.file_tag = file_tag
        self.target = target
        self.trnX, self.tstX, self.trnY, self.tstY, self.labels, self.vars = read_train_test_from_files(trainFilename,
                                                                                                        testFilename,
                                                                                                        target)

    def trees_study(
        self, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, d_max, lag: int = 2, metric='accuracy'
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
                print(f'DT {c} and d={d}')
            values[c] = y_tst_values
        print(f'DT best with {best_params["params"][0]} and d={best_params["params"][1]}')
        plot_multiline_chart(depths, values, title=f'DT Models ({metric})', xlabel='d', ylabel=metric, percentage=True)

        return best_model, best_params
    
    def full_metrics_DT_study(self):
        eval_metric = 'accuracy'
        figure()
        best_model, params = self.trees_study(self.trnX, self.trnY, self.tstX, self.tstY, 25, metric=eval_metric)
        savefig(f'images/classification/DT/{self.file_tag}_dt_{eval_metric}_study.png')
        #show()
        return best_model, params
    
    def performance_analysis(self, best_model, params):
        prd_trn: array = best_model.predict(self.trnX)
        prd_tst: array = best_model.predict(self.tstX)
        figure()
        plot_evaluation_results(params, self.trnY, prd_trn, self.tstY, prd_tst, self.labels)
        savefig(f'images/classification/DT/{self.file_tag}_dt_{params["name"]}_best_{params["metric"]}_eval.png')
        #show()

    def variables_importance(self, best_model, eval_metric):
        tree_filename: str = f"images/classification/DT/{self.file_tag}_dt_{eval_metric}_best_tree"
        max_depth2show = 3
        st_labels: list[str] = [str(value) for value in self.labels]

        dot_data: str = export_graphviz(
            best_model,
            out_file=tree_filename + ".dot",
            max_depth=max_depth2show,
            feature_names=self.vars,
            class_names=st_labels,
            filled=True,
            rounded=True,
            impurity=False,
            special_characters=True,
            precision=2,
        )
        # Convert to png
        call(
            ["C:\\Program Files\\Graphviz\\bin\\dot.exe", "-Tpng", tree_filename + ".dot", "-o", tree_filename + ".png", "-Gdpi=600"]
        )

        figure(figsize=(14, 6))
        imshow(imread(tree_filename + ".png"))
        axis("off")
        #show()

    def simple_tree(self, best_model):
        figure(figsize=(14, 6))
        max_depth2show = 3
        st_labels: list[str] = [str(value) for value in self.labels]
        plot_tree(
            best_model,
            max_depth=max_depth2show,
            feature_names=self.vars,
            class_names=st_labels,
            filled=True,
            rounded=True,
            impurity=False,
            precision=2,
        )
        savefig(f"images/classification/DT/{self.file_tag}_simple_tree.png")
        #show()

    def tree_importance(self, best_model, eval_metric):
        importances = best_model.feature_importances_
        indices: list[int] = argsort(importances)[::-1]
        elems: list[str] = []
        imp_values: list[float] = []
        for f in range(len(self.vars)):
            elems += [self.vars[indices[f]]]
            imp_values += [importances[indices[f]]]
            print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

        figure()
        plot_horizontal_bar_chart(
            elems,
            imp_values,
            title="Decision Tree variables importance",
            xlabel="importance",
            ylabel="variables",
            percentage=True,
        )
        savefig(f"images/classification/DT/{self.file_tag}_dt_{eval_metric}_vars_ranking.png")
        #show()

    def overfitting_study(self, eval_metric, params):
        crit: Literal["entropy", "gini"] = params["params"][0]
        d_max = 25
        depths: list[int] = [i for i in range(2, d_max + 1, 1)]
        y_tst_values: list[float] = []
        y_trn_values: list[float] = []
        acc_metric = "accuracy"
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=crit, min_impurity_decrease=0)
            clf.fit(self.trnX, self.trnY)
            prd_tst_Y: array = clf.predict(self.tstX)
            prd_trn_Y: array = clf.predict(self.trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](self.tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](self.trnY, prd_trn_Y))

        figure()
        plot_multiline_chart(
            depths,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"DT overfitting study for {crit}",
            xlabel="max_depth",
            ylabel=str(eval_metric),
            percentage=True,
        )
        savefig(f"images/classification/DT/{self.file_tag}_dt_{eval_metric}_overfitting.png")
        #show()
