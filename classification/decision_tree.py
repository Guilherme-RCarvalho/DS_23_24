from typing import Literal
from numpy import array, argsort
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show
from sklearn.tree import DecisionTreeClassifier
from auxiliar.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart, plot_horizontal_bar_chart
from auxiliar.dslabs_functions import read_train_test_from_files, plot_evaluation_results
from matplotlib.pyplot import figure, savefig, show, imread, imshow, axis
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.tree import plot_tree

class DecisionTree:

    def __init__(self, file_tag, trainFilename: str, testFilename: str, target):
        self.file_tag = file_tag
        self.target   = target
        self.trnX, self.tstX, self.trnY, self.tstY, self.labels, self.vars = read_train_test_from_files(trainFilename, testFilename, target)

    def trees_study(self, d_max: int=10, lag:int=2, metric='accuracy') -> tuple:        
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
                clf.fit(self.trnX, self.trnY)
                prdY: array = clf.predict(self.tstX)
                eval: float = CLASS_EVAL_METRICS[metric](self.tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params['params'] = (c, d)
                    best_model = clf
                # print(f'DT {c} and d={d}')
            values[c] = y_tst_values
        print(f'DT best with {best_params['params'][0]} and d={best_params['params'][1]}')
        plot_multiline_chart(depths, values, title=f'DT Models ({metric})', xlabel='d', ylabel=metric, percentage=True)

        return best_model, best_params
    

    def full_metrics_decision_tree_study(self, d_max, lag):
        eval_metrics = ["accuracy","recall","precision","auc","f1"]
        for eval_metric in eval_metrics:
            figure()
            best_model, params = self.trees_study(d_max, lag, metric=eval_metric)
            plt.tight_layout()
            savefig(f"images/classification/{self.file_tag}_dt_{eval_metric}_study.png")


    def performance_analysis(self, best_model, params):
        prd_trn: array = best_model.predict(self.trnX)
        prd_tst: array = best_model.predict(self.tstX)
        figure()
        plot_evaluation_results(params, self.trnY, prd_trn, self.tstY, prd_tst, self.labels)
        savefig(f'images/classification/{self.file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')


    def visualize_tree(self, best_model, params, max_depth2show=3):
        tree_filename = f"images/classification/{self.file_tag}_dt_{params['metric']}_best_tree"
        st_labels = [str(value) for value in self.labels]

        dot_data = export_graphviz(
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
            ["dot", "-Tpng", tree_filename + ".dot", "-o", tree_filename + ".png", "-Gdpi=600"]
        )

        figure(figsize=(14, 6))
        imshow(imread(tree_filename + ".png"))
        axis("off")
        #show()

    def visualize_tree_simplified(self, best_model, params, max_depth2show=3):
        tree_filename = f"images/classification/{self.file_tag}_dt_{params['metric']}_best_tree_simplified"
        st_labels = [str(value) for value in self.labels]

        figure(figsize=(14, 6))
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
        savefig(tree_filename + ".png")
        #show()

    def visualize_variable_importance(self, best_model, eval_metric): # passei eval_metric
        importances = best_model.feature_importances_
        indices = argsort(importances)[::-1]
        elems = []
        imp_values = []
        for f in range(len(self.vars)):
            elems.append(self.vars[indices[f]])
            imp_values.append(importances[indices[f]])
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
        savefig(f"images/classification/{self.file_tag}_dt_{eval_metric}_vars_ranking.png")
        show()


    def dt_overfitting_study(self, params, eval_metric, d_max=25): # passei eval_metric
        crit: Literal["entropy", "gini"] = params["params"][0]
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
            ylabel=str(eval_metric), # tinha eval_metric pus acc_metric
            percentage=True,
        )
        savefig(f"images/{self.file_tag}_dt_{eval_metric}_overfitting.png")