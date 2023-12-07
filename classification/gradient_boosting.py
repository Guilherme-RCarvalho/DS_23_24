from subprocess import call
from numpy import argsort, array, ndarray, std
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, subplots, figure, savefig, show, axis, imread
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_graphviz, plot_tree
from auxiliar.dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from auxiliar.dslabs_functions import HEIGHT, plot_evaluation_results, plot_horizontal_bar_chart, plot_multiline_chart

class GradientBoosting:

    def __init__(self, file_tag, trainFilename: str, testFilename: str, target):
        self.file_tag = file_tag
        self.target   = target
        self.trnX, self.tstX, self.trnY, self.tstY, self.labels, self.vars = read_train_test_from_files(trainFilename, testFilename, target)

    def gradient_boosting_study(self, nr_max_trees: int = 2500, lag: int = 500, metric: str = "accuracy") -> tuple[GradientBoostingClassifier | None, dict]:
        n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
        max_depths: list[int] = [2, 5, 7]
        learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

        best_model: GradientBoostingClassifier | None = None
        best_params: dict = {"name": "GB", "metric": metric, "params": ()}
        best_performance: float = 0.0

        values: dict = {}
        cols: int = len(max_depths)
        _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
        for i in range(len(max_depths)):
            d: int = max_depths[i]
            values = {}
            for lr in learning_rates:
                y_tst_values: list[float] = []
                for n in n_estimators:
                    clf = GradientBoostingClassifier(
                        n_estimators=n, max_depth=d, learning_rate=lr
                    )
                    clf.fit(self.trnX, self.trnY)
                    prdY: array = clf.predict(self.tstX)
                    eval: float = CLASS_EVAL_METRICS[metric](self.tstY, prdY)
                    y_tst_values.append(eval)
                    if eval - best_performance > DELTA_IMPROVE:
                        best_performance = eval
                        best_params["params"] = (d, lr, n)
                        best_model = clf
                    # print(f'GB d={d} lr={lr} n={n}')
                values[lr] = y_tst_values
            plot_multiline_chart(
                n_estimators,
                values,
                ax=axs[0, i],
                title=f"Gradient Boosting with max_depth={d}",
                xlabel="nr estimators",
                ylabel=metric,
                percentage=True,
            )
        print(
            f'GB best for {best_params["params"][2]} trees (d={best_params["params"][0]} and lr={best_params["params"][1]}'
        )

        return best_model, best_params
    

    def performance_analysis(self, best_model, params):
        prd_trn: array = best_model.predict(self.trnX)
        prd_tst: array = best_model.predict(self.tstX)
        figure()
        plot_evaluation_results(params, self.trnY, prd_trn, self.tstY, prd_tst, self.labels)
        savefig(f'images/classification/{self.file_tag}_gb_{params["name"]}_best_{params["metric"]}_eval.png')

    

    def full_metrics_gradient_boosting_study(self, nr_max_trees, lag):
        #eval_metrics = ["accuracy","recall","precision","auc","f1"]
        eval_metrics = ["accuracy"]
        for eval_metric in eval_metrics:
            figure()
            print(f"new metric: {eval_metric}")
            best_model, params = self.gradient_boosting_study(nr_max_trees, lag, eval_metric)
            plt.tight_layout()
            savefig(f"images/classification/{self.file_tag}_gb_{nr_max_trees}_{eval_metric}_study.png")
            if eval_metric == "accuracy":
                self.performance_analysis(best_model,params)
                self.gb_overfitting_study(params)                                
                self.visualize_variable_importance(best_model,eval_metric)                
                self.visualize_tree(best_model,params) 

        
    def visualize_variable_importance(self, best_model, eval_metric): # passei eval_metric
        trees_importances: list[float] = []
        for lst_trees in best_model.estimators_:
            for tree in lst_trees:
                trees_importances.append(tree.feature_importances_)

        stdevs: list[float] = list(std(trees_importances, axis=0))
        importances = best_model.feature_importances_
        indices: list[int] = argsort(importances)[::-1]
        elems: list[str] = []
        imp_values: list[float] = []
        for f in range(len(vars)):
            elems += [vars[indices[f]]]
            imp_values.append(importances[indices[f]])
            print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

        figure()
        plot_horizontal_bar_chart(
            elems,
            imp_values,
            error=stdevs,
            title="GB variables importance",
            xlabel="importance",
            ylabel="variables",
            percentage=True,
        )
        savefig(f"images/classification/{self.file_tag}_gb_{eval_metric}_vars_ranking.png")


    def gb_overfitting_study(self,  params):
        d_max: int = params["params"][0]
        lr: float = params["params"][1]
        nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

        y_tst_values: list[float] = []
        y_trn_values: list[float] = []
        acc_metric: str = "accuracy"

        for n in nr_estimators:
            clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr)
            clf.fit(self.trnX, self.trnY)
            prd_tst_Y: array = clf.predict(self.tstX)
            prd_trn_Y: array = clf.predict(self.trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](self.tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](self.trnY, prd_trn_Y))

        figure()
        plot_multiline_chart(
            nr_estimators,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"GB overfitting study for d={d_max} and lr={lr}",
            xlabel="nr_estimators",
            ylabel=str(acc_metric), # eval_metric
            percentage=True,
        )
        savefig(f"images/classification/{self.file_tag}_gb_{acc_metric}_overfitting.png")



    # Iguais aos do Decision Tree
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