from auxiliar.dslabs_functions import HEIGHT, read_train_test_from_files, plot_evaluation_results, plot_multiline_chart
from typing import Literal
import matplotlib.pyplot as plt
from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.neural_network import MLPClassifier
from auxiliar.dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)


class MultiLayerPerceptron:

    def __init__(self, file_tag, trainFilename: str, testFilename: str, target):
        self.file_tag = file_tag
        self.target   = target
        self.trnX, self.tstX, self.trnY, self.tstY, self.labels, self.vars = read_train_test_from_files(trainFilename, testFilename, target)


    def mlp_study(self, nr_max_iterations: int = 2500, lag: int = 500, metric: str = "accuracy") -> tuple[MLPClassifier | None, dict]:
        nr_iterations: list[int] = [lag] + [
            i for i in range(2 * lag, nr_max_iterations + 1, lag)
        ]

        lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
            "constant",
            "invscaling",
            "adaptive",
        ]  # only used if optimizer='sgd'
        learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

        best_model: MLPClassifier | None = None
        best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
        best_performance: float = 0.0

        values: dict = {}
        _, axs = subplots(
            1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
        )
        for i in range(len(lr_types)):
            type: str = lr_types[i]
            values = {}
            for lr in learning_rates:
                warm_start: bool = False
                y_tst_values: list[float] = []
                for j in range(len(nr_iterations)):
                    clf = MLPClassifier(
                        learning_rate=type,
                        learning_rate_init=lr,
                        max_iter=lag,
                        warm_start=warm_start,
                        activation="logistic",
                        solver="sgd",
                        verbose=False,
                    )
                    clf.fit(self.trnX, self.trnY)
                    prdY: array = clf.predict(self.tstX)
                    eval: float = CLASS_EVAL_METRICS[metric](self.tstY, prdY)
                    y_tst_values.append(eval)
                    warm_start = True
                    if eval - best_performance > DELTA_IMPROVE:
                        best_performance = eval
                        best_params["params"] = (type, lr, nr_iterations[j])
                        best_model = clf
                    print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
                values[lr] = y_tst_values
            plot_multiline_chart(
                nr_iterations,
                values,
                ax=axs[0, i],
                title=f"MLP with {type}",
                xlabel="nr iterations",
                ylabel=metric,
                percentage=True,
            )
        print(
            f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
        )

        return best_model, best_params    


    def full_metrics_mlp_study(self, nr_max_iterations, lag):
        #eval_metrics = ["accuracy","recall","precision","auc","f1"]
        eval_metrics = ["accuracy","recall"]
        for eval_metric in eval_metrics:
            figure()
            print(f"new metric: {eval_metric}")
            best_model, params = self.mlp_study(nr_max_iterations, lag, eval_metric)
            plt.tight_layout()
            savefig(f"images/classification/{self.file_tag}_mlp_{nr_max_iterations}_{eval_metric}_study.png")
            self.performance_analysis(best_model,params)
            if eval_metric == "accuracy":
                self.mlp_overfitting_study(params)                

    
    def performance_analysis(self, best_model, params):
        prd_trn: array = best_model.predict(self.trnX)
        prd_tst: array = best_model.predict(self.tstX)
        figure()
        plot_evaluation_results(params, self.trnY, prd_trn, self.tstY, prd_tst, self.labels)
        savefig(f'images/classification/{self.file_tag}_mlp_{params["name"]}_best_{params["metric"]}_eval.png')


    def mlp_overfitting_study(self,  params):
        lr_type: Literal["constant", "invscaling", "adaptive"] = params["params"][0]
        lr: float = params["params"][1]
        nr_iterations: list[int] = [i for i in range(100, 1001, 100)]

        y_tst_values: list[float] = []
        y_trn_values: list[float] = []
        acc_metric = "accuracy"

        warm_start: bool = False
        for n in nr_iterations:
            clf = MLPClassifier(
                warm_start=warm_start,
                learning_rate=lr_type,
                learning_rate_init=lr,
                max_iter=n,
                activation="logistic",
                solver="sgd",
                verbose=False,
            )
            clf.fit(self.trnX, self.trnY)
            prd_tst_Y: array = clf.predict(self.tstX)
            prd_trn_Y: array = clf.predict(self.trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](self.tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](self.trnY, prd_trn_Y))
            warm_start = True

        figure()
        plot_multiline_chart(
            nr_iterations,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
            xlabel="nr_iterations",
            ylabel=str(acc_metric), # eval_metric
            percentage=True,
        )
        savefig(f"images/{self.file_tag}_mlp_{acc_metric}_overfitting.png")
