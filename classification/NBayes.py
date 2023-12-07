'''
file:       NBayes.py
'''
from auxiliar.dslabs_functions import read_train_test_from_files,CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart, plot_evaluation_results
from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show
import matplotlib.pyplot as plt

class NBayes:

    def __init__(self, file_tag, trainFilename: str, testFilename: str, target):
        self.file_tag = file_tag
        self.target   = target
        self.trnX, self.tstX, self.trnY, self.tstY, self.labels, self.vars = read_train_test_from_files(trainFilename, testFilename, target)

    def naive_Bayes_study(self, metric: str = "accuracy") -> tuple:
        estimators: dict = {
            "GaussianNB": GaussianNB(),
            "MultinomialNB": MultinomialNB(),
            "BernoulliNB": BernoulliNB(),
        }

        xvalues: list = []
        yvalues: list = []
        best_model = None
        best_params: dict = {"name": "", "metric": metric, "params": ()}
        best_performance = 0
        for clf in estimators:
            xvalues.append(clf)
            estimators[clf].fit(self.trnX, self.trnY)
            prdY: array = estimators[clf].predict(self.tstX)
            eval: float = CLASS_EVAL_METRICS[metric](self.tstY, prdY)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance: float = eval
                best_params["name"] = clf
                best_params[metric] = eval
                best_model = estimators[clf]
            yvalues.append(eval)
            # print(f'NB {clf}')
        plot_bar_chart(
            xvalues,
            yvalues,
            title=f"Naive Bayes Models ({metric})",
            ylabel=metric,
            percentage=True,
        )

        return best_model, best_params
    
    def full_metrics_NB_study(self):
        #eval_metrics = ["accuracy","recall","precision","auc","f1"]
        eval_metrics = ["accuracy","recall"]
        for eval_metric in eval_metrics:
            figure()
            best_model, params = self.naive_Bayes_study(eval_metric)
            plt.tight_layout()
            savefig(f"images/classification/{self.file_tag}_nb_{eval_metric}_study.png")
            #if eval_metric == "accuracy":
            self.performance_analysis(best_model,params)

    def performance_analysis(self, best_model, params):
        prd_trn: array = best_model.predict(self.trnX)
        prd_tst: array = best_model.predict(self.tstX)
        figure()
        plot_evaluation_results(params, self.trnY, prd_trn, self.tstY, prd_tst, self.labels)
        savefig(f'images/classification/{self.file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')