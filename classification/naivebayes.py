from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show
from auxiliar.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart, plot_evaluation_results

class NaiveBayes:

    def __init__(self, file_tag, trnX, trnY, tstX, tstY):        
        self.file_tag = file_tag
        self.trnX     = trnX
        self.trnY     = trnY
        self.tstX     = tstX
        self.tstY     = tstY

    def naive_Bayes_study(self, metric: str = "accuracy") -> tuple:
        estimators: dict = {
            "GaussianNB":    GaussianNB(),
            "MultinomialNB": MultinomialNB(),
            "BernoulliNB":   BernoulliNB(),
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

        savefig(f"images/{self.file_tag}_nb_recall_study.png")
        return best_model, best_params
    

    def performance_analysis(self, best_model, params, labels):
        prd_trn: array = best_model.predict(self.trnX)
        prd_tst: array = best_model.predict(self.tstX)
        figure()
        plot_evaluation_results(params, self.trnY, prd_trn, self.tstY, prd_tst, labels)
        savefig(f'images/{self.file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
        show()