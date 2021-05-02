from parameterOptimization.HyperparameterOpt import GridHyperparamOpt
from splitters.splitters import SingletaskStratifiedSplitter
from metrics.Metrics import Metric
from metrics.metricsFunctions import roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from models.sklearnModels import SklearnModel
from autoML.default_params import rf_model_builder as rfmb

class OptimizationSelector(object):

    def __init__(self, models, dataset):
       self.models = models
       self.partitioned_dataset = self.prepare_dataset(dataset)
       

    def prepare_dataset(self, dataset):
        splitter = SingletaskStratifiedSplitter()
        train_dataset, valid_dataset, test_dataset = \
            splitter.train_valid_test_split(dataset=dataset, 
                                            frac_train=0.6, 
                                            frac_valid=0.2, 
                                            frac_test=0.2)

        return train_dataset, valid_dataset, test_dataset

    def select_models(self):
        best_models = []
        for opt_model in self.models:
            best = self.select_best(opt_model)
            best_models.append(best)
        return best_models

    def select_best(self, opt_model):

        optimizer = GridHyperparamOpt(rfmb)
        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            opt_model['params'],
            self.partitioned_dataset[0],
            self.partitioned_dataset[1],
            Metric(roc_auc_score))
        return best_model
