from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from models.sklearnModels import SklearnModel


class ModelBuilder(object):
    
    def __init__(self, model_params):
        self.model_params = model_params


    def rf_model_builder(self, params):
        rf_model = RandomForestClassifier(**params)
        return SklearnModel(model=rf_model)


    def rf_model_optimizer(self, params):
        optimizer = GridHyperparamOpt(rf_model_builder)
        best_rf, best_hyperparams, all_results = optimizer.hyperparam_search(params, train_dataset,
                                                                         valid_dataset, Metric(roc_auc_score))


    def svm_model_builder(self, params):
        svm = SVC(**params)
        return SklearnModel(model=svm)


    def return_initialized_models(self):
        initialized_models = []
        for model in self.model_params:
            if(model['name'] == "RandomForestClassifier"):
                if(model['type'] == 'default'):
                    initialized_models.append(self.rf_model_builder(params=None))
                elif(model['type'] == 'params'):
                    initialized_models.append(self.rf_model_builder(params=model['params']))
            if(model['name'] == "SVM"):
                if(model['type'] == 'default'):
                    initialized_models.append(self.svm_model_builder(params=None))
                elif(model['type'] == 'params'):
                    initialized_models.append(self.svm_model_builder(params=model['params']))
        return initialized_models
        