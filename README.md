# Projeto Pŕatico - Aprendizagem Automática II 20/21
## Contributors:
- [Ângelo Sousa](https://github.com/AngeloACSousa)
- [David Neto](https://github.com/DivadotenGit)
- [Diogo Pereira](https://github.com/dpereira7)
- [Pedro Delgado](https://github.com/PedroPDelgado)


# AutoML
## Desenvolvimento de uma abordagem de AutoML integrada numa ferramenta de machine learning existente (DeepMol).

## **Nota:** 
Este repositório apenas contém o **código desenvolvido pelo grupo**, e não contém código do repositório **DeepMol**.


# Tarefas já desenvolvidas
- Carregamento dos Datasets
- Escolha de featurization
- Escolha de Modelos
- Definição de parâmetros dos Modelos e de featurization
- Definição de Métricas de comparação de Modelos
- Otimização de Modelos

### Modelos Suportados
- RandomForestClassifier
- SVM

## Formato dos Ficheiros de Input

```json
{
    "load": {
        "dataset": "preprocessed_dataset_wfoodb.csv",
        "mols": "Smiles",
        "labels_fields": "Class",
        "id_field": "ID"
    },
    "featurizer": {
        "name" : "morgan",
        "type" : "default",
        "params":{}
    },
    "models": [
        {"name": "RandomForestClassifier", "params": {"n_estimators": [5,25,50,100],
                                                      "criterion": ["entropy","gini"],
                                                      "max_features": ["auto", "sqrt", "log2", "None"]},
        {"name": "SVM", "params": {}}
    ],
    "metrics": ["roc_auc_score", "precision_score", "accuracy_score"]
}

``` 
## **Nota:**
Quando no lugar de um parâmetro aparece uma lista, executa-se a otimização dos parâmetros do modelo.

```python
from autoML.AutoML import AutoML

auto_ml = AutoML('teste.json')
auto_ml.execute()
```

[Notebook]()
