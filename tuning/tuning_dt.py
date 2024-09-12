import itertools
import json
import random
from tqdm import tqdm
from vinum_analytica.data.dataset import WineDatasetManager  # type: ignore
from vinum_analytica.models.tree_model import TreeModel  # type: ignore

# Definizione della griglia di iperparametri per la rete neurale
dt_param_grid = {
    'criterion': ['gini', 'log_loss'],                   # Funzione di qualità da ottimizzare
    'min_impurity_decrease': [0.0, 1e-8, 1e-10, 1e-12],              # Soglia per la riduzione dell'impurità
    'max_depth': [150, 200, None]                                       # Profondità massima dell'albero
}

# Ottieni le chiavi e i valori degli iperparametri
param_keys = list(dt_param_grid.keys())
param_values = list(dt_param_grid.values())

# Genera tutte le combinazioni degli iperparametri
param_combinations = list(itertools.product(*param_values))

# Carica il dataset
dataset = WineDatasetManager()
dataset.load('./data/processed/train_classification.csv')

# Campiona 8 combinazioni di iperparametri casuali
param_combinations = random.sample(param_combinations, 8)

results = []
printable_combinations = []
for combination in param_combinations:
    printable_combinations.append(dict(zip(param_keys, combination)))
# salva gli iper parametri selezionati su un file
with open('./results/hyper_dt.json', 'w') as f:
    json.dump(printable_combinations, f, indent=4)

# Ciclo sulle combinazioni degli iperparametri
for combination in param_combinations:
    # Crea un dizionario con i nomi degli iperparametri e i valori della combinazione corrente
    params = dict(zip(param_keys, combination))

    # Ciclo sui fold
    accuracies = []
    print(f'Training model with hyperparameters: {params}')
    dataset.reset_folds()
    for fold in tqdm(range(n_folds := 6)):
        # Suddividi il dataset in training e validation per questo fold
        train, valid = dataset.fold(fold, n_folds=n_folds)
        vec = train.vectorize()
        train.resample()

        # Inizializza il modello con i parametri attuali
        dt = TreeModel(
            vectorizer=vec,
            criterion=params['criterion'],
            min_impurity_decrease=params['min_impurity_decrease'],
            max_depth=params['max_depth']
        )

        # Addestra il modello per il numero di epoche specificato
        dt.train(train)

        # Valuta il modello sui dati di validazione
        accuracy = dt.evaluate(valid)
        accuracies.append(accuracy)
    
    # crea un json con i parametri e l'accuratezza
    result = {
        'hyperparams': params,
        'accuracies': accuracies
    }
    results.append(result)
    with open('./results/dt_tuning.json', 'w') as f:
        json.dump(results, f, indent=4)