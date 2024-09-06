import itertools
import json
import random
from tqdm import tqdm
from vinum_analytica.data.dataset import WineDatasetManager  # type: ignore
from vinum_analytica.models.forest_model import RandomForestModel  # type: ignore

# Definizione della griglia di iperparametri per la rete neurale
rf_param_grid = {
    'n_estimators': [50, 100, 150],                     # Numero di alberi nella foresta
    'criterion': ['gini', 'entropy'],                   # Funzione di qualità da ottimizzare
    'min_impurity_decrease': [0.0, 1e-2, 1e-4, 1e-8]    # Soglia per la riduzione dell'impurità
}

# Ottieni le chiavi e i valori degli iperparametri
param_keys = list(rf_param_grid.keys())
param_values = list(rf_param_grid.values())

# Genera tutte le combinazioni degli iperparametri
param_combinations = list(itertools.product(*param_values))

# Carica il dataset
dataset = WineDatasetManager()
dataset.load('./data/processed/train.csv')

# Campiona 8 combinazioni di iperparametri casuali
param_combinations = random.sample(param_combinations, 8)

results = []
printable_combinations = []
for combination in param_combinations:
    printable_combinations.append(dict(zip(param_keys, combination)))
# salva gli iper parametri selezionati su un file
with open('./results/hyper_rf.json', 'w') as f:
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
        train.oversample()

        # Inizializza il modello con i parametri attuali
        rf = RandomForestModel(
            vectorizer=vec,
            criterion=params['criterion'],
            min_impurity_decrease=params['min_impurity_decrease'],
            n_estimators=params['n_estimators']
        )

        # Addestra il modello per il numero di epoche specificato
        rf.train(train)

        # Valuta il modello sui dati di validazione
        accuracy = rf.evaluate(valid)
        accuracies.append(accuracy)
    
    # crea un json con i parametri e l'accuratezza
    result = {
        'hyperparams': params,
        'accuracies': accuracies
    }
    results.append(result)
    with open('./results/rf_tuning.json', 'w') as f:
        json.dump(results, f, indent=4)