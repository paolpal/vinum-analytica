import itertools
import json
import random
from vinum_analytica.data.dataset import WineDatasetManager  # type: ignore
from vinum_analytica.models.nn_model import NeuralNetworkModel  # type: ignore

# Definizione della griglia di iperparametri per la rete neurale
nn_param_grid = {
    'hidden_size': [32, 64, 128, 128],  # Dimensioni del livello nascosto
    'epochs': [5, 10, 15],         # Numero di epoche
    'lr': [0.001, 0.01, 0.1]       # Learning rate
}

# Ottieni le chiavi e i valori degli iperparametri
param_keys = list(nn_param_grid.keys())
param_values = list(nn_param_grid.values())

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
with open('./results/hyper_nn.json', 'w') as f:
    json.dump(printable_combinations, f, indent=4)

# Ciclo sulle combinazioni degli iperparametri
for combination in param_combinations:
    # Crea un dizionario con i nomi degli iperparametri e i valori della combinazione corrente
    params = dict(zip(param_keys, combination))

    # Ciclo sui fold
    accuracies = []
    print(f'Training model with hyperparameters: {params}')
    dataset.reset_folds()
    for fold in range(n_folds := 6):
        # Suddividi il dataset in training e validation per questo fold
        train, valid = dataset.fold(fold, n_folds=n_folds)
        vec = train.vectorize()
        train.oversample()

        # Inizializza il modello con i parametri attuali
        nn = NeuralNetworkModel(
            input_size=vec.get_feature_names_out().shape[0],
            hidden_size=params['hidden_size'],
            output_size=len(train.classes()),
            epochs=params['epochs'],
            lr=params['lr'],
            vectorizer=vec
        )

        # Addestra il modello per il numero di epoche specificato
        nn.train(train, verbose=False)

        # Valuta il modello sui dati di validazione
        accuracy = nn.evaluate(valid)
        accuracies.append(accuracy)
    
    # crea un json con i parametri e l'accuratezza
    result = {
        'hyperparams': params,
        'accuracies': accuracies
    }
    results.append(result)
    with open('./results/nn_tuning.json', 'w') as f:
        json.dump(results, f, indent=4)