import itertools
from vinum_analytica.data.dataset import WineDatasetManager  # type: ignore
from vinum_analytica.models.nn_model import NeuralNetworkModel  # type: ignore

# Definizione della griglia di iperparametri per la rete neurale
nn_param_grid = {
    'hidden_size': [32, 64, 128],  # Dimensioni del livello nascosto
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

# Definisci il numero di fold per il KFold
n_folds = 6

# Ciclo sulle combinazioni degli iperparametri
for combination in param_combinations:
    # Crea un dizionario con i nomi degli iperparametri e i valori della combinazione corrente
    params = dict(zip(param_keys, combination))

    # Ciclo sui fold
    for fold in range(n_folds):
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
        nn.train(train)

        # Valuta il modello sui dati di validazione
        accuracy = nn.evaluate(valid)
        print(f'Combinazione: {params}, Fold {fold}, Accuracy: {accuracy}')

        # Salva il modello addestrato
        nn.save(f'./models/nn_model_comb_{combination}_fold_{fold}.pkl')
