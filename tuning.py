nn_param_grid = {
    'hidden_size': [32, 64, 128],  # Esempio di dimensioni del livello nascosto
    'epochs': [5, 10, 15],         # Numero di epoche
    'lr': [0.001, 0.01, 0.1]       # Learning rate
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],                     # Numero di alberi nella foresta
    'criterion': ['gini', 'entropy'],                   # Funzione di qualità da ottimizzare
    'min_impurity_decrease': [0.0, 1e-2, 1e-4, 1e-8]    # Soglia per la riduzione dell'impurità
}

dt_param_grid = {
    'criterion': ['gini', 'entropy'],                   # Funzione di qualità da ottimizzare
    'min_impurity_decrease': [0.0, 1e-2, 1e-4, 1e-8]    # Soglia per la riduzione dell'impurità
}
