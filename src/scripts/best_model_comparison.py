import json
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from ..utils import wilcoxon_test
from ..visualization import Plotter

"""
To run this script
$ python -m vinum_analytica.scripts.best_model_comparison.py data/best.json
"""

def main(json_file):
    """
    Funzione principale che carica i dati, esegue il test di Wilcoxon e plotta i risultati.

    Parameters:
        json_file (str): Percorso al file JSON contenente i dati dei modelli.
    """
    try:
        # Carica i dati dal file JSON
        with open(json_file, 'r') as file:
            model_data = json.load(file)

        # Calcola la media delle accuracies per ciascun modello
        mean_accuracies = [np.mean(d['accuracies']) for d in model_data]

        # Stampa le accuratezze medie
        print(f"Mean accuracies: {mean_accuracies}")

        # Trova l'indice del modello con la migliore accuracy media
        best_index = np.argmax(mean_accuracies)
        best_data = model_data[best_index]

        # Esegui il test di Wilcoxon tra il miglior modello e gli altri
        wilcoxon_results = wilcoxon_test(best_index, model_data)

        # Stampa i risultati
        print(f"\nBest model index: {best_index}")
        print(f"Best model name: {best_data['model_name']}")
        print(f"Best model hyperparameters: {best_data['hyperparams']}")
        print(f"Best model mean accuracy: {mean_accuracies[best_index]}")
        print("\nWilcoxon test results:")
        for result in wilcoxon_results:
            print(f"Model: {result['model']}, p-value: {result['p_value']}")

        # Crea e mostra il violin plot per il confronto delle accuracy
        plotter = Plotter()
        name_map = {
            'rf': 'Random Forest',
            'dt': 'Decision Tree',
            'nn': 'Neural Network'
        }
        x_labels = [name_map[d['model_name']] for d in model_data]

        plotter.plot_model_accuracy_comparison(model_data, title='Model Accuracy Comparison', x_label='Models', x_ticks_labels=x_labels, label_rotation=0)

    except FileNotFoundError:
        print(f"File not found: {json_file}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {json_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing required argument: json_file")
    else:
        json_file = sys.argv[1]
        main(json_file)
