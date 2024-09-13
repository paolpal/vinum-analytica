import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from vinum_analytica.data.dataset import WineDatasetManager # type: ignore

class Plotter:
    def __init__(self):
        """
        Inizializza la classe Plotter.
        """
        sns.set_theme(style='whitegrid')  # Imposta lo stile di default per i plot

    def plot_confusion_matrix(self, cm, classes, figsize=(12, 12), cmap='Blues', fmt='.1f'):
        """
        Plot della confusion matrix usando seaborn e matplotlib.

        Parameters:
            cm (array-like): La confusion matrix da plottare.
            classes (list): Lista delle classi.
            figsize (tuple): Dimensioni della figura (default: (10, 10)).
            cmap (str): Mappa di colori per la heatmap (default: 'Blues').
            fmt (str): Formattazione dei valori nella matrice (default: '.1f').
        """
        # Crea il plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=classes, yticklabels=classes, cmap=cmap)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_variety_count(self, dataset_manager):
        """
        Plot the count of each variety in the dataset.

        Parameters:
            dataset_manager (WineDatasetManager): The dataset manager containing the wine reviews dataset.
        """
        dataset = dataset_manager.get_dataset()
        plt.figure(figsize=(12, 6))
        sns.countplot(x='variety', data=dataset, color='steelblue', order=dataset['variety'].value_counts().index)
        plt.xticks(rotation=90)
        plt.xlabel('Variety')
        plt.ylabel('Count')
        plt.title('Count of Wine Varieties')
        plt.show()

    def plot_violin_accuracy_comparison(self, model_data, title='Model Accuracy Comparison', x_label='Hyperparameters', x_ticks_labels=None, label_rotation=45):
        """
        Crea e mostra un grafico a violino delle distribuzioni di accuratezza per ciascun modello.
        
        Parameters:
            model_data (list): Lista di dizionari contenenti i dati dei modelli.
            title (str): Titolo del grafico (default: 'Model Accuracy Comparison').
            x_label (str): Etichetta dell'asse x (default: 'Hyperparameters').
            x_ticks_labels (list): Etichette per l'asse x (default: None).
            label_rotation (int): Rotazione delle etichette sull'asse x (default: 45).
        """
        if x_ticks_labels is not None:
            assert len(x_ticks_labels) == len(model_data), "Number of x_labels must match number of models"
        model_labels = []
        accuracies = []

        for i, d in enumerate(model_data):
            index_label = x_ticks_labels[i] if x_ticks_labels is not None else f'Model {i}'
            model_labels.extend([index_label] * len(d['accuracies']))
            accuracies.extend(d['accuracies'])

        plt.figure(figsize=(12, 8))
        sns.violinplot(x=model_labels, y=accuracies, inner="quartile", color='steelblue')

        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.xticks(rotation=label_rotation)
        plt.show()

    def plot_validation_test_accuracy_comparison(self, model_data, title='', x_label='Models', x_ticks_labels=None, bar_width=0.8):
        """
        Crea e mostra un grafico a barre per confrontare le accuratezze dei modelli su validation e test.

        Parameters:
            model_data (list): Lista di dizionari contenenti i dati dei modelli. Ogni dizionario deve contenere:
                - 'model': nome del modello (es. 'Decision Tree')
                - 'validation_accuracy': accuratezza su validation
                - 'test_accuracy': accuratezza su test
        """
        # Prepara i dati per il grafico
        models = [d['model'] for d in model_data]
        validation_accuracies = [d['validation_accuracy'] for d in model_data]
        test_accuracies = [d['test_accuracy'] for d in model_data]

        # Crea un DataFrame per i dati
        data = {
            'Model': models * 2,
            'Accuracy': validation_accuracies + test_accuracies,
            'Type': ['Validation'] * len(models) + ['Test'] * len(models)
        }
        df = pd.DataFrame(data)

        # Creare il bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Accuracy', hue='Type', data=df, width=bar_width)

        # Personalizzare il grafico
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()

        # Mostrare il grafico
        plt.show()

    def plot_regressor_comparison(self, y, y_tree_pred, y_neural_pred):
        """
        Crea un grafico a dispersione per confrontare le predizioni del Tree Model e del Neural Network con i valori reali,
        utilizzando seaborn per la visualizzazione.
        
        Parameters:
            y (array-like): Valori reali.
            y_tree_pred (array-like): Predizioni del modello ad albero.
            y_neural_pred (array-like): Predizioni del modello di rete neurale.
        """

        y = np.array(y).ravel()
        y_tree_pred = np.array(y_tree_pred).ravel()
        y_neural_pred = np.array(y_neural_pred).ravel()

        plt.figure(figsize=(10, 10))

        # Creazione di un DataFrame per facilitare il plotting con seaborn
        data = pd.DataFrame({
            'Actual Price': y,
            'Tree Model Prediction': y_tree_pred,
            'Neural Network Prediction': y_neural_pred
        })

        # Scatter plot per le predizioni del modello ad albero
        sns.scatterplot(x='Actual Price', y='Tree Model Prediction', data=data, label='Tree Model')

        # Scatter plot per le predizioni del modello di rete neurale
        sns.scatterplot(x='Actual Price', y='Neural Network Prediction', data=data, label='Neural Network Model')

        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Model Predictions vs Actual')
        plt.legend()

        # Set the limits for x and y axis to be the same
        max_value = max(y)
        plt.xlim(0, max_value)
        plt.ylim(0, max_value)

        plt.show()
