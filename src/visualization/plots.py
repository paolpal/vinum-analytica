import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from vinum_analytica.data.dataset import WineDatasetManager # type: ignore

class Plotter:
    def __init__(self):
        """
        Inizializza la classe Plotter.
        """
        sns.set_theme(style='whitegrid')  # Imposta lo stile di default per i plot

    def plot_confusion_matrix(self, cm, classes, figsize=(10, 10), cmap='Blues', fmt='.1f'):
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

    def plot_violin_accuracy_comparison(self, model_data, title='Model Accuracy Comparison', x_label='Hyperparameters',x_ticks_labels=None, label_rotation=45):
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

    def plot_bar_accuracy_comparison(self, model_data, title='Model Accuracy Comparison', x_label='Hyperparameters', x_ticks_labels=None, label_rotation=45):
        """
        Crea e mostra un grafico a barre delle accuratezze dei modelli.

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
        sns.barplot(x=model_labels, y=accuracies, color='steelblue')

        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.xticks(rotation=label_rotation)
        plt.show()

