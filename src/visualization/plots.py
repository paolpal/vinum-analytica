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

    def plot_variety_difference(self, dataset_manager1, dataset_manager2):
        """
        Plot the difference in count of each variety between two datasets.

        Parameters:
            dataset_manager1 (WineDatasetManager): The first dataset manager containing a wine reviews dataset.
            dataset_manager2 (WineDatasetManager): The second dataset manager containing another wine reviews dataset.
        """
        # Ottenere i dataset
        dataset1 = dataset_manager1.get_dataset()
        dataset2 = dataset_manager2.get_dataset()

        # Conteggio varietà per ciascun dataset
        count1 = dataset1['variety'].value_counts()
        count2 = dataset2['variety'].value_counts()

        # Allineare i due conteggi sui medesimi indici (varietà di vino)
        combined_count = pd.DataFrame({'dataset1': count1, 'dataset2': count2}).fillna(0)

        # Calcolare la differenza tra i due dataset
        combined_count['difference'] = combined_count['dataset1'] - combined_count['dataset2']

        # Plot della differenza
        plt.figure(figsize=(12, 6))
        sns.barplot(x=combined_count.index, y='difference', data=combined_count, color='steelblue', order=dataset1['variety'].value_counts().index)
        plt.xticks(rotation=90)
        plt.xlabel('Variety')
        plt.ylabel('Difference in Count')
        plt.title('Difference in Wine Variety Counts Between Two Datasets')
        plt.show()

    def plot_model_accuracy_comparison(self, model_data, index_to_hyperparams, title='Model Accuracy Comparison'):
        """
        Crea e mostra un grafico a violino delle distribuzioni di accuratezza per ciascun modello.
        
        Parameters:
            model_data (list): Lista di dizionari contenenti i dati dei modelli.
            index_to_hyperparams (dict): Mappa degli indici ai rispettivi iperparametri.
        """
        model_labels = []
        accuracies = []

        for i, d in enumerate(model_data):
            index_label = f"Index {i}"
            model_labels.extend([index_label] * len(d['accuracies']))
            accuracies.extend(d['accuracies'])

        plt.figure(figsize=(12, 8))
        sns.violinplot(x=model_labels, y=accuracies, inner="quartile", color='steelblue')

        plt.xlabel('Hyperparameters')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.show()

        # Print the table of hyperparameters
        print("\nHyperparameters Table:")
        for i, hyperparams in index_to_hyperparams.items():
            print(f"Index {i}: {hyperparams}")


