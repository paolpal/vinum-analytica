import matplotlib.pyplot as plt
import seaborn as sns

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
        sns.countplot(x='variety', data=dataset, color='steelblue')
        plt.xticks(rotation=90)
        plt.xlabel('Variety')
        plt.ylabel('Count')
        plt.title('Count of Wine Varieties')
        plt.show()
