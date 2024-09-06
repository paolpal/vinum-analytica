import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix

from .data_preprocessing import TextPreprocessor

class WineDatasetManager:
    def __init__(self, X: pd.Series = None, y: pd.Series = None):
        """
        Inizializza la classe con i dati opzionali X e y.
        
        Parameters:
            X (pd.Series): Serie contenente le caratteristiche (descrizioni) dei vini.
            y (pd.Series): Serie contenente le etichette (varietà) dei vini.
        """
        self.X = X
        self.y = y
        self.folds = None
        self.vectorizer = None

    @classmethod
    def read_csv(cls, csv_path: str, text_column: str = 'description', label_column: str = 'variety'):
        """
        Carica i dati da un file CSV.

        Parameters:
            csv_path (str): Percorso al file CSV.
            text_column (str): Nome della colonna contenente il testo (default 'description').
            label_column (str): Nome della colonna contenente le etichette (default 'variety').

        Returns:
            WineDatasetManager: Un'istanza di WineDatasetManager con i dati caricati.
        """
        df = pd.read_csv(csv_path)

        df = df[~df['variety'].str.contains('Blend', case=False, na=False)]  # Rimuovere i blend
        variety_counts = df['variety'].value_counts()

        support_threshold = 0.006*len(df)  # Set the support threshold as desired

        selected_varieties = variety_counts[variety_counts >= support_threshold]

        filtered_df = df[df['variety'].isin(selected_varieties.index)]
        df = filtered_df

        X = df[text_column]
        y = df[label_column]
        return cls(X, y)
    
    def save(self, csv_path: str):
        """
        Salva i dati in un file CSV.

        Parameters:
            csv_path (str): Percorso al file CSV.
        """
        assert isinstance(self.X, pd.Series), "X must be a pandas Series."
        df = pd.DataFrame({'description': self.X, 'variety': self.y})
        df.to_csv(csv_path, index=False)
    
    def load(self, csv_path: str):
        """
        Carica i dati da un file CSV.

        Parameters:
            csv_path (str): Percorso al file CSV.
        """
        df = pd.read_csv(csv_path)
        self.X = df['description']
        self.y = df['variety']

    def get_x(self) -> pd.Series:
        """
        Restituisce le caratteristiche (descrizioni dei vini).
        
        Returns:
            pd.Series: Le descrizioni dei vini.
        """
        return self.X
    
    def get_y(self) -> pd.Series:
        """
        Restituisce le etichette (varietà di vini).
        
        Returns:
            pd.Series: Le varietà di vini.
        """
        return self.y
    
    def preprocess(self):
        """
        Pre-elabora le descrizioni dei vini.
        """
        preprocessor = TextPreprocessor()
        self.X = self.X.apply(preprocessor.preprocess)
    
    def vectorize(self, vectorizer: TfidfVectorizer | None = None) -> TfidfVectorizer:
        """
        Vettorizza le descrizioni dei vini e aggiorna X con i dati vettorizzati.
        """
        if vectorizer is not None:
            self.vectorizer = vectorizer
            self.X = self.vectorizer.transform(self.X)
        else:
            self.vectorizer = TfidfVectorizer(strip_accents='ascii')
            self.X = self.vectorizer.fit_transform(self.X)
        return self.vectorizer

    def oversample(self, max_samples : int = 200000):
        """
        Esegue l'oversampling del training set usando SMOTE.
        """
        class_counts = self.y.value_counts()
        num_classes = len(class_counts)
        max_samples_per_class = max_samples // num_classes
        class_dict = {cls: max_samples_per_class if count < max_samples_per_class else count for cls, count in class_counts.items()}

        smote = SMOTE(sampling_strategy=class_dict)
        X_res, y_res = smote.fit_resample(self.X, self.y)

        self.X, self.y = X_res, y_res

    def reset_folds(self):
        """
        Resetta gli indici dei fold.
        """
        self.folds = None

    def fold(self, fold_idx: int, n_folds: int = 5) -> tuple['WineDatasetManager', 'WineDatasetManager']:
        """
        Divide il dataset in training e validation set per il fold corrente.
        
        Parameters:
            fold_idx (int): L'indice del fold corrente.
            n_folds (int): Numero totale di fold per la cross-validation (default 5).
        
        Returns:
            tuple: Due istanze di WineDatasetManager (train_set, valid_set).
        """
        if self.folds is None:
            # Calcola gli indici dei fold una sola volta
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=None)
            self.folds = list(skf.split(self.X, self.y))

        train_idx, valid_idx = self.folds[fold_idx]

        if isinstance(self.X, pd.Series) or isinstance(self.X, pd.DataFrame):
            # Se X è un pd.Series o un pd.DataFrame
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
        elif isinstance(self.X, csr_matrix):
            # Se X è una matrice sparsa
            X_train, X_valid = self.X[train_idx], self.X[valid_idx]
        else:
            # Nel caso X sia un numpy array o altro
            X_train, X_valid = self.X[train_idx], self.X[valid_idx]
        
        y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]

        return WineDatasetManager(X_train, y_train), WineDatasetManager(X_valid, y_valid)

    
    def split(self, test_size: float = 0.2, random_state: int = None) -> tuple['WineDatasetManager', 'WineDatasetManager']:
        """
        Divide il dataset in training e test set.
        
        Parameters:
            test_size (float): La proporzione del dataset da includere nel test set (default 0.2).
            random_state (int): Il seed per la riproducibilità (default None).
        
        Returns:
            tuple: Due istanze di WineDatasetManager (train_set, test_set).
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        return WineDatasetManager(X_train, y_train), WineDatasetManager(X_test, y_test)
    
    def get_dataset(self):
        """
        Restituisce il dataset come DataFrame.
        
        Returns:
            pd.DataFrame: Il dataset.
        """
        return pd.DataFrame({'description': self.X, 'variety': self.y})
    
    def classes(self):
        """
        Restituisce le classi uniche nel dataset.
        
        Returns:
            list: Le classi uniche.
        """
        return self.y.unique()
