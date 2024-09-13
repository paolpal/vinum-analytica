import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix

from .data_preprocessing import TextPreprocessor

class WineDatasetManager:
    def __init__(self, X: pd.Series = None, y: pd.Series = None, x_label: str = 'description', y_label: str = 'variety'):
        """
        Inizializza la classe con i dati opzionali X, y, x_label, e y_label.
        
        Parameters:
            X (pd.Series): Serie contenente le caratteristiche.
            y (pd.Series): Serie contenente le etichette.
            x_label (str): Nome della colonna per X (default 'description').
            y_label (str): Nome della colonna per y (default 'variety').
        """
        self.X = X
        self.y = y
        self.x_label = x_label
        self.y_label = y_label
        self.folds = None
        self.vectorizer = None
        self.scaler = None

    @classmethod
    def read_csv(cls, csv_path: str, x_label: str = 'description', y_label: str = 'variety'):
        """
        Carica i dati da un file CSV.

        Parameters:
            csv_path (str): Percorso al file CSV.
            x_label (str): Nome della colonna per X (default 'description').
            y_label (str): Nome della colonna per y (default 'variety').

        Returns:
            WineDatasetManager: Un'istanza di WineDatasetManager con i dati caricati.
        """
        df = pd.read_csv(csv_path)
        X = df[x_label]
        y = df[y_label]
        return cls(X, y, x_label, y_label)
    
    def save(self, csv_path: str):
        """
        Salva i dati in un file CSV.

        Parameters:
            csv_path (str): Percorso al file CSV.
        """
        assert isinstance(self.X, pd.Series), "X must be a pandas Series."
        df = pd.DataFrame({self.x_label: self.X, self.y_label: self.y})
        df.to_csv(csv_path, index=False)
    
    def load(self, csv_path: str):
        """
        Carica i dati da un file CSV.

        Parameters:
            csv_path (str): Percorso al file CSV.
        """
        df = pd.read_csv(csv_path)
        self.X = df[self.x_label]
        self.y = df[self.y_label]

    def get_x(self) -> pd.Series:
        """
        Restituisce le caratteristiche.

        Returns:
            pd.Series: Le caratteristiche.
        """
        return self.X
    
    def get_y(self) -> pd.Series:
        """
        Restituisce le etichette.

        Returns:
            pd.Series: Le etichette.
        """
        return self.y
    
    def preprocess(self):
        """
        Pre-elabora le caratteristiche.
        """
        preprocessor = TextPreprocessor()
        self.X = self.X.apply(preprocessor.preprocess)
    
    def vectorize(self, vectorizer: TfidfVectorizer | None = None) -> TfidfVectorizer:
        """
        Vettorizza le caratteristiche e aggiorna X con i dati vettorizzati.
        """
        if vectorizer is not None:
            self.vectorizer = vectorizer
            self.X = self.vectorizer.transform(self.X)
        else:
            self.vectorizer = TfidfVectorizer(strip_accents='ascii')
            self.X = self.vectorizer.fit_transform(self.X)
        return self.vectorizer
    
    def normalize(self, scaler : StandardScaler = None):
        """
        Normalizza le caratteristiche.
        Da usare se y è numerico [prezzo].
        """
        if scaler is None:
            self.scaler = StandardScaler()
            self.y = pd.Series(self.scaler.fit_transform(self.y.values.reshape(-1, 1)).flatten(), index=self.y.index)
        else:
            self.scaler = scaler
            self.y = pd.Series(self.scaler.fit_transform(self.y.values.reshape(-1, 1)).flatten(), index=self.y.index)
        return self.scaler

    def resample(self, max_samples : int = 200000):
        """
        Esegue il resample del training set usando SMOTE e RandomUnderSampler.
        """
        class_counts = self.y.value_counts()
        num_classes = len(class_counts)
        samples_per_class = max_samples // num_classes
        
        # Dizionari per oversampling e undersampling
        undersample_dict = {}
        oversample_dict = {}

        for cls, count in class_counts.items():
            if count > samples_per_class:
                # La classe ha più campioni di quelli desiderati, quindi undersample
                undersample_dict[cls] = samples_per_class
            elif count < samples_per_class:
                # La classe ha meno campioni di quelli desiderati, quindi oversample
                oversample_dict[cls] = samples_per_class

        # Esegui undersampling e oversampling
        if undersample_dict:
            self.undersample(undersample_dict)
        if oversample_dict:
            self.oversample(oversample_dict)

    def undersample(self, class_dict: dict = None):

        undersampler = RandomUnderSampler(sampling_strategy=class_dict)
        self.X, self.y = undersampler.fit_resample(self.X, self.y)

    def oversample(self, class_dict: dict = None):

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
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
        elif isinstance(self.X, csr_matrix):
            X_train, X_valid = self.X[train_idx], self.X[valid_idx]
        else:
            X_train, X_valid = self.X[train_idx], self.X[valid_idx]
        
        y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]

        return WineDatasetManager(X_train, y_train, self.x_label, self.y_label), WineDatasetManager(X_valid, y_valid, self.x_label, self.y_label)

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
        return WineDatasetManager(X_train, y_train, self.x_label, self.y_label), WineDatasetManager(X_test, y_test, self.x_label, self.y_label)
    
    def get_dataset(self):
        """
        Restituisce il dataset come DataFrame.
        
        Returns:
            pd.DataFrame: Il dataset.
        """
        return pd.DataFrame({self.x_label: self.X, self.y_label: self.y})
    
    def classes(self):
        """
        Restituisce le classi uniche nel dataset.
        
        Returns:
            list: Le classi uniche.
        """
        return self.y.unique()
    
    def sample(self, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Seleziona casualmente un campione di n osservazioni dal dataset.
        
        Parameters:
            n (int): Il numero di osservazioni da selezionare (default 1).
        
        Returns:
            tuple: Due array numpy (X, y).
        """
        idx = np.random.choice(len(self.X), n)
        X = self.X[idx]
        y = self.y[idx]
        return X, y
