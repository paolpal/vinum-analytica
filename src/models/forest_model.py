from sklearn.ensemble import RandomForestClassifier
from .model import Model
import pickle

class RandomForestModel(Model):
    def __init__(self, vectorizer = None, n_estimators: int = 100, criterion='gini', min_impurity_decrease: float = 0.0):
        """
        Inizializza il modello RandomForestClassifier con i parametri specificati.
        
        Parameters:
            n_estimators (int): Il numero di alberi nella foresta (default 100).
            max_depth (int): La profondità massima degli alberi (default None).
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, min_impurity_decrease=min_impurity_decrease, criterion=criterion, n_jobs=-1)
        self.model.set_params(class_weight='balanced')
        self.vectorizer = vectorizer
        self.hyperparameters = {
            'n_estimators': n_estimators,
            'criterion': criterion,
            'min_impurity_decrease': min_impurity_decrease
        }

    def train(self, train, **kwargs):
        """
        Allena il modello sui dati di addestramento.
        
        Parameters:
            train (Dataset): Un'istanza della classe Dataset che fornisce i dati di addestramento.
        """
        self.model.fit(train.get_x(), train.get_y())

    def predict(self, x, **kwargs):
        """
        Effettua previsioni sui dati di input.
        
        Parameters:
            x (list[str] | str): Dati di input per le previsioni.
        
        Returns:
            list[str] | str: Le previsioni del modello.
        """
        token = self.vectorizer.transform(x)
        return self.model.predict(token)
    
    def reset(self):
        """
        Reimposta il modello ai suoi parametri iniziali.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.model.n_estimators,
        )
        self.model.set_params(class_weight='balanced')
        self.vectorizer = None

    def save(self, path):
        """
        Salva il modello su disco.
        
        Parameters:
            path (str): Il percorso del file in cui salvare il modello.
        """
        with open(path, 'wb') as file:
            pickle.dump((self.model, self.vectorizer, self.hyperparameters), file)

    @classmethod
    def load(cls, path):
        """
        Carica un modello precedentemente salvato da un file.
        
        Parameters:
            path (str): Il percorso del file da cui caricare il modello.
        """
        with open(path, 'rb') as file:
            model, vectorizer, hyperparameters = pickle.load(file)
            rf = cls(vectorizer=vectorizer, **hyperparameters)
            rf.model = model
        return rf

    def classes(self):
        """
        Restituisce le classi del modello.
        
        Returns:
            list[str]: Le classi del modello.
        """
        return self.model.classes_
