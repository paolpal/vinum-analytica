import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..data.dataset import WineDatasetManager
from .model import Model

class RegressorNeuralNetworkModel(Model):
    def __init__(self, input_size: int, vectorizer: TfidfVectorizer=None, epochs: int = 10, lr: float = 0.001):
        """
        Inizializza la rete neurale con un livello nascosto e specifica il numero di neuroni per ogni livello.
        
        Parameters:
            input_size (int): La dimensione del vettore di input (numero di feature).
            vectorizer: Il vettorizzatore da usare per trasformare i dati (opzionale).
            lr (float): Il tasso di apprendimento (default 0.001).
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )  # Utilizziamo la GPU se disponibile
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),  
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),    
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),     
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Linear(8, 1)        
        ).to(self.device)                        # Spostiamo il modello sulla GPU se disponibile
        self.criterion = nn.MSELoss()  # Funzione di perdita per regressione
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Ottimizzatore Adam
        self.vectorizer = vectorizer
        self.hyperparameters = {
            'input_size': input_size,
            'epochs': epochs,
            'lr': lr
        }

    def train(self, train: WineDatasetManager, batch_size: int = 32, **kwargs):
        """
        Addestra la rete neurale sui dati di training.
        
        Parameters:
            train (Dataset): Un'istanza della classe Dataset che fornisce i dati di addestramento.
            batch_size (int): Dimensione del batch per l'addestramento (default 32).
        """
        self.model.train()

        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_train_sparse = train.get_x()

        disable_tqdm = kwargs.get('verbose', True) == False

        for epoch in range(epochs := self.hyperparameters['epochs']):
            permutation = np.random.permutation(X_train_sparse.shape[0])
            for i in tqdm(range(0, X_train_sparse.shape[0], batch_size), desc=f'Epoch {epoch+1}/{epochs}', disable=disable_tqdm):
                max_index = min(i + batch_size, X_train_sparse.shape[0])
                indices = permutation[i:max_index]
                batch_x_sparse = X_train_sparse[indices]
                batch_y = y_train[indices]
                
                batch_x_dense = torch.tensor(batch_x_sparse.toarray(), dtype=torch.float32, device=self.device)

                self.optimizer.zero_grad()  # Azzeriamo i gradienti
                outputs = self.model(batch_x_dense)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            if not disable_tqdm:
                print(f'Loss: {loss.item()}')


    def predict(self, x, **kwargs):
        if isinstance(x, str):
            x = [x]
        self.model.eval()
        with torch.no_grad():
            token = self.vectorizer.transform(x)
            x = torch.tensor(token.toarray(), dtype=torch.float32, device=self.device)      
            outputs = self.model(x)
            return outputs.cpu().numpy()
        
    def save(self, path: str):
        with open(path, 'wb') as file:
            pickle.dump((self.model.state_dict(), self.vectorizer, self.hyperparameters), file)

    
    @classmethod
    def load(cls, path: str) -> 'RegressorNeuralNetworkModel':
        with open(path, 'rb') as file:
            state_dict, vectorizer, hyperparameters = pickle.load(file)
            regressor = cls(**hyperparameters)
            regressor.model.load_state_dict(state_dict)
            regressor.vectorizer = vectorizer
        return regressor
    
