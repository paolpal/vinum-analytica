import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..data.dataset import WineDatasetManager
from .model import Model

class NeuralNetworkModel(Model):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128, vectorizer: TfidfVectorizer=None, epochs: int = 10, lr: float = 0.001):
        """
        Inizializza la rete neurale con un livello nascosto e specifica il numero di neuroni per ogni livello.
        
        Parameters:
            input_size (int): La dimensione del vettore di input (numero di feature).
            hidden_size (int): Il numero di neuroni nel livello nascosto.
            output_size (int): Il numero di classi da predire.
            vectorizer: Il vettorizzatore da usare per trasformare i dati (opzionale).
            lr (float): Il tasso di apprendimento (default 0.001).
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )  # Utilizziamo la GPU se disponibile
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Livello di input a livello nascosto
            nn.ReLU(),                           # Funzione di attivazione ReLU
            nn.Linear(hidden_size, output_size), # Livello nascosto a livello di output
            nn.Softmax(dim=1)                    # Funzione di attivazione Softmax, normalizza le probabilità
        ).to(self.device)                        # Spostiamo il modello sulla GPU se disponibile
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # Funzione di perdita per classificazione multiclasse
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Ottimizzatore Adam
        self.vectorizer = vectorizer
        self.label_encoder = LabelEncoder()
        self.hyperparameters = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
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
        self.model.train()  # Mettiamo il modello in modalità di training

        # Vettorizziamo le etichette di training
        y_train = self.label_encoder.fit_transform(train.get_y().values)
        y_train = torch.tensor(y_train, dtype=torch.long, device=self.device)

        # Convertiamo i dati di training in formato sparso di SciPy
        X_train_sparse = train.get_x()

        disable_tqdm = kwargs.get('verbose', True) == False

        for epoch in range(epochs := self.hyperparameters['epochs']):
            permutation = np.random.permutation(X_train_sparse.shape[0])
            for i in tqdm(range(0, X_train_sparse.shape[0], batch_size), desc=f'Epoch {epoch+1}/{epochs}', disable=disable_tqdm):
                max_index = min(i + batch_size, X_train_sparse.shape[0])
                indices = permutation[i:max_index]
                batch_x_sparse = X_train_sparse[indices]
                batch_y = y_train[indices]

                # Convertiamo il batch sparso in un tensore denso per il training
                batch_x_dense = torch.tensor(batch_x_sparse.toarray(), dtype=torch.float32, device=self.device)

                self.optimizer.zero_grad()  # Azzeriamo i gradienti
                
                # Forward pass
                outputs = self.model(batch_x_dense)  # Passiamo il batch denso nel modello
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass e ottimizzazione
                loss.backward()
                self.optimizer.step()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')  # Logging della loss

    def predict(self, x, **kwargs):
        """
        Effettua previsioni sui dati di input.
        
        Parameters:
            x (list[str] | str): Dati di input per le previsioni.
        
        Returns:
            list[str] | str: Le previsioni del modello.
        """
        if isinstance(x, str):
            x = [x]
        self.model.eval()  # Mettiamo il modello in modalità di valutazione
        with torch.no_grad():  # Disabilitiamo il calcolo dei gradienti per efficienza
            token = self.vectorizer.transform(x)  # Vettorizziamo l'input
            inputs_sparse = token  # Tenere l'input in formato sparso
            inputs_dense = torch.tensor(inputs_sparse.toarray(), dtype=torch.float32, device=self.device)  # Convertiamo in denso
            outputs = self.model(inputs_dense)
            _, predicted = torch.max(outputs.data, 1)  # Prendiamo la classe con la probabilità più alta
            predicted = predicted.cpu()
        return self.label_encoder.inverse_transform(predicted.numpy())

    def reset(self):
        """
        Reimposta il modello ai suoi parametri iniziali.
        """
        # Ricostruiamo il modello da zero, mantenendo gli stessi parametri di rete e ottimizzatore
        self.model.apply(self._init_weights)  # Reinzializziamo i pesi della rete

    def _init_weights(self, layer):
        """
        Inizializza i pesi del layer specificato.
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def save(self, path):
        """
        Salva il modello su disco.
        
        Parameters:
            path (str): Il percorso del file in cui salvare il modello.
        """
        torch.save({
            'state_dict': self.model.state_dict(),
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'hyperparameters': self.hyperparameters
        }, path)

    @classmethod
    def load(cls, path) -> "NeuralNetworkModel":
        """
        Carica un modello precedentemente salvato da un file.
        
        Parameters:
            path (str): Il percorso del file da cui caricare il modello.
        """
        device = torch.device(
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        # Caricare tutto con torch.load e mappare su CPU o GPU
        checkpoint = torch.load(path, map_location=device)

        hyperparameters = checkpoint['hyperparameters']
        nn = cls(input_size=hyperparameters['input_size'], output_size=hyperparameters['output_size'], hidden_size=hyperparameters['hidden_size'], vectorizer=checkpoint['vectorizer'], lr=hyperparameters['lr'])
        
        # Caricare lo stato del modello
        nn.model.load_state_dict(checkpoint['state_dict'])
        nn.model.to(device)
        
        # Ripristinare altri attributi
        nn.label_encoder = checkpoint['label_encoder']
        nn.device = device  # Salva il dispositivo per futuri usi

        return nn

    def classes(self):
        """
        Restituisce le classi del modello. Non è direttamente applicabile alla rete neurale,
        quindi potrebbe essere personalizzata se necessario.
        """
        return self.label_encoder.classes_
